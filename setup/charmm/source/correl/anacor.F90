module anacor_m
   implicit none

contains

SUBROUTINE ANACOR(X,Y,Z,XREF,YREF,ZREF,TQ,MXNTOT, &
     NTOT,NUNIT,NFU,NBEGN,NSTOP,NSKIP,DELTA,VELCD, &
     NSERIE,SNAME,STOT,ICLASS,GECOD,VECCOD,SERVAL, &
     LMASS,SAVEG,SFLCT,SERPT,SERNQ, &
     SSELEN,KSLCT,DOIMAGES, &
#if KEY_SHELL==1
     SHLLST,SHBLKL, &    
#endif
     QAT,DDV,ISLCT,ATOMIN,WMAIN,QORIENT,QMASS,JSLCT, &
     TBT)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE READS 'NUNIT' FILES BEGINNING 'NFU'
  !     CONTAINING DYNAMICS COORDINATES OR VELOCITIES AND
  !     AND CALCULATES THE QUANTITIES TO BE CORRELATED
  !
  !     X,Y,Z,WMAIN(NATOM) - Working coordinate sets and Weighting array
  !     XREF,YREF,ZREF(NATOM) - Reference coordinate set
  !     TQ(MNXTOT,NSERIE) - Time series values
  !     MXNTOT - Maximum number of time entries
  !     NTOT - Number of data points returned
  !     NUNIT,NFU,NBEGN,NSTOP,NSKIP,DELTA - Values for trajectory reading
  !     NSERIE - Number of time series to compute
  !     SNAME(NSERIE) - Name of series
  !     STOT(NSERIE) - number of data points in each time series.
  !                 Only those with zero will be filled with this routine.
  !     ICLASS(NSERIE) - class name for each series
  !     GECOD(NSERIE) - Flag specifying geometry as opposed to energy
  !     VECCOD(NSERIE) - Vector code
  !     SERVAL(NSERIE) - Series value (definition depends on ICLASS)
  !     LMASS(NSERIE) - Logical flag for mass weighting of terms
  !     SAVEG(NSERIE) - Average value for time series
  !     SFLCT(NSERIE) - Fluctuation value for each series
  !     SERPT(NSERIE) - Pointer to start of QAT atom descriptor
  !     SERNQ(NSERIE) - Number of entries to average over for each series
  !     QAT(*) - Atom descriptors for some series
  !     DDV(NAT3) - Eigenvectors
  !     ISLCT - First atom selection (used by the time series)
  !     ATOMIN - Scratch array used by READCV and ORINTC
  !     WMAIN  - The standard weighting array
  !     QORIENT - Frames will be best fit to the reference
  !     QMASS  - A mass weighting will be used in the best fit
  !     JSLCT  - The second atom selection, used for best fit atoms
  !     TBT    -  moment of inertia tensor values from previous frame
  !
  !
  !     Time series codes(ICLASS),  secondary code(GECOD)
  !        USER
  !        ENER
  !
  !        BOND
  !        ANGLE                   1 - GEOMETRY
  !        TORSION                 2 - ENERGY
  !        IMPROPER
  !
  !        HBOND                   1 - DISTance
  !                                2 - HANGle
  !                                3 - ENERgy
  !                                4 - AANGle
  !
  !                                1 - X
  !        ATOM                    2 - Y
  !        FLUCTUATION             3 - Z
  !        VECTOR                  4 - R
  !                                5 - DOTPRODUCT
  !        GYRATION                6 - X-CROSSPROD
  !        DENSITY                 7 - Y-CROSSPROD
  !        MODE                    8 - Z-CROSSPROD
  !        TEMPERATURE             9 - FDIM (i.e. W coordinate)
  !        TIME
  !        ZERO
  !        HELIX
  !        INERTIA
  !       
  !        SURF                    1 - ACCESSIBLE 
  !                                2 - CONTACT          
  !                                -1,-2 - as above, but w/ radii from WMAIN
  !
  !        RMS (WRT TO COMP)       1 - NO ORIENTATION
  !                                2 - ORIENT
  !
  !        PUCKER                  1 - PHASE
  !                                2 - AMPLITUDE
  !        DRMS
  !        SECS                    1 - FRACTION ALPHA
  !                                2 - FRACTION BETA
  !
  !      Original version: S. Swaminathan
  !      Rewritten: Bernard R. Brooks  9/84
  !
  use chm_kinds
  use chm_types
#if KEY_CHEQ==1
  use cheq,only:QCG           
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor      
#endif
  use dimens_fcm
  use exfunc
  use number
  ! we need this for images
  use bases_fcm
  use ctitla
  use consta
  use eintern
  use hbondm
  use image
  use image_routines_module, only: transo
  use imgup
  use param
  use param_store, only: set_param, get_param
  use psf
  use code
  use fourdm
  use deriv
  use cvio
  use select
  use secondary_structure,only:secstr
  use stream
  ! needed for TRANSO
#if KEY_FLUCQ==1
  use flucq         
#endif
#if KEY_SHELL==1
  use shell,only: qshell,qshlup,nashl,nshl,nshblk,shlupd     
#endif
  use corsubs,only:cdipole,orintc
#if KEY_PROTO==1
  use proto_mod,only:calprdi,calprve     
#endif
  use helix,only:helix1
  use intcor2,only:geticv
  use pucker_mod,only:pucka,pucker6cp
  use memory
  use chutil,only:getres,qhavcrd
  use machutil,only:die
  use surfacmod,only:surfac
  use usermod,only:usrtim
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec         
#endif
  implicit none

  integer,allocatable,dimension(:) :: FREEAT
  type stphi_type
     type(chm_array),dimension(5) :: s
     integer,dimension(:),allocatable :: iarr
  end type stphi_type
  type(stphi_type),allocatable,dimension(:) :: stphi
  integer :: locierr

  real(chm_real4),allocatable,dimension(:) :: TEMPa
  real(chm_real),allocatable,dimension(:),target :: matspace
  real(chm_real),pointer,dimension(:) :: XMAT,ymat,zmat,mat,scrr,wm
  real(chm_real) X(:),Y(:),Z(:),WMAIN(*),XREF(*),YREF(*),ZREF(*)
  INTEGER MXNTOT
  real(chm_real) TQ(MXNTOT,*)
  INTEGER NTOT,NUNIT,NFU,NBEGN,NSTOP,NSKIP
  real(chm_real) DELTA
  INTEGER VELCD
  INTEGER NSERIE,STOT(*)
  character(len=*) ICLASS(*)
  character(len=*) SNAME(*)
  INTEGER GECOD(*),VECCOD(*)
  LOGICAL LMASS(*)
  real(chm_real) SERVAL(*),SAVEG(*),SFLCT(*)
  INTEGER SERPT(*),SERNQ(*)
  INTEGER QAT(*),ISLCT(*),ATOMIN(2,*)
  real(chm_real) DDV(*)
  LOGICAL QORIENT,QMASS
  INTEGER JSLCT(*)
  real(chm_real) TBT(NSERIE,9)
  INTEGER SSELEN(*),KSLCT(*)
  LOGICAL DOIMAGES
  INTEGER OLDPRN
  !
  !
  ! would belong above this INCLUDE block but I need NATOM
#if KEY_SHELL==1
  INTEGER SHLLST(NATOM,NSHL),SHBLKL(*)
#endif 
  real(chm_real) XDIP, YDIP, ZDIP, QTOT
  INTEGER ILIST,II,NMISS
  LOGICAL QOXYZ
  INTEGER NAT,IUNIT,ISTATS
  INTEGER NFREAT,NFILE,ISTEP,NDEGF,NSAVV
  INTEGER ISERIE,NQ,IS,KNQ,I,IAT,IPT,JAT,IQ,J,JJ,JS,JQ
  INTEGER IATB,K,GECD,NDIG
  INTEGER NINIT
  INTEGER  IS2, NNQ, N1,NBPAIR
  real(chm_real) DTOT,TMASS,AX(3),R0(3)
  real(chm_real) QAVEGX,QFLCTX
  real(chm_real) RAT2,DELTA2
  real(chm_real) AVE,RADGYR,RCUT,RCUT2,VOL,DEN
  real(chm_real) AMS,SQAMS,XCM,YCM,ZCM
  real(chm_real) VAL,RDAT,GDAT
  real(chm_real) TVAL
  real(chm_real) TEMP,TEMPI,PHASE,AMPL
  real(chm_real) AMASST,RMST
  real(chm_real) XAF,YAF,ZAF,XAL,YAL,ZAL
  real(chm_real) XX,YY,ZZ,XY,XZ,YZ
  real(chm_real) XI,YI,ZI,XCI,YCI,ZCI,XIJ,YIJ,ZIJ,XCIJ,YCIJ,ZCIJ,RIJ,RCIJ
  real(chm_real) U(3,3),SCR(21),AMOM(6),EV(3),THETA,THETA2
  real(chm_real) P(3), S(3), T(3), TMP1,TMP2,TMP3
#if KEY_DIMS==1
  real(chm_real) OMSCORE                                    
#endif
  INTEGER NSLCT,NSLCT2,MODE,IRES0,IRES1
  INTEGER VECCD
  real(chm_real) ACCURACY
  LOGICAL LWEIG  
  !mu...20-Jul-93 add one line of declaration
  real(chm_real) AVETEMP, DIFF
  !
  LOGICAL MASSWT,QPRINT
  CHARACTER(len=4) ::  HDRC='COOR',HDRD='CORD',HDRV='VELD',HDR1,HDR2
  real(chm_real) Q_puc, theta_puc, phi_puc

  call chmalloc('anacor.src','ANACOR','FREEAT',NATOM,intg=FREEAT)
  call chmalloc('anacor.src','ANACOR','TEMPa',NATOM,cr4=TEMPa)
  call allocate_stphi

  N1 = 1

  HDR1=HDRC
  HDR2=HDRD
  IF(VELCD.EQ.1) THEN
     HDR1=HDRV
     HDR2=HDRV
  ENDIF
  DO I=1,NSERIE
     DO J = 1, 9
        TBT(I,J) = ZERO
     ENDDO
  ENDDO
  DO I=1,3
     AX(I)=ZERO
     R0(I)=ZERO
  ENDDO
  NAT=NATOM
  QPRINT=(PRNLEV.GT.5)
#if KEY_DIMS==1
  !     OMScore                               
#endif
#if KEY_DIMS==1
  OMSCORE=-1                            
#endif
  !
  !     some informational print out.
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,33)
33   FORMAT(' The following time series will be filled:')
     DO ISERIE=1,NSERIE
        IF(STOT(ISERIE).EQ.0) WRITE(OUTU,34) SNAME(ISERIE)
34      FORMAT(20X,A4)
     ENDDO
     IF(PRNLEV.GE.2) WRITE(OUTU,35)
35   FORMAT(/)
  ENDIF
  !
  !=======================================================================
  !     BEGIN READING FILES
  !
  NTOT=0
  IUNIT=NFU
  ISTATS=1
  loop200: do while(istats >= 0)  !200 CONTINUE
#if KEY_FOURD==1
     if (.not.allocated(fdim)) call allocate_fourd_ltm  
#endif
     CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                             & 
#endif
          TEMPa,NAT,FREEAT,NFREAT, &
          NFU,NUNIT,IUNIT,NFILE, &
          ISTEP,ISTATS,NDEGF,DELTA2, &    
          NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
#if KEY_FOURD==1
          TITLEA,NTITLA,.TRUE.,FDIM,.true.            & 
#endif
#if KEY_FOURD==0
          TITLEA,NTITLA,.FALSE., (/ ZERO /), .true.   & 
#endif
          )
     IF(NTOT.EQ.0) THEN
        DELTA=DELTA2*TIMFAC
        ! Check to see if DELTA should be rounded to nearest integer femtosecond.
        IF(ABS(DELTA-INT(THOSND*DELTA+HALF)/THOSND).LT.DELTA/FTHSND) &
             DELTA=INT(THOSND*DELTA+HALF)/THOSND

       IF(.NOT. QHAVCRD(NATOM,(/ -1 /),XREF) )THEN
         XREF(1:NATOM)=X(1:NATOM)
         YREF(1:NATOM)=Y(1:NATOM)
         ZREF(1:NATOM)=Z(1:NATOM)
         IF(PRNLEV > 1 ) WRITE(OUTU,*) &
          ' ***** WARNING ***** No reference coordinates present. USING FIRST FRAME!'
       ENDIF
     ENDIF
     ! Orient this frame if requested.
     IF(QORIENT) THEN
        CALL ORINTC(NATOM,X,Y,Z,XREF,YREF,ZREF,AMASS,QMASS, &
             .TRUE.,ATOMIN,JSLCT,.FALSE.,WMAIN, &
             .FALSE.,QPRINT)
     ENDIF
     !     UPDATE IMAGES...
     !     QUIET!
     IF(DOIMAGES) THEN
        !        Construct the coordinates.
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
        !        we don't want the output from UPIMAG -> reset PRNLEV
        OLDPRN=PRNLEV
        PRNLEV=1
        CALL UPIMAG0(X, Y, Z, WMAIN, 0)
        !        reset printlevel
        PRNLEV=OLDPRN
     ENDIF
#if KEY_SHELL==1 /*shell*/
     ! update shell if needed
     IF(QSHELL) THEN
        !        update shell
        QSHLUP=.FALSE.
        CALL SHLUPD
        IF(.NOT.QSHLUP) THEN
           CALL WRNDIE(-3,'<CORREL>', &
                'SHELL update failed.')
        ENDIF
     ENDIF
#endif /*        (shell)*/
     !-----------------------------------------------------------------------
     bigseriesloop: DO ISERIE=1,NSERIE
        stot_eq_0: IF(STOT(ISERIE).EQ.0) THEN
           !
           NQ=SERNQ(ISERIE)
           IS=SERPT(ISERIE)
           MASSWT=LMASS(ISERIE)
           GECD=GECOD(ISERIE)

           ! USER SUPPLIED FUNCTION
           IF(ICLASS(ISERIE).EQ.'USER') THEN
              CALL USRTIM(SERVAL(ISERIE),QAT(IS),NQ,NTOT, &
                   NATOM,X,Y,Z,XREF,YREF,ZREF,NSKIP,DELTA,TVAL,ISLCT)
              !.......................................................................
#if KEY_SHELL==1
              !     atom in shell GECD?
           ELSE IF(ICLASS(ISERIE).EQ.'SATM') THEN
              TVAL=0.0D0
              !        now look if atom X is in this shell
              DO II=1,NASHL(GECD)
                 IF(SHLLST(II,GECD).EQ.QAT(IS)) THEN
                    TVAL=1.0D0
                    GOTO 120
                 ENDIF
              ENDDO
120           CONTINUE
              !.......................................................................
#endif 
              ! MINIMUM DISTANCE VECTORS TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'VECM') THEN
              IF(GECD.EQ.1) THEN
                 IPT=IS
                 IAT=QAT(IPT)
                 JAT=QAT(IPT+1)
                 CALL VECMIN(IAT,JAT,TVAL)
              ELSE IF(GECD.EQ.2) THEN
                 call get_param('YVMI', tval)
              ELSE IF(GECD.EQ.3) THEN
                 call get_param('ZVMI', tval)
              ELSE
                 CALL DIE
              ENDIF
              !
              !.......................................................................
              ! DIPOLE MOMENT TIME SERIES
           ELSE IF(ICLASS(ISERIE).EQ.'DIPO') THEN
              IF(GECD.EQ.1) THEN
                 !           something slipped me here: what happens if the selection
                 !           contains NO atoms? -> x-component is correct but
                 !           y and zet get nonsense values ==> set x/y/zdip to 0
                 call set_param('XDIP', ZERO)
                 call set_param('YDIP', ZERO)
                 call set_param('ZDIP', ZERO)
                 !tr         this is the x component of the dipole moment
                 !tr         -> calculate the dipole moment (atoms are in QAT)
                 QOXYZ=.FALSE.
                 IF(SERVAL(ISERIE).GT.ZERO) QOXYZ=.TRUE.
                 CALL CDIPOLE(NATOM,X,Y,Z,CG,NQ,QAT(IS), &
                      XDIP,YDIP,ZDIP,QTOT,QOXYZ, &
                      MASSWT,AMASS)
                 TVAL=XDIP
              ELSE IF(GECD.EQ.2) THEN
                 !tr         calculation should already have been done by x-comp...
                 call get_param('YDIP', tval)
              ELSE IF(GECD.EQ.3) THEN
                 call get_param('ZDIP', tval)
              ELSE
                 !tr            CALL WRNDIE(0,'<ANACOR>',
                 !tr     1           'WRONG CODING IN DIPOLE SERIES - IGNORED')
                 !        wanted to do warndie but the other series all just call die...
                 CALL DIE
              ENDIF
              !.......................................................................
#if KEY_SHELL==1 /*shell*/
              ! SHELL DIPOLE MOMENT TIME SERIES
           ELSE IF(ICLASS(ISERIE).EQ.'SDIP') THEN
              J=SSELEN(ISERIE)
              IF(GECD.EQ.1) THEN
                 !           something slipped me here: what happens if this shell
                 !           contains NO atoms? -> x-component is correct but
                 !           y and zet get nonsense values ==> set x/y/zdip to 0
                 call set_param('XDIP', ZERO)
                 call set_param('YDIP', ZERO)
                 call set_param('ZDIP', ZERO)
                 !tr         this is the x component of the dipole moment
                 !tr         -> calculate the dipole moment
                 QOXYZ=.FALSE.
                 IF(SERVAL(ISERIE).GT.ZERO) QOXYZ=.TRUE.
                 IF(J.GT.0) THEN
                    !              a shell
                    ILIST=0
                    NMISS=0
                    DO II=1,NASHL(J)
                       IF(X(SHLLST(II,J)).EQ.ANUM)THEN
                          NMISS=NMISS+1
                       ELSE
                          ILIST=ILIST+1
                          KSLCT(ILIST)=SHLLST(II,J)
                       ENDIF
                    ENDDO
                    !tr            check if all coordinates were defined (only X, really...)
                    IF (NMISS.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,22) NMISS
22                  FORMAT(/' **** WARNING **** THERE WERE',I5, &
                         ' MISSING COORDINATES'/)
                    !TR            and calculate the dipole moment
                    CALL CDIPOLE(NATOM,X,Y,Z,CG,ILIST,KSLCT, &
                         XDIP,YDIP,ZDIP,QTOT,QOXYZ, &
                         MASSWT,AMASS)
                 ELSE IF(J.LT.0) THEN
                    !              bulk
                    CALL CDIPOLE(NATOM,X,Y,Z,CG,NSHBLK,SHBLKL, &
                         XDIP,YDIP,ZDIP,QTOT,QOXYZ, &
                         MASSWT,AMASS)
                 ELSE
                    !              we should NEVER get here -> should be checked in CORREL main
                    CALL WRNDIE(-3,'<CORREL>', &
                         'Wrong SELEN in serie.')
                 ENDIF
                 TVAL=XDIP
              ELSE IF(GECD.EQ.2) THEN
                 !tr         calculation should already have been done by x-comp...
                 call get_param('YDIP', tval)
              ELSE IF(GECD.EQ.3) THEN
                 call get_param('ZDIP', tval)
              ELSE IF(GECD.EQ.4) THEN
                 IF(J.GT.0) THEN
                    !              shell J
                    TVAL=NASHL(J)
                 ELSE
                    !              bulk
                    TVAL=NSHBLK
                 ENDIF
              ELSE
                 !tr            CALL WRNDIE(0,'<ANACOR>',
                 !tr     1           'WRONG CODING IN DIPOLE SERIES - IGNORED')
                 !        wanted to do warndie but the other series all just call die...
                 CALL DIE
              ENDIF
#endif /* (shell)*/
              !.......................................................................
#if KEY_PROTO==1 /*proto*/
              ! PROTOTYPE DIPOLE (SUM) SERIES
              ! 
           ELSE IF(ICLASS(ISERIE).EQ.'PRDI') THEN
              J=INT(SERVAL(ISERIE))
              IF(GECD.EQ.1) THEN
                 !           calculate once and reuse setmsr-values for y and z comp
                 !           zero values at beginning to allow for "empty" prototypes
                 call set_param('XDIP', ZERO)
                 call set_param('YDIP', ZERO)
                 call set_param('ZDIP', ZERO)
                 !
                 CALL CALPRDI(J,MASSWT)
                 call get_param('XDIP', tval)
              ELSE IF(GECD.EQ.2) THEN
                 call get_param('YDIP', tval)
              ELSE IF(GECD.EQ.3) THEN
                 call get_param('ZDIP', tval)
              ENDIF
              !
#endif /* (proto)*/
              !.......................................................................
#if KEY_PROTO==1 /*proto*/
              ! PROTTOYPE CENTER OF MASS VELOCITY/COOR (SUM) SERIES
              ! 
           ELSE IF(ICLASS(ISERIE).EQ.'PRVE') THEN
              J=INT(SERVAL(ISERIE))
              IF(GECD.EQ.1) THEN
                 !           calculate once and reuse setmsr-values for y and z comp
                 !           zero values at beginning to allow for "empty" prototypes
                 call set_param('XVCM', ZERO)
                 call set_param('YVCM', ZERO)
                 call set_param('ZVCM', ZERO)
                 !
                 CALL CALPRVE(J,MASSWT)
                 call get_param('XVCM', tval)
              ELSE IF(GECD.EQ.2) THEN
                 call get_param('YVCM', tval)
              ELSE IF(GECD.EQ.3) THEN
                 call get_param('ZVCM', tval)
              ENDIF
              !
#endif /* (proto)*/
              !.......................................................................
              ! ENERGY TIME SERIES
           ELSE IF(ICLASS(ISERIE).EQ.'ENER') THEN
              CALL NEXTE(X,Y,Z,TVAL,NTOT)
              !.......................................................................
              ! BOND TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'DIST') THEN
              IF(GECD.EQ.2) THEN
                 CALL EBOND(DTOT,QAT(IS),QAT(IS+NQ),QAT(IS+NQ*2), &
                      NQ,CBC,CBB,DX,DY,DZ, &
                      X,Y,Z,.FALSE.,(/ZERO/),-1,(/0/),(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

#if KEY_DOMDEC==1
                 if (q_domdec) call wrndie(-5,'<anacor>','EBOND not tested for DOMDEC')
#endif 

                 TVAL=DTOT/NQ
              ELSE
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1),0,0, &
                         .FALSE.,GDAT,0._chm_real,0._chm_real, &
                         0._chm_real,0._chm_real,X,Y,Z)
                    AVE=AVE+GDAT
                 ENDDO
                 TVAL=AVE/NQ
              ENDIF
              !.......................................................................
              ! BOND TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'BOND') THEN
              IF(GECD.EQ.2) THEN
                 CALL EBOND(DTOT,QAT(IS),QAT(IS+NQ),QAT(IS+NQ*2), &
                      NQ,CBC,CBB,DX,DY,DZ, &
                      X,Y,Z,.FALSE.,(/ZERO/),-1,(/0/),(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

#if KEY_DOMDEC==1
                 if (q_domdec) call wrndie(-5,'<anacor>','EBOND not tested for DOMDEC')
#endif 

                 TVAL=DTOT/NQ
              ELSE
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1),0,0, &
                         .FALSE.,GDAT,0._chm_real,0._chm_real, &
                         0._chm_real,0._chm_real,X,Y,Z)
                    AVE=AVE+GDAT
                 ENDDO
                 TVAL=AVE/NQ
              ENDIF
              !.......................................................................
              ! ANGLE TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'ANGL') THEN
              IF(GECD.EQ.2) THEN
                 CALL EANGLE(DTOT,QAT(IS),QAT(IS+NQ),QAT(IS+NQ*2), &
                      QAT(IS+NQ*3),NQ,CTC,CTB,DX,DY,DZ,X,Y,Z, &
                      .FALSE.,(/ZERO/),-1,(/0/),(/ZERO/),(/0/),.FALSE. &
                      )

                 TVAL=DTOT/NQ
              ELSE
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1), &
                         QAT(IS+KNQ+NQ+NQ-1),0,.FALSE.,RDAT,GDAT, &
                         0._chm_real,0._chm_real,0._chm_real,X,Y,Z)
                    AVE=AVE+GDAT
                 ENDDO
                 TVAL=AVE/NQ
              ENDIF
              !
              !.......................................................................
              ! DIHEDRAL TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'DIHE') THEN
              IF(GECD.EQ.2) THEN
                 CALL EPHI(DTOT,QAT(IS),QAT(IS+NQ),QAT(IS+NQ*2),QAT(IS+NQ*3), &
                      QAT(IS+NQ*4),NQ,CPC,CPD,CPB,CPCOS,CPSIN,DX,DY,DZ,X,Y,Z, &
                      .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),-1,(/0/),(/ZERO/),(/0/),.FALSE. &
                      )

                 TVAL=DTOT/NQ
              ELSE IF (SERVAL(ISERIE) .GE. 0.0) THEN
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1), &
                         QAT(IS+KNQ+NQ+NQ-1),QAT(IS+KNQ+3*NQ-1), &
                         .FALSE.,RDAT,RDAT,GDAT,RDAT,RDAT,X,Y,Z)
                    AVE=AVE+GDAT
                 ENDDO
                 TVAL=AVE/NQ
                 !
              ELSE
                 NINIT=0
                 IF(NTOT.EQ.0) THEN
                    NINIT=1
                    ! Set aside space for six NQ-length arrays (five real, one integer) 
                    ! needed in TRNPHI.

                    call allocate_stphi_arrays
                 ENDIF
                 NNQ = NQ
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1), &
                         QAT(IS+KNQ+NQ+NQ-1),QAT(IS+KNQ+3*NQ-1), &
                         .FALSE.,RDAT,RDAT, &
                         stphi(iserie)%s(1)%a(knq), &
                         RDAT,RDAT,X,Y,Z)
                    AVE=AVE+stphi(iserie)%s(1)%a(knq)
                 ENDDO
                 TVAL=AVE/NQ
                 CALL TRNPHI(stphi(iserie)%s(1),stphi(iserie)%s(2), &
                      stphi(iserie)%s(3),stphi(iserie)%s(4), &
                      stphi(iserie)%s(5), &
                      CPB, &
                      CPD,ATYPE,RES,IBASE,QAT(IS),QAT(IS+NQ), &
                      QAT(IS+NQ*2),QAT(IS+NQ*3),QAT(IS+NQ*4), &
                      NRES,stphi(iserie)%iarr,  &
                      NQ,ISTEP,NINIT,OUTU,ISTATS,0)
                 ! *** End of Modification
              ENDIF
              !
              !.......................................................................
              ! IMPROPER DIHEDRAL TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'IMPR') THEN
              IF(GECD.EQ.2) THEN
                 CALL EPHI(DTOT,QAT(IS),QAT(IS+NQ),QAT(IS+NQ*2),QAT(IS+NQ*3), &
                      QAT(IS+NQ*4),NQ,CIC,CID,CIB,CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
                      .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),-1,(/0/),(/ZERO/),(/0/),.FALSE. &
                      )

                 TVAL=DTOT/NQ
              ELSE
                 AVE=ZERO
                 DO KNQ=1,NQ
                    CALL GETICV(QAT(IS+KNQ-1),QAT(IS+KNQ+NQ-1), &
                         QAT(IS+KNQ+NQ+NQ-1),QAT(IS+KNQ+3*NQ-1), &
                         .FALSE.,RDAT,RDAT,GDAT,RDAT,RDAT,X,Y,Z)
                    AVE=AVE+GDAT
                 ENDDO
                 TVAL=AVE/NQ
              ENDIF
              !
              !.......................................................................
              ! HYDROGEN BOND TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'HBON') THEN
              CALL EHBOND(DTOT,QAT(IS),QAT(IS+2),QAT(IS+1),QAT(IS+3), &
                   QAT(IS+4),1,CHBA,CHBB,0,0,0,X,Y,Z,.FALSE., &
                   0,GECD,TVAL,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA, &
                   HBEXPN,0,0,.FALSE.)
              !
              !....................................................................
#if KEY_CHEQ==1
              ! CHARGE VELOCITY TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'VCHA') THEN
              AVE=ZERO
              TMASS=ZERO
              IPT=IS
              DO I=1,NQ
                 IAT=QAT(IPT)
                 IPT=IPT+1
                 AMS=ONE
                 IF (MASSWT) AMS=AMASS(IAT)
                 VAL=CG(IAT)
                 AVE=AVE+VAL*AMS
                 TMASS=TMASS+AMS
              ENDDO
              AVE=AVE/TMASS
              TVAL=AVE
              !
#endif 
              !...............................................................
              !.......................................................................
              ! ATOM TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'ATOM') THEN
              IF (GECD.LT.5 .OR. GECD.EQ.9 .OR. GECD.EQ.10) THEN
                 AVE=ZERO
                 TMASS=ZERO
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF (MASSWT) AMS=AMASS(IAT)
                    IF(GECD.EQ.1) THEN
                       VAL=X(IAT)
                    ELSE IF(GECD.EQ.2) THEN
                       VAL=Y(IAT)
                    ELSE IF(GECD.EQ.3) THEN
                       VAL=Z(IAT)
                    ELSE IF(GECD.EQ.9) THEN
#if KEY_FOURD==1
                       VAL=FDIM(IAT)     
#endif
#if KEY_FOURD==0
                       VAL=ZERO          
#endif
                    ELSE IF(GECD.EQ.10) THEN
#if KEY_FLUCQ==1
                       VAL=CG(IAT)       
#endif
#if KEY_FLUCQ==0
                       VAL=ZERO          
#endif
                    ELSE IF(GECD.EQ.4) THEN
                       VAL=SQRT(X(IAT)**2+Y(IAT)**2+Z(IAT)**2)
                    ELSE
                       CALL DIE
                    ENDIF
                    AVE=AVE+VAL*AMS
                    TMASS=TMASS+AMS
                 ENDDO
                 AVE=AVE/TMASS
                 TVAL=AVE
              ELSE
                 ! Atom dotproducts or crossproducts
                 XAF=0.0
                 YAF=0.0
                 ZAF=0.0
                 XAL=0.0
                 YAL=0.0
                 ZAL=0.0
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(IAT)
                    XAF=XAF+X(IAT)*AMS
                    YAF=YAF+Y(IAT)*AMS
                    ZAF=ZAF+Z(IAT)*AMS
                    JAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(JAT)
                    XAL=XAL+X(JAT)*AMS
                    YAL=YAL+Y(JAT)*AMS
                    ZAL=ZAL+Z(JAT)*AMS
                 ENDDO
                 !
                 TMASS=SQRT(XAF*XAF+YAF*YAF+ZAF*ZAF)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAF=XAF*TMASS
                 YAF=YAF*TMASS
                 ZAF=ZAF*TMASS
                 TMASS=SQRT(XAL*XAL+YAL*YAL+ZAL*ZAL)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAL=XAL*TMASS
                 YAL=YAL*TMASS
                 ZAL=ZAL*TMASS
                 IF(GECD.EQ.5) THEN
                    VAL=XAF*XAL+YAF*YAL+ZAF*ZAL
                 ELSE IF(GECD.EQ.6) THEN
                    VAL=YAF*ZAL-YAL*ZAF
                 ELSE IF(GECD.EQ.7) THEN
                    VAL=ZAF*XAL-ZAL*XAF
                 ELSE IF(GECD.EQ.8) THEN
                    VAL=XAF*YAL-XAL*YAF
                 ELSE
                    CALL DIE
                 ENDIF
                 TVAL=VAL
              ENDIF
              !
              !.......................................................................
              ! FLUCTUATIONS TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'FLUC') THEN
              IF(GECD.LT.5 .OR. GECD.EQ.9 .OR. GECD.EQ.10) THEN
                 AVE=ZERO
                 TMASS=ZERO
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(IAT)
                    IF(GECD.EQ.1) THEN
                       VAL=X(IAT)-XREF(IAT)
                    ELSE IF(GECD.EQ.2) THEN
                       VAL=Y(IAT)-YREF(IAT)
                    ELSE IF(GECD.EQ.3) THEN
                       VAL=Z(IAT)-ZREF(IAT)
                    ELSE IF(GECD.EQ.9) THEN
#if KEY_FOURD==1
                       VAL=FDIM(IAT)-FDCOMP(IAT)    
#endif
#if KEY_FOURD==0
                       VAL=ZERO                     
#endif
                       ! N.B. No FLUCQ fluctuation code yet
                    ELSE IF(GECD.EQ.10) THEN
                       VAL=ZERO
                    ELSE IF(GECD.EQ.4) THEN
                       VAL=SQRT((X(IAT)-XREF(IAT))**2+(Y(IAT)-YREF(IAT))**2+ &
                            (Z(IAT)-ZREF(IAT))**2)
                    ELSE
                       CALL DIE
                    ENDIF
                    AVE=AVE+VAL*AMS
                    TMASS=TMASS+AMS
                 ENDDO
                 AVE=AVE/TMASS
                 TVAL=AVE
              ELSE
                 ! Atom dotproducts or crossproducts
                 XAF=0.0
                 YAF=0.0
                 ZAF=0.0
                 XAL=0.0
                 YAL=0.0
                 ZAL=0.0
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(IAT)
                    XAF=XAF+(X(IAT)-XREF(IAT))*AMS
                    YAF=YAF+(Y(IAT)-YREF(IAT))*AMS
                    ZAF=ZAF+(Z(IAT)-ZREF(IAT))*AMS
                    JAT=QAT(IPT)
                    IPT=IPT+1
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(JAT)
                    XAL=XAL+(X(JAT)-XREF(JAT))*AMS
                    YAL=YAL+(Y(JAT)-YREF(JAT))*AMS
                    ZAL=ZAL+(Z(JAT)-ZREF(JAT))*AMS
                 ENDDO
                 !
                 TMASS=SQRT(XAF*XAF+YAF*YAF+ZAF*ZAF)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAF=XAF*TMASS
                 YAF=YAF*TMASS
                 ZAF=ZAF*TMASS
                 TMASS=SQRT(XAL*XAL+YAL*YAL+ZAL*ZAL)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAL=XAL*TMASS
                 YAL=YAL*TMASS
                 ZAL=ZAL*TMASS
                 IF(GECD.EQ.5) THEN
                    VAL=XAF*XAL+YAF*YAL+ZAF*ZAL
                 ELSE IF(GECD.EQ.6) THEN
                    VAL=YAF*ZAL-YAL*ZAF
                 ELSE IF(GECD.EQ.7) THEN
                    VAL=ZAF*XAL-ZAL*XAF
                 ELSE IF(GECD.EQ.8) THEN
                    VAL=XAF*YAL-XAL*YAF
                 ELSE
                    CALL DIE
                 ENDIF
                 TVAL=VAL
              ENDIF
              !
              !.......................................................................
              ! VECTORS TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'VECT') THEN
              IF(GECD.LT.5 .OR. GECD.EQ.9 .OR. GECD.EQ.10) THEN
                 AVE=ZERO
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    JAT=QAT(IPT+1)
                    IPT=IPT+2
                    IF(GECD.EQ.1) THEN
                       VAL=X(IAT)-X(JAT)
                    ELSE IF(GECD.EQ.2) THEN
                       VAL=Y(IAT)-Y(JAT)
                    ELSE IF(GECD.EQ.3) THEN
                       VAL=Z(IAT)-Z(JAT)
                    ELSE IF(GECD.EQ.9) THEN
#if KEY_FOURD==1
                       VAL=FDIM(IAT)-FDIM(JAT)       
#endif
#if KEY_FOURD==0
                       VAL=ZERO                      
#endif
                    ELSE IF(GECD.EQ.10) THEN
#if KEY_FLUCQ==1
                       VAL=CG(IAT)-CG(JAT)           
#endif
#if KEY_FLUCQ==0
                       VAL=ZERO                      
#endif
                    ELSE IF(GECD.EQ.4) THEN
                       VAL=SQRT((X(IAT)-X(JAT))**2+(Y(IAT)-Y(JAT))**2+ &
                            (Z(IAT)-Z(JAT))**2)
                    ELSE
                       CALL DIE
                    ENDIF
                    AVE=AVE+VAL
                 ENDDO
                 AVE=AVE/NQ
                 TVAL=AVE
              ELSE
                 ! Atom dotproducts or crossproducts
                 XAF=0.0
                 YAF=0.0
                 ZAF=0.0
                 XAL=0.0
                 YAL=0.0
                 ZAL=0.0
                 IPT=IS
                 DO I=1,NQ
                    IAT=QAT(IPT)
                    JAT=QAT(IPT+1)
                    IPT=IPT+2
                    XAF=XAF+(X(IAT)-X(JAT))
                    YAF=YAF+(Y(IAT)-Y(JAT))
                    ZAF=ZAF+(Z(IAT)-Z(JAT))
                    IAT=QAT(IPT)
                    JAT=QAT(IPT+1)
                    IPT=IPT+2
                    XAL=XAL+(X(IAT)-X(JAT))
                    YAL=YAL+(Y(IAT)-Y(JAT))
                    ZAL=ZAL+(Z(IAT)-Z(JAT))
                 ENDDO
                 !
                 TMASS=SQRT(XAF*XAF+YAF*YAF+ZAF*ZAF)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAF=XAF*TMASS
                 YAF=YAF*TMASS
                 ZAF=ZAF*TMASS
                 TMASS=SQRT(XAL*XAL+YAL*YAL+ZAL*ZAL)
                 IF(TMASS.GT.0.0) TMASS=1.0/TMASS
                 XAL=XAL*TMASS
                 YAL=YAL*TMASS
                 ZAL=ZAL*TMASS
                 IF(GECD.EQ.5) THEN
                    VAL=XAF*XAL+YAF*YAL+ZAF*ZAL
                 ELSE IF(GECD.EQ.6) THEN
                    VAL=YAF*ZAL-YAL*ZAF
                 ELSE IF(GECD.EQ.7) THEN
                    VAL=ZAF*XAL-ZAL*XAF
                 ELSE IF(GECD.EQ.8) THEN
                    VAL=XAF*YAL-XAL*YAF
                 ELSE
                    CALL DIE
                 ENDIF
                 TVAL=VAL
              ENDIF
              !
              !.......................................................................
              ! RADGYR TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'GYRA') THEN
              RCUT=SERVAL(ISERIE)
              RCUT2=RCUT*RCUT
              TMASS=ZERO
              RADGYR=ZERO
              ! find center of mass:
              XCM=ZERO
              YCM=ZERO
              ZCM=ZERO
              DO IAT=1,NAT
                 IF(ISLCT(IAT).NE.0) THEN
                    AMS=ONE
                    IF(MASSWT) AMS=AMASS(IAT)
                    TMASS=TMASS+AMS
                    XCM=X(IAT)*AMS+XCM
                    YCM=Y(IAT)*AMS+YCM
                    ZCM=Z(IAT)*AMS+ZCM
                 ENDIF
              ENDDO
              IF(TMASS .GT. ZERO) THEN
                 XCM=XCM/TMASS
                 YCM=YCM/TMASS
                 ZCM=ZCM/TMASS
              ENDIF
              DO IAT=1,NAT
                 IF(ISLCT(IAT).NE.0) THEN
                    RAT2=(X(IAT)-XCM)**2 +(Y(IAT)-YCM)**2+(Z(IAT)-ZCM)**2
                    IF (RAT2.LE.RCUT2) THEN
                       AMS=ONE
                       IF(MASSWT) AMS=AMASS(IAT)
                       RADGYR=RADGYR+RAT2*AMS
                    ENDIF
                 ENDIF
              ENDDO
              TVAL=ZERO
              IF(TMASS.GT.ZERO) TVAL=SQRT(RADGYR/TMASS)
              !.......................................................................
              ! UNIT CELL VALUES; REQUIRES CRYSTAL SETUP
           ELSE IF(ICLASS(ISERIE).EQ.'CELL') THEN
              IF (SERVAL(ISERIE).LT.ONE) THEN
                 CALL XTLLAT(XUCELL,XTLABC)
                 TVAL = XUCELL(GECD)
              ELSE
                 TVAL = XTLABC(GECD)
              ENDIF
              !.......................................................................
              ! INERTIA TENSOR TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'INER') THEN
              KNQ = 0
              IF (SERVAL(ISERIE).GT.ZERO) KNQ = 1
              IF(GECD.EQ.1) THEN
                 ! find center of mass of atom selection
                 XCM=ZERO
                 YCM=ZERO
                 ZCM=ZERO
                 TMASS=ZERO
                 IQ = IS + NQ -1
                 DO J=IS,IQ
                    IAT=QAT(J)
                    AMS=AMASS(IAT)
                    TMASS=TMASS+AMS
                    XCM=X(IAT)*AMS+XCM
                    YCM=Y(IAT)*AMS+YCM
                    ZCM=Z(IAT)*AMS+ZCM
                 ENDDO
                 !
                 IF(TMASS .GT. ZERO) THEN
                    XCM=XCM/TMASS
                    YCM=YCM/TMASS
                    ZCM=ZCM/TMASS
                 ENDIF
                 !  calculate the moments of the atom selection
                 XX=ZERO
                 XY=ZERO
                 XZ=ZERO
                 YY=ZERO
                 YZ=ZERO
                 ZZ=ZERO
                 DO J=IS,IQ
                    IAT=QAT(J) 
                    AMS=AMASS(IAT)
                    XX=XX + AMS*(X(IAT)-XCM)*(X(IAT)-XCM)
                    XY=XY + AMS*(X(IAT)-XCM)*(Y(IAT)-YCM)
                    XZ=XZ + AMS*(X(IAT)-XCM)*(Z(IAT)-ZCM)
                    YY=YY + AMS*(Y(IAT)-YCM)*(Y(IAT)-YCM)
                    YZ=YZ + AMS*(Y(IAT)-YCM)*(Z(IAT)-ZCM)
                    ZZ=ZZ + AMS*(Z(IAT)-ZCM)*(Z(IAT)-ZCM)
                 ENDDO
                 !
                 TMASS = -XX - YY - ZZ
                 AMOM(1)=-XX - TMASS
                 AMOM(2)=-XY
                 AMOM(3)=-XZ
                 AMOM(4)=-YY - TMASS
                 AMOM(5)=-YZ
                 AMOM(6)=-ZZ - TMASS
                 !  diagonalize the matrix
                 CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
                      SCR(16),SCR(19),SCR(1),0)
                 !  smallest eigenvalue,EV(1), is principal inertia vector,u(*,1)
                 !  secondary vector is u(*,2), tertiary is u(*,3)
                 !
                 THETA=U(1,1)*TBT(ISERIE,1)+U(2,1)*TBT(ISERIE,2) &
                      +U(3,1)*TBT(ISERIE,3)
                 !   make sure there were no 180 degree flips of the principal axis 
                 IF(THETA.LT.ZERO) THEN
                    U(1,1) = -U(1,1)
                    U(2,1) = -U(2,1)
                    U(3,1) = -U(3,1)
                 ENDIF
                 !
                 TBT(ISERIE,1) = U(1,1)
                 TBT(ISERIE,2) = U(2,1)
                 TBT(ISERIE,3) = U(3,1)

                 !
                 IF (KNQ.EQ.0) THEN
                    TVAL=U(1,1) 
                 ELSE
                    TVAL=EV(1)
                 ENDIF
                 !
              ELSE IF(GECD.EQ.2) THEN
                 IF (KNQ.EQ.0) THEN
                    TVAL=U(2,1) 
                 ELSE
                    TVAL=EV(2)
                 ENDIF
              ELSE IF(GECD.EQ.3) THEN
                 IF (KNQ.EQ.0) THEN
                    TVAL=U(3,1) 
                 ELSE
                    TVAL=EV(3)
                 ENDIF
                 !Lni added the two other vectors (keyword ALL)
              ELSE IF(GECD.EQ.4) THEN
                 THETA=U(1,2)*TBT(ISERIE,4)+U(2,2)*TBT(ISERIE,5) &
                      +U(3,2)*TBT(ISERIE,6)
                 TMP1=TBT(ISERIE,4)**2+TBT(ISERIE,5)**2+TBT(ISERIE,6)**2
                 IF(TMP1.GT. 0.0) THEN
                    THETA=THETA/TMP1
                 ELSE
                    THETA=0.0
                 ENDIF
                 THETA2=U(1,2)*TBT(ISERIE,7)+U(2,2)*TBT(ISERIE,8) &
                      +U(3,2)*TBT(ISERIE,9)
                 TMP2=TBT(ISERIE,7)**2+TBT(ISERIE,8)**2+TBT(ISERIE,9)**2 
                 IF(TMP2.GT. 0.0)THEN
                    THETA2=THETA2/TMP2
                 ELSE
                    THETA2=0.0
                 ENDIF
                 ! Check for V2 <-> V3 swapping
                 IF( ( ABS(THETA) .LT. ABS(THETA2) ) &
                      .AND. TMP1 .GT.0.0 )THEN
                    ! OK, so swap...
                    IF(PRNLEV.GE.7)THEN
                       WRITE(OUTU,*) ' ANACOR-INERTIA> SWAPPING V2 <-> V3'
                    ENDIF
                    TMP1 = U(1,2)
                    TMP2 = U(2,2)
                    TMP3 = U(3,2)
                    U(1,2) = U(1,3)
                    U(2,2) = U(2,3)
                    U(3,2) = U(3,3)
                    U(1,3) = TMP1
                    U(2,3) = TMP2
                    U(3,3) = TMP3
                    THETA=THETA2
                 ENDIF
                 ! Check for 180 degree flip of second vector
                 IF(THETA.LT.ZERO) THEN
                    U(1,2) = -U(1,2)
                    U(2,2) = -U(2,2)
                    U(3,2) = -U(3,2)
                 ENDIF
                 TVAL = U(1,2)

                 TBT(ISERIE,4) = U(1,2)
                 TBT(ISERIE,5) = U(2,2)
                 TBT(ISERIE,6) = U(3,2)
              ELSE IF(GECD.EQ.5) THEN
                 TVAL = U(2,2)   
              ELSE IF(GECD.EQ.6) THEN
                 TVAL = U(3,2)  
              ELSE IF(GECD.EQ.7) THEN
                 ! Check for 180 degree flip of third vector
                 THETA=U(1,3)*TBT(ISERIE,7)+U(2,3)*TBT(ISERIE,8) &
                      +U(3,3)*TBT(ISERIE,9)
                 IF(THETA.LT.ZERO) THEN
                    U(1,3) = -U(1,3)
                    U(2,3) = -U(2,3)
                    U(3,3) = -U(3,3)
                 ENDIF
                 TVAL = U(1,3)

                 TBT(ISERIE,7) = U(1,3)
                 TBT(ISERIE,8) = U(2,3)
                 TBT(ISERIE,9) = U(3,3)
              ELSE IF(GECD.EQ.8) THEN
                 TVAL = U(2,3)   
              ELSE IF(GECD.EQ.9) THEN
                 TVAL = U(3,3)   
              ENDIF
              !.......................................................................
              ! DENSITY TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'DENS') THEN
              RCUT=SERVAL(ISERIE)
              RCUT2=RCUT*RCUT
              VOL=RCUT*RCUT2*FOUR*PI/THREE
              DEN=ZERO
              DO IAT=1,NAT
                 IF(ISLCT(IAT).NE.0) THEN
                    RAT2=X(IAT)*X(IAT)+Y(IAT)*Y(IAT)+Z(IAT)*Z(IAT)
                    IF(RAT2.LE.RCUT2) DEN=DEN+ONE
                 ENDIF
              ENDDO
              DEN=DEN/VOL
              TVAL=DEN
              !
              !.......................................................................
              ! HELIX TIMESERIE
           ELSE IF(ICLASS(ISERIE).EQ.'HELI') THEN
              IF(GECD.EQ.1) THEN
                 CALL HELIX1(AX,R0,NDIG)
                 TVAL=AX(1)
              ELSE IF(GECD.EQ.2) THEN
                 TVAL=AX(2)
              ELSE IF(GECD.EQ.3) THEN
                 TVAL=AX(3)
              ENDIF
              !
              !.......................................................................
              ! RMS TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'RMS') THEN
              IF(GECD.EQ.1) THEN
                 AMASST=ZERO
                 RMST=ZERO
                 DO IAT=1,NAT
                    IF(ISLCT(IAT).NE.0) THEN
                       AMS=ONE
                       IF(MASSWT) AMS=AMASS(IAT)
                       AMASST=AMASST+AMS
                       RMST=RMST+((X(IAT)-XREF(IAT))**2 + &
                            (Y(IAT)-YREF(IAT))**2 + &
                            (Z(IAT)-ZREF(IAT))**2)*AMS
                    ENDIF
                 ENDDO
                 TVAL=ZERO
                 IF(AMASST.GT.ZERO) TVAL=SQRT(RMST/AMASST)
              ELSE IF(GECD.EQ.2) THEN
                 CALL ORINTC(NATOM,X,Y,Z,XREF,YREF,ZREF,AMASS,MASSWT, &
                      .TRUE.,ATOMIN,ISLCT,.FALSE.,WMAIN, &
                      .FALSE.,QPRINT)
                 call get_param('RMS ', tval)
              ENDIF
              !
              !.......................................................................
              ! DRMS TIMESERIES - inter atom distance rms vs ref coordinates
           ELSE IF(ICLASS(ISERIE).EQ.'DRMS') THEN
             NBPAIR=0
             TVAL=ZERO
             NSLCT2=INT(SERVAL(ISERIE))
             IF(NSLCT2 == -1)THEN
               ! use all atoms for both selections
               DO I=1,NATOM
                 XI=X(I)
                 YI=Y(I)
                 ZI=Z(I)
                 XCI=XREF(I)
                 YCI=YREF(I)
                 ZCI=ZREF(I)
                 DO J=I+1,NATOM
                   NBPAIR=NBPAIR+1
                   XIJ=(XI-X(J))**2
                   YIJ=(YI-Y(J))**2
                   ZIJ=(ZI-Z(J))**2
                   XCIJ=(XCI-XREF(J))**2
                   YCIJ=(YCI-YREF(J))**2
                   ZCIJ=(ZCI-ZREF(J))**2
                   RIJ=SQRT(XIJ+YIJ+ZIJ)
                   RCIJ=SQRT(XCIJ+YCIJ+ZCIJ)
                   TVAL=TVAL+(RIJ-RCIJ)**2
                 ENDDO
               ENDDO
             ELSE IF(NSLCT2 >= 0) THEN
               ! use same QAT pointer list for both selections (NSLCT2==0),
               ! or different parts (NSLCT2> 0)
               IF(NSLCT2 == 0)THEN
                 IQ=IS+NQ-1
                 JQ=IQ
               ELSE
                 IQ=IS+NQ-NSLCT2-1
                 JS=IQ+1
                 JQ=JS+NSLCT2-1
               ENDIF
               DO I=IS,IQ
                 II=QAT(I)
                 XI=X(II)
                 YI=Y(II)
                 ZI=Z(II)
                 XCI=XREF(II)
                 YCI=YREF(II)
                 ZCI=ZREF(II)
                 IF(NSLCT2 == 0) JS=I+1
                 DO J=JS,JQ
                   JJ=QAT(J)
                   IF( JJ /= II)THEN ! Exclude the trivial 0-distance to self
                     NBPAIR=NBPAIR+1
                     XIJ=(XI-X(JJ))**2
                     YIJ=(YI-Y(JJ))**2
                     ZIJ=(ZI-Z(JJ))**2
                     XCIJ=(XCI-XREF(JJ))**2
                     YCIJ=(YCI-YREF(JJ))**2
                     ZCIJ=(ZCI-ZREF(JJ))**2
                     RIJ=SQRT(XIJ+YIJ+ZIJ)
                     RCIJ=SQRT(XCIJ+YCIJ+ZCIJ)
                     TVAL=TVAL+(RIJ-RCIJ)**2
                   ENDIF
                 ENDDO
               ENDDO
             ELSE  ! NSLCT2 < -1, we should never get here
               CALL WRNDIE(-5,'<ANACOR>','Internal error')
             ENDIF 
             TVAL=SQRT(TVAL/NBPAIR)
       !-----------------------------------------------------------------------
       ! SECS TIMESERIES - fraction alpha and beta in isel within jsel
       ! from the two TRAJ selections

           ELSE IF(ICLASS(ISERIE).EQ.'SECS') THEN
             TVAL=MINONE
             IF(GECD == 1) THEN
                OLDPRN=PRNLEV
                IF(PRNLEV < 5) PRNLEV=0
                CALL SECSTR(WMAIN,ISLCT,JSLCT,0)
                PRNLEV=OLDPRN
                call get_param('ALPHA', tval)
             ELSE IF(GECD == 2) THEN
                call get_param('BETA', tval)
             ELSE IF(GECD == 3) THEN
                call get_param('H310', tval)
             ELSE IF(GECD == 4) THEN
                call get_param('HPI', tval)
             ELSE
                CALL WRNDIE(-1,'<ANACOR>','SECS: Illegal GECD')
             ENDIF
       !-----------------------------------------------------------------------
#if KEY_DIMS==1 /*omscore*/
              !......................................................................
              ! Onsager-Machlup Timeseries
           ELSE IF(ICLASS(ISERIE).EQ.'OMSC') THEN
              IF(NTOT.EQ.0) THEN
                 OMSCORE=ZERO
                 TVAL=ZERO
                 DO IAT=1,NAT
                    IF(ISLCT(IAT).NE.0) THEN
                       XREF(IAT)=X(IAT)
                       YREF(IAT)=Y(IAT)
                       ZREF(IAT)=Z(IAT)
                    ENDIF
                 ENDDO
              ELSE
                 OLDPRN=PRNLEV
                 PRNLEV=0
                 CALL NEXTE(X,Y,Z,OMSCORE,NTOT)
                 PRNLEV=OLDPRN
                 OMSCORE=ZERO
                 DO IAT=1,NAT
                    IF(ISLCT(IAT).NE.0) THEN
                       !     dX = Xi - Xi-1
                       XREF(IAT)=X(IAT)-XREF(IAT)
                       YREF(IAT)=Y(IAT)-YREF(IAT)               
                       ZREF(IAT)=Z(IAT)-ZREF(IAT)
                       !     n*MdX/dt
                       XREF(IAT)=(AMASS(IAT)*XREF(IAT)) &
                            *(SERVAL(ISERIE)/(DELTA*NSKIP))
                       YREF(IAT)=(AMASS(IAT)*YREF(IAT)) &
                            *(SERVAL(ISERIE)/(DELTA*NSKIP))
                       ZREF(IAT)=(AMASS(IAT)*ZREF(IAT)) &
                            *(SERVAL(ISERIE)/(DELTA*NSKIP))

                       !     n*MdX/dt - F
                       XREF(IAT)=XREF(IAT)-DX(IAT)
                       YREF(IAT)=YREF(IAT)-DY(IAT)
                       ZREF(IAT)=ZREF(IAT)-DZ(IAT)
                       !    S = S + (n*MdX/dt - F)^2
                       OMSCORE=OMSCORE +  &
                            (XREF(IAT)**2 + YREF(IAT)**2 + ZREF(IAT)**2) &
                            *((DELTA*NSKIP)/(2*SERVAL(ISERIE)))
                       !     Xi-1 = Xi
                       XREF(IAT)=X(IAT)
                       YREF(IAT)=Y(IAT)
                       ZREF(IAT)=Z(IAT)
                    ENDIF
                 ENDDO
                 IF(NTOT.EQ.1) THEN
                    TQ(1,ISERIE)=OMSCORE
                    WRITE(OUTU,*) 'OMSCORE> Scaling factor: ', TQ(1,ISERIE)
                 ENDIF
                 IF(PRNLEV.GE.3) WRITE(OUTU,*)'OMSCORE> ',OMSCORE &
                      ,TVAL
                 TVAL=TVAL + OMSCORE/TQ(1,ISERIE)
              ENDIF
              !
#endif /* (omscore)*/
              !.......................................................................
              ! PUCKER TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'PUC6') THEN
              ! Puckering 6-member ring
              ave = zero
              do knq=1,nq
                 ! Calculate puckering coordinates
                 call pucker6cp((/x(qat(is+knq-1)), x(qat(is+nq+knq-1)), x(qat(is+2*nq+knq-1)),&
                      x(qat(is+3*nq+knq-1)), x(qat(is+4*nq+knq-1)), x(qat(is+5*nq+knq-1)) /), &
                      (/ y(qat(is+knq-1)), y(qat(is+nq+knq-1)), y(qat(is+2*nq+knq-1)), &
                      y(qat(is+3*nq+knq-1)), y(qat(is+4*nq+knq-1)), y(qat(is+5*nq+knq-1)) /), &
                      (/ z(qat(is+knq-1)), z(qat(is+nq+knq-1)), z(qat(is+2*nq+knq-1)), &
                      z(qat(is+3*nq+knq-1)), z(qat(is+4*nq+knq-1)), z(qat(is+5*nq+knq-1)) /), &
                      Q_puc, theta_puc, phi_puc)
                 if (gecd == 1) then
                    ave = ave + Q_puc
                 elseif (gecd == 2) then
                    ave = ave + theta_puc
                 else
                    ave = ave + phi_puc
                 endif
              enddo
              tval = ave/nq
              !.......................................................................
              ! PUCKER TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'PUCK') THEN
              ! Puckering 5-member ring
              IF(SERVAL(ISERIE) .GT. 0.0) THEN
                 CALL PUCKA(INT(SERVAL(ISERIE)),PHASE,AMPL,'AS  ',X,Y,Z)
              ELSE
                 CALL PUCKA(-INT(SERVAL(ISERIE)),PHASE,AMPL,'CP  ',X,Y,Z)
              ENDIF
              IF(GECD.EQ.1) TVAL=PHASE
              IF(GECD.EQ.2) TVAL=AMPL
              !
              !.......................................................................
              ! NORMAL MODE PROJECTION TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'MODE') THEN
              VAL=ZERO
              DO IAT=1,NAT
                 IF(ISLCT(IAT).NE.0) THEN
                    IATB=((IAT-1)+(GECD-1)*NAT)*3
                    SQAMS=SQRT(AMASS(IAT))
                    VAL=VAL+(DDV(IATB+1)*(X(IAT)-XREF(IAT)) &
                         +DDV(IATB+2)*(Y(IAT)-YREF(IAT)) &
                         +DDV(IATB+3)*(Z(IAT)-ZREF(IAT)))*SQAMS
                 ENDIF
              ENDDO
              !C         IF(GECD.EQ.2) THEN
              !C            VAL=VAL*VAL*HALF
              !C            IF(VELCD.NE.1) THEN
              !CC POTENTIAL ENERGY ALONG THE MODE
              !CC
              !C               VAL=8.48388D-05*VAL*SERVAL(ISERIE)**2
              !C            ENDIF
              !C         ENDIF
              TVAL=VAL
              !
              !.......................................................................
              ! SURFACE AREA TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'SURF') THEN
#if KEY_NOMISC==0
              VECCD=VECCOD(ISERIE) 
              IF(VECCD.GE.1)THEN
                 NSLCT=NSELCT(NAT,ISLCT)

                 call chmalloc('anacor.src','ANACOR','matspace',4*NSLCT+2*nat,crl=matspace)
                 xmat => matspace(0*nslct+1 : 1*nslct)
                 ymat => matspace(1*nslct+1 : 2*nslct)
                 zmat => matspace(2*nslct+1 : 3*nslct)
                 scrr  => matspace(3*nslct+1 : 4*nslct)
                 mat  => matspace(4*nslct+1 : 4*nslct+nat)
                 wm   => matspace(4*nslct+nat+1 : 4*nslct+2*nat)
                 GECD=GECD-1
                 IF(VECCD.EQ.1)GECD=GECD+1   
                 LWEIG=(GECD.LT.0)
                 MODE=ABS(GECD)
                 ACCURACY=-1.0
                 IPT=IS
                 IRES0=GETRES(QAT(IPT),IBASE,NRES)
                 ! SURFAC destroys "WMAIN", so  we need to save it if weighting is used
                 IF(LWEIG) wm(1:nat) = WMAIN(1:nat)
                 OLDPRN=PRNLEV
                 PRNLEV=2
                 CALL SURFAC(NAT,X,Y,Z,WMAIN,LWEIG,ISLCT,NSLCT,XMAT, &
                      YMAT,ZMAT,MAT,SCRr, &
                      MODE,SERVAL(ISERIE),ACCURACY)
                 PRNLEV=OLDPRN
              ENDIF
              ! Add up ASA for requested atoms
              TVAL=0.0
              ! Keep track of residue borders if VECCD .NE. 1
              IAT=QAT(IPT)
305           TVAL=TVAL+ WMAIN(IAT)
              IPT=IPT+1
              ! Check that we don't exhaust the atom list
              IF(IPT.LE. (IS+NQ-1) )THEN
                 IAT=QAT(IPT)
                 ! Check that we are in same residue
                 IF(VECCD.NE.1) THEN
                    IRES1=GETRES(IAT,IBASE,NRES)
                    IF(IRES1.EQ.IRES0) GOTO 305
                    IRES0=IRES1
                 ELSE 
                    GOTO 305
                 ENDIF
              ELSE
                 IF(LWEIG) wmain(1:nat) = wm(1:nat)
                 call chmdealloc('anacor.src','ANACOR','matspace',4*NSLCT+2*nat,crl=matspace)
              ENDIF
#else /**/
              CALL WRNDIE(-1,'<ANACOR>','SURFAC code is not compiled')
#endif 
              !.......................................................................
              ! TEMPERATURE TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'TEMP') THEN
              TEMP=ZERO
              IPT=0
              DO IAT=1,NAT
                 IF(ISLCT(IAT).NE.0) THEN
                    AMS=AMASS(IAT)
                    TEMPI=X(IAT)*X(IAT)+Y(IAT)*Y(IAT)+Z(IAT)*Z(IAT)
                    TEMPI=TEMPI*AMS
                    TEMP=TEMP+TEMPI
                    IPT=IPT+3
                 ENDIF
              ENDDO
              TEMP=TEMP*HALF
              IF(SERVAL(ISERIE).GT.ZERO) IPT=SERVAL(ISERIE)
              IF(SERVAL(ISERIE).EQ.ZERO) IPT=NDEGF
              TEMP=TWO*TEMP/(KBOLTZ*IPT)
              TVAL=TEMP
              !
              !.......................................................................
              ! TIME TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'TIME') THEN
              TVAL=(NBEGN+NTOT)*DELTA*NSKIP*SERVAL(ISERIE)
              !
              !.......................................................................
              ! ZERO TIMESERIES
           ELSE IF(ICLASS(ISERIE).EQ.'ZERO') THEN
              TVAL=ZERO
              !
              !.......................................................................
              ! UNRECOGNIZED TIMESERIES
           ELSE
              IF(NTOT.EQ.0) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,922) ICLASS(ISERIE),SNAME(ISERIE)
              ENDIF
              TVAL=ZERO
           ENDIF
922        FORMAT(' Unrecognized CLASS "',A4,'" for time series ',A4,'.')
           TQ(NTOT+1,ISERIE)=TVAL
           IF(PRNLEV.GT.7) WRITE(OUTU,933) ISERIE,NTOT+1,TVAL
933        FORMAT(' ANACOR:  SERIES',I5,', INDEX',I8,', VALUE',F14.6)
           !
        ENDIF stot_eq_0
     ENDDO bigseriesloop

     !-----------------------------------------------------------------------
     NTOT=NTOT+1
     IF(NTOT.GE.MXNTOT .AND. ISTATS.GE.0) THEN
        CALL WRNDIE(-2,'<ANACOR>', &
             'TOO MANY TIME STEPS. REMAINDER IGNORED')
        ISTATS=-1
     ENDIF
  enddo loop200

  !=======================================================================
  !
  ! Time series   q(t) has been calculated for all series
  ! Write average and r.m.s. fluctuations
  !
  ! **** Modification 5/93, M.E. Karpen, to compute correct average & rms
  !      fluctuation of angle data - takes into account angle periodicity.
  !              Corrected - BRB 11/93
  !
  series: DO ISERIE=1,NSERIE
     IF(STOT(ISERIE).EQ.0) THEN
        !
        IF( (GECD .NE. 2) .AND. &
             (ICLASS(ISERIE) .EQ. 'DIHE' .OR. &
             ICLASS(ISERIE) .EQ. 'IMPR')) THEN
           !
           QAVEGX=ZERO
           QFLCTX=ZERO
           AVETEMP = ZERO
           DO K=1,NTOT
              DIFF = TQ(K,ISERIE) - AVETEMP
              DO WHILE (DIFF .GE. 180.)
                 DIFF = DIFF - 360.
              ENDDO
              DO WHILE (DIFF .LT. -180.)
                 DIFF = DIFF + 360.
              ENDDO
              AVETEMP = AVETEMP + DIFF
              QAVEGX=QAVEGX + AVETEMP
              QFLCTX=QFLCTX + AVETEMP**2
           ENDDO
           QAVEGX=QAVEGX/NTOT
           QFLCTX=QFLCTX/NTOT - QAVEGX**2
           SFLCT(ISERIE)=ZERO
           IF(QFLCTX.GT.0.0) SFLCT(ISERIE)=SQRT(QFLCTX)
           !
           DO WHILE (QAVEGX .GE. 180.)
              QAVEGX = QAVEGX - 360.
           ENDDO
           DO WHILE (QAVEGX .LT. -180.)
              QAVEGX = QAVEGX + 360.
           ENDDO
           SAVEG(ISERIE)=QAVEGX
        ELSE
           QAVEGX=ZERO
           DO K=1,NTOT
              QAVEGX=QAVEGX+TQ(K,ISERIE)
           ENDDO
           QAVEGX=QAVEGX/NTOT
           QFLCTX=ZERO
           DO K=1,NTOT
              QFLCTX=QFLCTX+(TQ(K,ISERIE)-QAVEGX)**2
           ENDDO
           SAVEG(ISERIE)=QAVEGX
           SFLCT(ISERIE)=ZERO
           IF(QFLCTX.GT.0.0) SFLCT(ISERIE)=SQRT(QFLCTX/NTOT)
        ENDIF

        IF(PRNLEV.GE.2) WRITE(OUTU,2306) ISERIE,SNAME(ISERIE), &
             SAVEG(ISERIE),SFLCT(ISERIE)
2306    FORMAT(I5,' Series "',A4,'"  Average =',F15.6, &
             '  rms Fluctuation =',F15.6)

     ENDIF
  ENDDO series
  !
  call chmdealloc('anacor.src','ANACOR','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('anacor.src','ANACOR','TEMPa',NATOM,cr4=TEMPa)
  if(allocated(stphi))then
     call deallocate_stphi_arrays
     call deallocate_stphi
  endif

  RETURN

contains
  subroutine allocate_stphi
    integer locierr
    allocate(stphi(nserie),stat=locierr)
    if(locierr /= 0) then
       print *,"nserie"
       call wrndie(-3,"anacor<anacor.src>","Cannot allocate stphi(nserie)")
    endif
    return
  end subroutine allocate_stphi

  subroutine deallocate_stphi
    integer locierr
    deallocate(stphi,stat=locierr)
    if(locierr /= 0) then
       call wrndie(-3,"anacor<anacor.src>","Cannot allocate stphi(nserie)")
    endif
    return
  end subroutine deallocate_stphi

  subroutine allocate_stphi_arrays
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%s(1)%a',NQ, crl=STPHI(iserie)%s(1)%a)
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%s(2)%a',NQ, crl=STPHI(iserie)%s(2)%a)
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%s(3)%a',NQ, crl=STPHI(iserie)%s(3)%a)
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%s(4)%a',NQ, crl=STPHI(iserie)%s(4)%a)
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%s(5)%a',NQ, crl=STPHI(iserie)%s(5)%a)
    call chmalloc('anacor.src','ANACOR','STPHI(iserie)%iarr',  NQ,intg=STPHI(iserie)%iarr)
    return
  end subroutine allocate_stphi_arrays

  subroutine deallocate_stphi_arrays
    integer i,nq
    do i=1,nserie
       if(.not. allocated(STPHI(i)%s(1)%a) )cycle
       if(size(STPHI(i)%s(1)%a) > 0 ) then
          nq=size(STPHI(i)%s(1)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%s(1)%a',NQ, crl=STPHI(i)%s(1)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%s(2)%a',NQ, crl=STPHI(i)%s(2)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%s(3)%a',NQ, crl=STPHI(i)%s(3)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%s(4)%a',NQ, crl=STPHI(i)%s(4)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%s(5)%a',NQ, crl=STPHI(i)%s(5)%a)
          call chmdealloc('anacor.src','ANACOR','STPHI(i)%iarr',  NQ,intg=STPHI(i)%iarr)
       endif
    enddo
    return
  end subroutine deallocate_stphi_arrays

END SUBROUTINE ANACOR

SUBROUTINE NEXTE(XT,YT,ZT,TVAL,NTOT)
  !-----------------------------------------------------------------------
  !    THIS ROUTINE CALLS ENERGY WITH THE STANDARD ARRAYS AND FILLS
  !    ARRAYS LIKE DX,DY,DZ AND ALL THE ENERGY TERMS.
  !    IT HAS A SHORT CALLING SEQUENCE FOR SIMPLICITY.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use coord
  use deriv
  use energym
  use stream
  use heurist,only:updeci
  !
  implicit none
  !
  real(chm_real) :: XT(:), YT(:), ZT(:)
  real(chm_real) TVAL
  INTEGER NTOT,ISTEP
  !
  ! Do list updates if appropriate
  ISTEP=NTOT+1
  CALL UPDECI(ISTEP,XT,YT,ZT,WMAIN, &
       0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
  !
  CALL ENERGY(XT,YT,ZT,DX,DY,DZ,BNBND,BIMAG,1)
  !C##IF PARALLEL
  !C Don't really need forces??
  !C      CALL VDGBR(DX,DY,DZ,1)
  !C##ENDIF
  TVAL=EPROP(EPOT)
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,44) ISTEP,TVAL
44 FORMAT(I10,F20.5)
  !
  RETURN
END SUBROUTINE NEXTE
!
!----------------------------------------------------------------------
!
SUBROUTINE VECMIN(IAT,JAT,XCOMP)
  !
  !     a routine that finds the minimum distance vector for atoms
  !     IAT and JAT
  !     it assumes that the image arrays are in use and up to date !
  !
  !     this is only a 'jump'-routine which calls the working routine
  !     VECMI2 with all needed data
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use coord
  use image
  use psf
  !
  implicit none
  INTEGER IAT,JAT
  real(chm_real) XCOMP
  !
  CALL VECMI2(IAT,JAT,XCOMP,X,Y,Z,NATOM,NATIM, &
       BIMAG%IMATTR)
  !
  RETURN
END SUBROUTINE VECMIN
!
!----------------------------------------------------------------------
!
SUBROUTINE VECMI2(IAT,JAT,XCOMP,X,Y,Z,NATOM,NATIM,IMATTR)
  !
  !     this is the work function that finds the minimum distance
  !     between IAT and JAT by evaluating all images
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use param_store, only: set_param
  !
  implicit none
  INTEGER IAT,JAT
  real(chm_real) XCOMP,X(*),Y(*),Z(*)
  INTEGER NATOM,NATIM,IMATTR(*)
  !
  INTEGER I,K,L
  real(chm_real) XMIN,YMIN,ZMIN,RMIN
  real(chm_real) XDD,YDD,ZDD,RDD
  !
  XMIN=(X(IAT)-X(JAT))
  YMIN=(Y(IAT)-Y(JAT))
  ZMIN=(Z(IAT)-Z(JAT))
  RMIN=(XMIN*XMIN)+(YMIN*YMIN)+(ZMIN*ZMIN)
  !
  !     and now check all images and see if we find one of the 2 atoms
  DO I=(NATOM+1),NATIM
     K=IMATTR(I)
     IF(K.EQ.IAT) THEN
        !           i.e. we found an image of IAT
        XDD=(X(I)-X(JAT))
        YDD=(Y(I)-Y(JAT))
        ZDD=(Z(I)-Z(JAT))
        RDD=(XDD*XDD)+(YDD*YDD)+(ZDD*ZDD)
        IF(RDD.LT.RMIN) THEN
           !              i.e. our new distance is smaller
           XMIN=XDD
           YMIN=YDD
           ZMIN=ZDD
           RMIN=RDD
        ENDIF
     ELSE IF(K.EQ.JAT) THEN
        !           i.e. we found an image of JAT
        XDD=(X(IAT)-X(I))
        YDD=(Y(IAT)-Y(I))
        ZDD=(Z(IAT)-Z(I))
        RDD=(XDD*XDD)+(YDD*YDD)+(ZDD*ZDD)
        IF(RDD.LT.RMIN) THEN
           !              i.e. our new distance is smaller
           XMIN=XDD
           YMIN=YDD
           ZMIN=ZDD
           RMIN=RDD
        ENDIF
     ENDIF
  ENDDO
  !
  RMIN=SQRT(RMIN)
  !     now put the 4 values int the MSR array
  call set_param('XVMI',XMIN)
  call set_param('YVMI',YMIN)
  call set_param('ZVMI',ZMIN)
  call set_param('RVMI',RMIN)            
  !
  !     and don't forget the return value
  XCOMP=XMIN
  !
  return
end SUBROUTINE VECMI2

end module anacor_m

