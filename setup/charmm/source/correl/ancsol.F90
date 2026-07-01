module ancsol_mod
  use chm_types
  use dimens_fcm
  implicit none

  type serdat_type
     integer row,col,spc
     real(chm_real), pointer,dimension(:,:) :: pt
     integer,        pointer,dimension(:,:) :: ipt
  end type serdat_type
  
contains


#if KEY_CORSOL==1 /*corsol*/
  SUBROUTINE ANCSOL(X,Y,Z,XREF,YREF,ZREF,TQ,MXNTOT,NTOT, &
       NUNIT,NFU,NBEGN,NSTOP,NSKIP, &
       NUNIT2,NFU2,NBEGN2,NSTOP2,NSKIP2, &
       DELTA,VELCD, &
       NSERIE,SNAME,STOT,ICLASS,GECOD,SERVAL, &
       LFFT,SAVEG,SFLCT,SERPT,SERNQ, &
       IUNMAX,INOMAX, &
#if KEY_SHELL==1
       NSHL,NASHL,NSHBLK,SHLLST,SHBLKL,QSHLUP,QSHIMG,QSHELL, &    
#endif
       QAT,ISLCT,ATOMIN,WMAIN,QORIENT,QMASS,JSLCT, &
       MAXSER,NSRST,SERSET,SERDAT,SERCOR,SERSEL,SERNCR, &
       SERACT,DOIMAGES,QBOTH,QAFFT)
    !
    !-----------------------------------------------------------------------
    !     THIS ROUTINE READS 'NUNIT' FILES BEGINNING 'NFU'
    !     CONTAINING DYNAMICS COORDINATES OR VELOCITIES AND
    !     AND CALCULATES THE QUANTITIES TO BE CORRELATED
    !     (IN LARGE PARTS IT REUSES ANACOR.SRC)
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
    !     SERVAL(NSERIE) - number of primary atoms for this serie
    !     LFFT(NSERIE) - Logical flag for fft or windowing correlation
    !     SAVEG(NSERIE) - Average value for time series
    !     SFLCT(NSERIE) - Fluctuation value for each series
    !     SERPT(NSERIE) - Pointer to start of QAT atom descriptor
    !     SERNQ(NSERIE) - Number of atoms in QAT
    !     IUNMAX(NSERIE) - unit no for output of INOMAX max contributions
    !     INOMAX(NSERIE) - nr of maximum contributions to write out
    !     QAT(*) - Atom descriptors for some series
    !     ISLCT - First atom selection (used by the time series)
    !     ATOMIN - Scratch array used by READCV and ORINTC
    !     WMAIN  - The standard weighting array
    !     QORIENT - Frames will be best fit to the reference
    !     QMASS  - A mass weighting will be used in the best fit
    !     JSLCT  - The second atom selection, used for best fit atoms
    !     SERA1/2 - values special to each series
    !     QAFFT   -  is at least one FFT series produced (check space availability)
    !
    !
    !     Time series codes(ICLASS),  secondary code(GECOD)
    !
    !        DDIP                    1 - precise sum of dip-dip interactions
    !                                2 - sum of radial parts
    !                                3 - sum of angular parts
    !
    !        SATM                    1 - sum of <N_i(0) * N_i(t)> for all i
    !
    !      Tibor Rudas 13. 02. 2003 - June 2006
    !
#if KEY_CHEQ==1
  use cheq,only:QCG                 
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor            
#endif
  use dimens_fcm
  use exfunc
  use number
  use bases_fcm
  use ctitla
  use consta
  use hbondm
  use image
  use image_routines_module, only: transo
  use imgup, only: upimag0
  use cvio, only: readcv
  use param
  use psf
  use code
  use fourdm
  use deriv
  use stream
    ! needed for TRANSO
#if KEY_FLUCQ==1
  use flucq      
#endif
  use memory
  use usermod
  use corsubs,only:orintc
#if KEY_SHELL==1
  use shell,only:shlupd     
#endif
    !
    integer,allocatable,dimension(:) :: FREEAT,jtemp
    real(chm_real4),allocatable,dimension(:) :: itemp
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*),XREF(*),YREF(*),ZREF(*)
    INTEGER MXNTOT,NTOT
    real(chm_real) TQ(MXNTOT,*)
    INTEGER NUNIT,NFU,NBEGN,NSTOP,NSKIP
    INTEGER NUNIT2,NFU2,NBEGN2,NSTOP2,NSKIP2
    real(chm_real) DELTA
    INTEGER VELCD,NPRM
    INTEGER NSERIE,STOT(*)
    character(len=*) ICLASS(*)
    character(len=*) SNAME(*)
    INTEGER GECOD(*)
    LOGICAL LFFT(*)
    real(chm_real) SERVAL(*),SAVEG(*),SFLCT(*)
    INTEGER SERPT(*),SERNQ(*)
    INTEGER IUNMAX(*),INOMAX(*)
    INTEGER QAT(*),ISLCT(*),ATOMIN(2,*)
    LOGICAL QORIENT,QMASS
    INTEGER JSLCT(*)
    INTEGER MAXSER,NSRST,SERSET(*)
    type(serdat_type),dimension(maxser) :: SERDAT,sercor,sersel
    integer,allocatable,dimension(:) :: STIDX,statm
    real(chm_real),allocatable,dimension(:) :: stval,sttmp
    INTEGER SERNCR(*),SERACT(*)
    LOGICAL DOIMAGES,QBOTH,QAFFT
    !
    !
    ! would belong above this INCLUDE block but I need NATOM
#if KEY_SHELL==1 /*shell*/
    INTEGER NSHL,NASHL(*),NSHBLK,SHLLST(NATOM,NSHL),SHBLKL(*)
    LOGICAL QSHLUP,QSHIMG,QSHELL
#endif /* (shell)*/
    INTEGER NAT,ISTATS
    !     i don't know if we need to double all these (maybe it is ok
    !     for some of them to be just overwritten by the second readcv
    !     or they are the same in a coor/vel pair - check later...)
    INTEGER IUNIT,IUNIT2,NFILE2,ISTEP2,ISTATS2,NDEGF2,NSAVV2
    INTEGER NFREAT,NFILE,ISTEP,NDEGF,NSAVV
    INTEGER ISERIE,NQ,IS,KNQ,I,IAT,IPT,JAT,IQ,J
    INTEGER IATB,K,GECD,NDIG
    INTEGER NINIT,II,NDAT
    INTEGER OLDPRN
    INTEGER IROWS,ICOLS
    real(chm_real), pointer,dimension(:,:) :: IDATA
    INTEGER DIMENS,DIMENC
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
    real(chm_real) U(3,3),SCR(21),AMOM(6),EV(3),THETA,THETA2
    real(chm_real) P(3), S(3), T(3)
    !
    LOGICAL QFFT,QPRINT
    character(len=4) HDRC,HDRD,HDRV,HDR1,HDR2,HDR3,HDR4
    DATA HDRC,HDRD,HDRV/'COOR','CORD','VELD'/

    call chmalloc('ancsol.src','ANCSOL','FREEAT',NATOM,intg=FREEAT)
    call chmalloc('ancsol.src','ANCSOL','ITEMP',NATOM,cr4=ITEMP)
    call chmalloc('ancsol.src','ANCSOL','JTEMP',NATOM,intg=JTEMP)
    HDR1=HDRC
    HDR2=HDRD
    IF(VELCD == 1) THEN
       HDR1=HDRV
       HDR2=HDRV
    ENDIF
    HDR3=HDRC
    HDR4=HDRD
    IF(QBOTH) THEN
       HDR3=HDRV
       HDR4=HDRV
    ENDIF
    NAT=NATOM
    QPRINT=(PRNLEV > 5)
    !
    !     some informational print out.
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,33)
33     FORMAT(' The following time series will be filled:')
       DO I=1,NSRST
          ISERIE=SERSET(I)
          IF(STOT(ISERIE) == 0) THEN
             IF(LFFT(ISERIE)) THEN
                WRITE(OUTU,34) SNAME(ISERIE), ' using FFT.'
             ELSE
                WRITE(OUTU,34) SNAME(ISERIE), ' using WINdowing.'
             ENDIF
          ENDIF
34        FORMAT(20X,A4,A)
       ENDDO
       WRITE(OUTU,35)
35     FORMAT(/)
    ENDIF
    !
    !=======================================================================
    !     INITIALIZE DATA ARRAYS
    DO I=1,NSRST
       SERACT(I)=0
       ISERIE=SERSET(I)
       IF((ICLASS(ISERIE) == 'VACF').OR. &
            (ICLASS(ISERIE) == 'WREO')) THEN
          !           pass: 3 dimensions, 1 correl dimension
          DIMENS=3
          DIMENC=1
          CALL INTCRS(I,MXNTOT, &
               SERDAT(I)%pt,SERDAT(I)%col,SERDAT(I)%row,SERDAT(I)%spc, &
               sercor(i)%pt,sercor(i)%col,sercor(i)%row,sercor(i)%spc, &
               SERSEL(I)%ipt,SERSEL(I)%col,SERSEL(I)%row,SERSEL(I)%spc, &
               SERNCR(I),LFFT(ISERIE),SERNQ(ISERIE), &
               QAT(SERPT(ISERIE)),DIMENS,DIMENC)
#if KEY_SHELL==1 /*shell*/
       ELSE IF(ICLASS(ISERIE) == 'SATM') THEN
          CALL INTSTM(I,MXNTOT, &
               SERDAT(I)%pt,SERDAT(I)%col,SERDAT(I)%row,SERDAT(I)%spc, &
               sercor(i)%pt,sercor(i)%col,sercor(i)%row,sercor(i)%spc, &
               SERSEL(I)%iPT,SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%SPC, &
               SERNCR(I),LFFT(ISERIE),SERNQ(ISERIE), &
               QAT(SERPT(ISERIE)))
       ELSE IF(ICLASS(ISERIE) == 'SATX') THEN
          CALL INTSTX(I,MXNTOT, &
               SERDAT(I)%pt,SERDAT(I)%col,SERDAT(I)%row,SERDAT(I)%spc, &
               sercor(i)%pt,sercor(i)%col,sercor(i)%row,sercor(i)%spc, &
               SERSEL(I)%iPT,SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%SPC, &
               SERNCR(I),LFFT(ISERIE),SERNQ(ISERIE), &
               QAT(SERPT(ISERIE)))
       ELSE IF(ICLASS(ISERIE) == 'MRTI') THEN
          CALL INTMRT(I,MXNTOT, &
               SERDAT(I)%ipt,SERDAT(I)%col,SERDAT(I)%row,SERDAT(I)%spc, &
               sercor(i)%ipt,sercor(i)%col,sercor(i)%row,sercor(i)%spc, &
               SERSEL(I)%iPT,SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%SPC, &
               SERNCR(I),LFFT(ISERIE),SERNQ(ISERIE), &
               QAT(SERPT(ISERIE)))
#endif /* (shell)*/
       ELSE IF((ICLASS(ISERIE) == 'DDIP').OR. &
            (ICLASS(ISERIE) == 'SDDP'))THEN
          NPRM=SERVAL(ISERIE)
          CALL INTDDP(I,MXNTOT, &
               SERDAT(I)%pt,SERDAT(I)%col,SERDAT(I)%row,SERDAT(I)%spc, &
               sercor(i)%pt,sercor(i)%col,sercor(i)%row,sercor(i)%spc, &
               SERSEL(I)%iPT,SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%SPC, &
               SERNCR(I),LFFT(ISERIE),SERNQ(ISERIE), &
               NPRM,QAT(SERPT(ISERIE)))
       ELSE
          CALL WRNDIE(-5,'<CORSOL>', &
               'Unknown series code.')
       ENDIF
    ENDDO
    !
    !=======================================================================
    !     BEGIN READING FILES
    !
    NTOT=0
    IUNIT=NFU
    IUNIT2=NFU2
    ISTATS=1
    ISTATS2=1
200 CONTINUE
    if(.not.allocated(fdim)) call allocate_fourd_ltm
    CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
         CG,QCG,                        & 
#endif
         ITEMP,NAT,FREEAT,NFREAT, &
         NFU,NUNIT,IUNIT,NFILE, &
         ISTEP,ISTATS,NDEGF,DELTA2, &
         NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
#if KEY_FOURD==1
         TITLEA,NTITLA,.TRUE.,FDIM,.true. & 
#endif
#if KEY_FOURD==0
         TITLEA,NTITLA,.FALSE.,0,.true. &   
#endif
         )
    IF(NTOT == 0) THEN
       DELTA=DELTA2*TIMFAC
       ! Check to see if DELTA should be rounded to nearest integer femtosecond.
       IF(ABS(DELTA-INT(THOSND*DELTA+HALF)/THOSND) < DELTA/FTHSND) &
            DELTA=INT(THOSND*DELTA+HALF)/THOSND
    ENDIF
    ! Orient this frame if requested.
    IF(QORIENT) THEN
       CALL ORINTC(NATOM,X,Y,Z,XREF,YREF,ZREF,AMASS,QMASS, &
            .TRUE.,ATOMIN,JSLCT,.FALSE.,WMAIN, &
            .FALSE.,QPRINT)
    ENDIF
    !     UPDATE IMAGES... (quietly...)
    IF(DOIMAGES) THEN
       !        Construct the coordinates.
       CALL TRANSO(X,Y,Z, &
            [zero], [zero], [zero], &
            .FALSE.,.FALSE., [zero], NATOM,NTRANS, &
            IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
            NOROT,NATIM &
#if KEY_FLUCQ==1
            ,QFLUC,CG,FQCFOR    & 
#endif
            )
       !        we don't want the output from UPIMAG -> reset PRNLEV
       OLDPRN=PRNLEV
       PRNLEV=1
       CALL UPIMAG0(X,Y,Z,WMAIN,0)
       !        reset printlevel
       PRNLEV=OLDPRN
       !
    ENDIF
#if KEY_SHELL==1 /*shell*/
    ! update shell if needed
    IF(QSHELL) THEN
       !        update shell
       QSHLUP=.FALSE.
       CALL SHLUPD
       IF(.NOT.QSHLUP) THEN
          CALL WRNDIE(-3,'<CORSOL>', &
               'SHELL update failed.')
       ENDIF
    ENDIF
#endif /* (shell)*/
    !-----------------------------------------------------------------------
    DO I=1,NSRST
       ISERIE=SERSET(I)
       IF(STOT(ISERIE) == 0) THEN
          !
          NQ=SERNQ(ISERIE)
          IS=SERPT(ISERIE)
          QFFT=LFFT(ISERIE)
          GECD=GECOD(ISERIE)
          !          /
          !         /
          !        /
          !       /
          !      /
          !.......................................................................
          ! USER SUPPLIED FUNCTION
          IF(ICLASS(ISERIE) == 'USER') THEN
             CALL USRTIM(SERVAL(ISERIE),QAT(IS),NQ,NTOT, &
                  NATOM,X,Y,Z,XREF,YREF,ZREF,NSKIP,DELTA,TVAL,ISLCT)
             !.......................................................................
#if KEY_SHELL==1 /*shell*/
          ELSE IF(ICLASS(ISERIE) == 'MRTI') THEN
             !        mean residence time
             CALL FILMRT(NATOM,QAT(IS),NQ,GECD, &
                  serdat(i)%spc,serdat(i)%ipt, &
                  sercor(i)%spc,sercor(i)%ipt, &
                  SERSEL(I)%SPC,SERSEL(I)%iPT, &
                  NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
             !
             !.......................................................................
          ELSE IF(ICLASS(ISERIE) == 'SATM') THEN
             !        atom in shell GECD?
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (SATM)')
             ENDIF
             !
             !        and now fill in position SERACT(I) in the series
             CALL FILSAT(NATOM,SERACT(I),serdat(i)%col,NQ,serdat(i)%pt, &
                  GECD,SERSEL(I)%iPT, &
                  NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL SATWIN(SERACT(I),NDAT,serdat(i)%col,NQ, &
                     serdat(i)%pt,sercor(i)%col,sercor(i)%pt)
             ENDIF
             !
             !.......................................................................
          ELSE IF(ICLASS(ISERIE) == 'SATX') THEN
             !        atom in shell GECD cross to shell SERVL?
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (SATX)')
             ENDIF
             !
             !        and now fill in position SERACT(I) in the series
             CALL FILSAX(NATOM,SERACT(I), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  GECD,INT(SERVAL(ISERIE)),SERSEL(I)%iPT, &
                  NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL SAXWIN(NATOM,SERACT(I),NDAT,NQ,QAT(SERPT(ISERIE)), &
                     serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                     sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                     SERSEL(I)%iPT)
             ENDIF
             !
             !.......................................................................
          ELSE IF(ICLASS(ISERIE) == 'SDDP') THEN
             !        dipole-dipole coupling for all atoms in shell GECOD of
             !        ISERIE
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (SDDP)')
             ENDIF
             !
             !        and now fill in position SERACT(I) in the series
             NPRM=SERVAL(ISERIE)
             CALL ZERSDD(SERACT(I), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%iPT)
             CALL FILSDD(SERACT(I),NATOM,NATIM,BIMAG%IMATTR, &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%iPT, &
                  GECD,NSHL,NASHL,NSHBLK,SHLLST,SHBLKL, &
                  JTEMP,DOIMAGES)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL DDPWIN(SERACT(I),NDAT, &
                     NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                     serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                     sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                     SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%iPT)
             ENDIF
             !
             !.......................................................................
#endif /* (shell)*/
          ELSE IF(ICLASS(ISERIE) == 'DDIP') THEN
             !        dipole-dipole coupling
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (DDIP)')
             ENDIF
             !
             !        and now fill in position SERACT(I) in the series
             NPRM=SERVAL(ISERIE)
             CALL FILDDP(SERACT(I),NATOM,NATIM,BIMAG%IMATTR, &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT, &
                  DOIMAGES)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL DDPWIN(SERACT(I),NDAT, &
                     NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                     serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                     sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                     SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT)
             ENDIF
             !
             !.......................................................................
          ELSE IF(ICLASS(ISERIE) == 'VACF') THEN
             !        velocity auto correlation function
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (VACF)')
             ENDIF
             !
             !        and now fill in position SERACT(I) in the series
             CALL FILVAC(SERACT(I),NATOM,SERVAL(ISERIE), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NQ,QAT(IS), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL WIN3D1(SERACT(I),NDAT,NQ,QAT(IS),zero, &
                     serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                     sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                     SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT)
             ENDIF
             !
             !.......................................................................
          ELSE IF(ICLASS(ISERIE) == 'WREO') THEN
             !        water auto orientation correlation function
             SERACT(I)=SERACT(I)+1
             !        correct if we are windowing
             IF((.NOT.QFFT).AND.(SERACT(I) > SERNCR(I))) THEN
                SERACT(I)=SERACT(I)-SERNCR(I)
             ELSE IF(SERACT(I) > MXNTOT) THEN
                CALL WRNDIE(-5,'<ANCSOL>', &
                     'Data overflow! (WREO)')
             ENDIF
             !
             CALL FILWRO(SERACT(I),NATOM,SERVAL(ISERIE), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NQ,QAT(IS), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT)
             !        finally: do window correlation if wanted
             IF(.NOT.QFFT) THEN
                NDAT=NTOT+1
                IF(NDAT > serdat(i)%col) NDAT=serdat(i)%col
                CALL WIN3D1(SERACT(I),NDAT,NQ,QAT(IS),SERVAL(ISERIE), &
                     serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                     sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                     SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT)
             ENDIF
             !
             !.......................................................................
             ! ZERO TIMESERIES
          ELSE IF(ICLASS(ISERIE) == 'ZERO') THEN
             !
             !.......................................................................
             ! UNRECOGNIZED TIMESERIES
          ELSE
             IF(NTOT == 0) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,922) ICLASS(ISERIE),SNAME(ISERIE)
             ENDIF
          ENDIF
922       FORMAT(' Unrecognized CLASS "',A4,'" for time series ',A4,'.')
          !      \
          !       \
          !        \
          !         \
          !          \
          !
       ENDIF
       !!       IF(STOT(ISERIE) == 0)
    ENDDO
    !!    DO ISERIE = ...
    !-----------------------------------------------------------------------
    NTOT=NTOT+1
    IF(QAFFT.AND.(NTOT >= MXNTOT).AND.(ISTATS.GE.0)) THEN
       CALL WRNDIE(-2,'<ANCSOL>', &
            'TOO MANY TIME STEPS. REMAINDER IGNORED')
       ISTATS=-1
    ENDIF
    !TR   istats2 is never touched from 1 if not reading two trajs
    IF((ISTATS >= 0).AND.(ISTATS2.GE.0)) GOTO 200
    !
    !=======================================================================
    !     NOW POST PROCESS ALL SERIES
    !
    DO I=1,NSRST
       ISERIE=SERSET(I)
       NPRM=SERVAL(ISERIE)
       NQ=SERNQ(ISERIE)
       IS=SERPT(ISERIE)
       !        pointers for the common functions (MAX/SUM fncs)
       !        (default = windowning -> corr structure)
       ICOLS=sercor(i)%col
       IROWS=sercor(i)%row
       IDATA => sercor(i)%pt
       !
       IF((ICLASS(ISERIE) == 'DDIP').OR. &
            (ICLASS(ISERIE) == 'SDDP'))THEN
          !
          IF(LFFT(ISERIE)) THEN
             !              doing in-place fft of all datasets
             CALL DDPFFT(NTOT, &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT, &
                  SERNCR(I))
             ICOLS=serdat(i)%col
             IROWS=serdat(i)%row
             IDATA => SERDAT(I)%pt
          ELSE
             !              renormalize the data acuumulated via windowing
             CALL NRMWIN(NTOT, &
                  sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                  SERNCR(I))
          ENDIF
          !           output of the maximum contributions
          IF(IUNMAX(ISERIE) > ZERO) THEN
             call chmalloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmalloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmalloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmalloc('ancsol.src','ANCSOL','sttmp',inomax(iserie)*3,crl=sttmp)
             CALL MAXDDP(NTOT,IUNMAX(ISERIE),INOMAX(ISERIE), &
                  STIDX,STVAL,STATM,STTMP, &
                  ICOLS,IROWS,IDATA, &
                  NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
                  SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT, &
                  SERNCR(I),DELTA*NSKIP)
             call chmdealloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmdealloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmdealloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmdealloc('ancsol.src','ANCSOL','sttmp',inomax(iserie)*3,crl=sttmp)
          ENDIF
          !           and now sum up the results
          CALL SUMDDP(MXNTOT,TQ(1,ISERIE), &
               ICOLS,IROWS,IDATA, &
               NPRM,QAT(IS),(NQ-NPRM),QAT(IS+NPRM), &
               SERSEL(I)%COL,SERSEL(I)%ROW,SERSEL(I)%IPT, &
               SERNCR(I))
#if KEY_SHELL==1 /*shell*/
       ELSE IF(ICLASS(ISERIE) == 'SATM') THEN
          IF(LFFT(ISERIE)) THEN
             !              doing in-place fft of all datasets
             CALL SATFFT(NTOT, &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  SERNCR(I))
             ICOLS=serdat(i)%col
             IROWS=serdat(i)%row
             IDATA => SERDAT(I)%pt
          ELSE
             !              renormalize the data acuumulated via windowing
             CALL NRMWIN(NTOT, &
                  sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                  SERNCR(I))
          ENDIF
          !           output of the maximum contributions
          IF(IUNMAX(ISERIE) > ZERO) THEN
             call chmalloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmalloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmalloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmalloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
             CALL MAXSAT(IUNMAX(ISERIE),INOMAX(ISERIE), &
                  STIDX,STVAL, &
                  STATM,STTMP, &
                  ICOLS,IROWS,IDATA, &
                  NQ,QAT(IS),SERSEL(I)%IPT, &
                  SERNCR(I),DELTA*NSKIP)
             call chmdealloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmdealloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmdealloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmdealloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
          ENDIF
          !           and sum up the results
          CALL SUMSAT(MXNTOT,TQ(1,ISERIE), &
               ICOLS,IROWS,IDATA, &
               SERNCR(I))
       ELSE IF(ICLASS(ISERIE) == 'SATX') THEN
          IF(LFFT(ISERIE)) THEN
             !              doing in-place fft of all datasets
             CALL SAXFFT(NATOM,NTOT,NQ,QAT(SERPT(ISERIE)), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  SERNCR(I),SERSEL(I)%IPT)
             ICOLS=serdat(i)%col
             IROWS=serdat(i)%row
             IDATA => SERDAT(I)%pt
          ELSE
             !              renormalize the data acuumulated via windowing
             CALL NRMWIN(NTOT, &
                  sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                  SERNCR(I))
          ENDIF
          !           output of the maximum contributions
          IF(IUNMAX(ISERIE) > ZERO) THEN
             call chmalloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmalloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmalloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmalloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
             CALL MAXSAX(IUNMAX(ISERIE),INOMAX(ISERIE), &
                  STIDX,STVAL, &
                  STATM,STTMP, &
                  ICOLS,IROWS,IDATA, &
                  NQ,QAT(IS),SERSEL(I)%IPT, &
                  SERNCR(I),DELTA*NSKIP,NATOM)
             call chmdealloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmdealloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmdealloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmdealloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
          ENDIF
          !           and sum up the results
          CALL SUMSAX(MXNTOT,TQ(1,ISERIE), &
               ICOLS,IROWS,IDATA, &
               SERNCR(I),NQ,1,1,one,.FALSE.,zero)
       ELSE IF(ICLASS(ISERIE) == 'MRTI') THEN
          !              for this series there are no individual components
          !              -> no max contributions
          !              so just sum up the results
          CALL SUMMRT(MXNTOT,TQ(1,ISERIE), &
               serdat(i)%ipt,SERNCR(I))
#endif /* (shell)*/
       ELSE IF(ICLASS(ISERIE) == 'VACF') THEN
          IF(LFFT(ISERIE)) THEN
             !              doing in-place fft of all datasets
             CALL FFT3D1(NATOM,NTOT,NQ,QAT(SERPT(ISERIE)), &
                  SERVAL(ISERIE), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  SERNCR(I),SERSEL(I)%IPT)
             ICOLS=serdat(i)%col
             IROWS=serdat(i)%row
             IDATA => SERDAT(I)%pt
          ELSE
             !              renormalize the data acuumulated via windowing
             CALL NRMWIN(NTOT, &
                  sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                  SERNCR(I))
          ENDIF
          !           output of the maximum contributions
          IF(IUNMAX(ISERIE) > ZERO) THEN
             call chmalloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmalloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmalloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmalloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
             CALL MAXSAX(IUNMAX(ISERIE),INOMAX(ISERIE), &
                  STIDX,STVAL, &
                  STATM,STTMP, &
                  ICOLS,IROWS,IDATA, &
                  NQ,QAT(IS),SERSEL(I)%IPT, &
                  SERNCR(I),DELTA*NSKIP,NATOM)
             call chmdealloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmdealloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmdealloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmdealloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
          ENDIF
          !           and sum up the results
          CALL SUMC1D(MXNTOT,TQ(1,ISERIE),zero, &
               ICOLS,IROWS,IDATA, &
               SERNCR(I),NQ,1,1,1.0/(NQ*TIMFAC*TIMFAC), &
               .TRUE.,(DELTA*NSKIP))
          !tr to get non-akma:
          !tr     $           SERNCR(I),NQ,1,1,(1.0D0/NQ),
       ELSE IF(ICLASS(ISERIE) == 'WREO') THEN
          IF(LFFT(ISERIE)) THEN
             !              doing in-place fft of all datasets
             CALL FFT3D1(NATOM,NTOT,NQ,QAT(SERPT(ISERIE)), &
                  SERVAL(ISERIE), &
                  serdat(i)%col,serdat(i)%row,serdat(i)%pt, &
                  SERNCR(I),SERSEL(I)%IPT)
             ICOLS=serdat(i)%col
             IROWS=serdat(i)%row
             IDATA => SERDAT(I)%pt
          ELSE
             !              renormalize the data acuumulated via windowing
             CALL NRMWIN(NTOT, &
                  sercor(i)%col,sercor(i)%row,sercor(i)%pt, &
                  SERNCR(I))
          ENDIF
          !           output of the maximum contributions
          IF(IUNMAX(ISERIE) > ZERO) THEN
             call chmalloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmalloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmalloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmalloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
             CALL MAXSAX(IUNMAX(ISERIE),INOMAX(ISERIE), &
                  STIDX,STVAL, &
                  STATM,STTMP, &
                  ICOLS,IROWS,IDATA, &
                  NQ,QAT(IS),SERSEL(I)%IPT, &
                  SERNCR(I),DELTA*NSKIP,NATOM)
             call chmdealloc('ancsol.src','ANCSOL','stidx',NQ-NPRM,intg=stidx)
             call chmdealloc('ancsol.src','ANCSOL','statm',NQ-NPRM,intg=statm)
             call chmdealloc('ancsol.src','ANCSOL','stval',NQ-NPRM,crl=stval)
             call chmdealloc('ancsol.src','ANCSOL','sttmp',inomax(iserie),crl=sttmp)
          ENDIF
          !           and sum up the results
          CALL SUMC1D(MXNTOT,TQ(1,ISERIE),SERVAL(ISERIE), &
               ICOLS,IROWS,IDATA, &
               SERNCR(I),NQ,1,1,(1.0D0/NQ), &
               .FALSE.,zero)
       ENDIF
       !
       !        FREE DATA ARRAYS
       !
       ! ALWAYS check for association since we don't want to do all the ifs
       !        for the series class

       ! first the pt-arrays: should be real or real8
       IF(associated(serdat(i)%pt)) CALL chmdealloc('ancsol.src','ANCSOL', &
            'serdat(i)%pt',serdat(i)%row,serdat(i)%col,crlp=serdat(i)%pt)
       IF(associated(sercor(i)%pt)) CALL chmdealloc('ancsol.src','ANCSOL', &
            'serdat(i)%pt',sercor(i)%row,sercor(i)%col,crlp=sercor(i)%pt)

       ! serdat%ipt is only used by some series (MRTI)
       IF(associated(SERdat(I)%ipt)) CALL chmdealloc('ancsol.src','ANCSOL', &
            'serdat(i)%ipt',serdat(i)%row,serdat(i)%col,intgp=serdat(i)%ipt)

       ! some series use sercor%ipt (offset pointers and the like)
       IF(associated(SERcor(I)%ipt)) CALL chmdealloc('ancsol.src','ANCSOL', &
            'sercor(i)%ipt',sercor(i)%row,sercor(i)%col,intgp=sercor(i)%ipt)

       ! just deallocate sersel%ipt, sersel%pt should never be used
       IF(associated(sersel(i)%ipt)) CALL chmdealloc('ancsol.src','ANCSOL', &
            'sersel(i)%ipt',sersel(i)%row,sersel(i)%col,intgp=sersel(i)%ipt)

    ENDDO
    !
    !=======================================================================
    !
    call chmdealloc('ancsol.src','ANCSOL','FREEAT',NATOM,intg=FREEAT)
    call chmdealloc('ancsol.src','ANCSOL','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('ancsol.src','ANCSOL','JTEMP',NATOM,intg=JTEMP)

    RETURN
  END SUBROUTINE ANCSOL
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTCRS(SERN,MXNTOT, &
       DATPT,DATCOL,DATROW,DATSPC, &
       CORPT,CORCOL,CORROW,CORSPC, &
       SELPT,SELCOL,SELROW,SELSPC, &
       NCORR,LFFT,NRAT,ATLST,DIMENS,DIMENC)
    !
    !     Initialize data, correl and selection arrays for one
    !     corsol series that is using only one atom selection (i.e. only NRAT
    !     primary atoms). DIMENS gives the dimensionality of the series, DIMENC
    !     that of the correl array.
    !     This routine allocates the space and fills the selection pointer
    !     array.
    !
    !     Tibor Rudas 15. 09. 2004 - June 2006
    !
  use psf
  use number,only:zero
  use memory
    !
    INTEGER SERN,MXNTOT
    real(chm_real), pointer,dimension(:,:) :: DATpt
    real(chm_real), pointer,dimension(:,:) :: corpt
    integer,pointer,dimension(:,:) :: selpt
    INTEGER DATCOL,DATROW,datspc
    INTEGER CORCOL,CORROW,CORSPC
    INTEGER SELCOL,SELROW,SELSPC
    INTEGER NCORR,NRAT,ATLST(*)
    LOGICAL LFFT
    INTEGER DIMENS,DIMENC
    !
    !
    INTEGER LEN
    !
    !     the DATA array which will hold our data
    !     we need DIMENS series for each primary atom
    DATCOL=NCORR
    IF(LFFT) DATCOL=MXNTOT
    DATROW=DIMENS*NRAT
    LEN=DATCOL*DATROW
    DATSPC=LEN
    call chmalloc('ancsol.src','INTCRS','DATPT',DATrow,datcol,crlp=DATPT)

    !     the CORREL array which holds the growing correlation function
    !     (only used when using windowing and not fft)
    IF(.NOT.LFFT) THEN
       !        this should not be necessary but let's be paranoid :)
       CORCOL=NCORR
       CORROW=NRAT*DIMENC
       LEN=CORCOL*CORROW
       CORSPC=LEN
       call chmalloc('ancsol.src','INTCRS','CORPT',CORrow,corcol,crlp=CORPT)
       corpt(1:corrow,1:corcol)=zero
    ELSE
       CORCOL=0
       CORROW=0
       CORSPC=0
    ENDIF
    !
    !     and finally the SELECTION array which is NATOM long and holds
    !     the number representing the data/correlation for a selected atom
    !     so: SEL(ATOM,1) holds entry for DATA and SEL(ATOM,2) for CORREL
    SELCOL=2
    SELROW=NATOM
    LEN=SELCOL*SELROW
    SELSPC=LEN
    call chmalloc('ancsol.src','INTCRS','SELPT',SELrow,selcol,intgp=SELPT)
    CALL INTSL3(SELPT,SELROW,NRAT,DIMENS,DIMENC,ATLST)
    !
    RETURN
  END SUBROUTINE INTCRS
  !
#if KEY_SHELL==1 /*shell*/
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTSTM(SERN,MXNTOT, &
       DATPT,DATCOL,DATROW,DATSPC, &
       CORPT,CORCOL,CORROW,CORSPC, &
       SELPT,SELCOL,SELROW,SELSPC, &
       NCORR,LFFT,NSAT,ATLST)
    !
    !     Initialize data, correl and selection arrays for one
    !     SATM series. This routine allocates the space and fills
    !     the selection pointer array.
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use memory
  use psf
    !
    INTEGER SERN,MXNTOT

    real(chm_real), pointer,dimension(:,:) :: DATpt
    real(chm_real), pointer,dimension(:,:) :: corpt
    integer,pointer,dimension(:,:) :: selpt
    INTEGER DATCOL,DATROW,datspc
    INTEGER CORCOL,CORROW,CORSPC
    INTEGER SELCOL,SELROW,SELSPC
    INTEGER NCORR,NSAT,ATLST(*)
    LOGICAL LFFT
    !
    !
    INTEGER LEN
    !
    !     the DATA array which will hold our data
    DATCOL=NCORR
    IF(LFFT) DATCOL=MXNTOT
    DATROW=NSAT
    LEN=DATCOL*DATROW
    datspc=len
    call chmalloc('ancsol.src','INTSTM','DATPT',DATrow,datcol,crlp=DATPT)

    !
    !     the CORREL array which holds the growing correlation function
    !     (only used when using windowing and not fft)
    IF(.NOT.LFFT) THEN
       !        this should not be necessary but let's be paranoid :)
       CORCOL=NCORR
       CORROW=NSAT
       LEN=CORCOL*CORROW
       CORSPC=LEN
       call chmalloc('ancsol.src','INTSTM','CORPT',CORrow,corcol,crlp=CORPT)
       corpt(1:corrow,1:corcol)=0
    ELSE
       CORCOL=0
       CORROW=0
       CORSPC=0
    ENDIF
    !
    !     and finally the SELECTION array which is NATOM long and holds
    !     the number representing the data/correlation for a selected atom
    SELCOL=NATOM
    SELROW=1
    LEN=SELCOL*SELROW
    SELSPC=LEN
    call chmalloc('ancsol.src','INTSTM','SELPT',SELrow,selcol,intgp=SELPT)
    CALL INTSEL(SELPT,SELCOL,NSAT,1,ATLST,NSAT)
    !
    RETURN
  END SUBROUTINE INTSTM
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTSTX(SERN,MXNTOT, &
       DATPT,DATCOL,DATROW,DATSPC, &
       CORPT,CORCOL,CORROW,CORSPC, &
       SELPT,SELCOL,SELROW,SELSPC, &
       NCORR,LFFT,NSAT,ATLST)
    !
    !     Initialize data, correl and selection arrays for one
    !     SATX series. This routine allocates the space and fills
    !     the selection pointer array.
    !
    !     Tibor Rudas 17. 05. 2004 -  June 2006
    !
  use number,only:zero
  use memory
  use psf
    !
    INTEGER SERN,MXNTOT
    real(chm_real), pointer,dimension(:,:) :: DATpt
    real(chm_real), pointer,dimension(:,:) :: corpt
    integer,pointer,dimension(:,:) :: selpt
    INTEGER DATCOL,DATROW,DATSPC
    INTEGER CORCOL,CORROW,CORSPC
    INTEGER SELCOL,SELROW,SELSPC
    INTEGER NCORR,NSAT,ATLST(*)
    LOGICAL LFFT
    INTEGER LEN
    !
    !     the DATA array which will hold our data
    DATCOL=NCORR
    IF(LFFT) DATCOL=MXNTOT
    DATROW=2*NSAT
    LEN=DATCOL*DATROW
    DATSPC=LEN
    call chmalloc('ancsol.src','INTSTX','DATPT',DATrow,datcol,crlp=DATPT)

    !     the CORREL array which holds the growing correlation function
    !     (only used when using windowing and not fft)
    IF(.NOT.LFFT) THEN
       CORCOL=NCORR
       CORROW=NSAT
       LEN=CORCOL*CORROW
       CORSPC=LEN
       call chmalloc('ancsol.src','INTstx','CORPT',CORrow,corcol,crlp=CORPT)
       corpt(1:corrow,1:corcol)=zero
    ELSE
       CORCOL=0
       CORROW=0
       CORSPC=0
    ENDIF
    !
    !     and finally the SELECTION array which is NATOM long and holds
    !     the number representing the data/correlation for a selected atom
    !     so: SEL(ATOM,1) holds entry for DATA and SEL(ATOM,2) for CORREL
    SELCOL=2
    SELROW=NATOM
    LEN=SELCOL*SELROW
    SELSPC=LEN
    call chmalloc('ancsol.src','INTstx','SELPT',SELrow,selcol,intgp=SELPT)
    CALL INTSL3(SELPT,SELROW,NSAT,2,1,ATLST)
    !
    RETURN
  END SUBROUTINE INTSTX
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTMRT(SERN,MXNTOT, &
       DATPT,DATCOL,DATROW,DATSPC, &
       CORPT,CORCOL,CORROW,CORSPC, &
       SELPT,SELCOL,SELROW,SELSPC, &
       NCORR,LFFT,NSAT,ATLST)
    !
    !     Initialize data, correl and selection arrays for one
    !     MRTI (mean residence time) series.
    !     This routine allocates the space and fills the selection pointer array.
    !
    !     Tibor Rudas 16. 03. 2004 - June 2006
    !
  use number
  use memory
  use psf
    !
    INTEGER SERN,MXNTOT
    integer,pointer,dimension(:,:) :: datpt,corpt,selpt
    INTEGER DATCOL,DATROW,datspc
    INTEGER CORCOL,CORROW,CORSPC
    INTEGER SELCOL,SELROW,SELSPC
    INTEGER NCORR,NSAT,ATLST(*)
    LOGICAL LFFT
    !
    !
    INTEGER LEN
    !
    !     this should be rather easy: we need three arrays:
    !     DATPT which holds the histogram how often a 'series' during
    !           which a selected atom was for x consecutive steps i the
    !           selected shells
    !     CORPT which holds for each atom the number of predecessing steps
    !           it has already been in the shell
    !     SELPT which will be filled with 0/1 if the atom is currently in the
    !           shell
    !
    DATCOL=NCORR
    IF(LFFT) DATCOL=MXNTOT
    DATROW=1
    LEN=DATCOL*DATROW
    DATSPC=LEN
    call chmalloc('ancsol.src','INTmrt','DATPT',DATrow,datcol,intgp=DATPT)
    DATPT(1:datrow,1:datcol)=0
    !
    CORCOL=NATOM
    CORROW=1
    LEN=CORCOL*CORROW
    CORSPC=LEN
    call chmalloc('ancsol.src','INTmrt','CORPT',CORrow,corcol,intgp=CORPT) 
    CORPT(1:CORrow,1:corcol)=0
    !
    SELCOL=NATOM
    SELROW=1
    LEN=SELCOL*SELROW
    SELSPC=LEN
    call chmalloc('ancsol.src','INTmrt','SELPT',SELrow,selcol,intgp=SELPT)
    SELPT(1:SELrow,1:selcol)=0
    !
    RETURN
  END SUBROUTINE INTMRT
  !
#endif /* (shell)*/
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTDDP(SERN,MXNTOT, &
       DATPT,DATCOL,DATROW,DATSPC, &
       CORPT,CORCOL,CORROW,CORSPC, &
       SELPT,SELCOL,SELROW,SELSPC, &
       NCORR,LFFT,NSAT,NSAT2,ATLST)
    !
    !     Initialize data, correl and selection arrays for one
    !     DDIP series. This routine allocates the space and fills
    !     the selection pointer array.
    !
    !     Tibor Rudas 20. 02. 2003 - June 2006
    !
  use number,only:zero
  use memory
  use psf
    !
    INTEGER SERN,MXNTOT
    real(chm_real), pointer,dimension(:,:) :: DATpt
    real(chm_real), pointer,dimension(:,:) :: corpt
    integer,pointer,dimension(:,:) :: selpt
    INTEGER DATCOL,DATROW,DATSPC
    INTEGER CORCOL,CORROW,CORSPC
    INTEGER SELCOL,SELROW,SELSPC
    INTEGER NCORR,NSAT,NSAT2,ATLST(*)
    LOGICAL LFFT
    !
    !
    INTEGER LEN,NPRAT,NSEAT
    !
    NPRAT=NSAT2
    NSEAT=NSAT-NSAT2
    !     the DATA array which will hold our data
    !     we need 4 series for each primary/secondary atom combination
    !     (x, y and z of the vector and its length)
    DATCOL=NCORR
    IF(LFFT) DATCOL=MXNTOT
    DATROW=4*NPRAT*NSEAT
    LEN=DATCOL*DATROW
    DATSPC=LEN
    call chmalloc('ancsol.src','INTddp','DATPT',DATrow,datcol,crlp=DATPT)

    !     the CORREL array which holds the growing correlation function
    !     (only used when using windowing and not fft)
    !     we need 3 series for each primary/secondary atom combination
    !     (1 x exact, and the 2 parts of the product approximation)
    IF(.NOT.LFFT) THEN
       !        this should not be necessary but let's be paranoid :)
       CORCOL=NCORR
       CORROW=3*NPRAT*NSEAT
       LEN=CORCOL*CORROW
       CORSPC=LEN
       call chmalloc('ancsol.src','INTddp','CORPT',CORrow,corcol,crlp=CORPT)
       corpt(1:corrow,1:corcol)=zero
    ELSE
       CORCOL=0
       CORROW=0
       CORSPC=0
    ENDIF
    !
    !     and finally the SELECTION array which is NATOM long and holds
    !     the number representing the data/correlation for a selected atom
    !     we need this 2 times: for primary and secondary atoms
    SELCOL=4
    SELROW=NATOM
    LEN=SELCOL*SELROW
    SELSPC=LEN
    call chmalloc('ancsol.src','INTddp','SELPT',SELrow,selcol,intgp=SELPT)
    CALL INTSL2(SELCOL,SELROW,SELPT,NPRAT,ATLST(1), &
         NSEAT,ATLST(NPRAT+1),DATROW,CORROW,LFFT)
    !
    RETURN
  END SUBROUTINE INTDDP
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTSEL(SEL,NAT,NLST,MULT,ATLST,MAX)
    !
    !     this routine fills the SEL array (of length NAT) with 0 for every
    !     atom which is not in ATLST and with a pointer to its data/correl
    !     series assuming that they both are MULT columns wide for one atom
    !     (used for: SATM)
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER SEL(*),NAT,NLST,MULT,ATLST(*),MAX
    !
    INTEGER I,K,PT
    !
    DO I=1,NAT
       SEL(I)=0
    ENDDO
    !
    PT=1
    DO I=1,NLST
       K=ATLST(I)
       SEL(K)=PT
       PT=PT+MULT
    ENDDO
    !
    !     (we should _never_ get here, but let's be paraoid)
    PT=PT-MULT
    IF(PT > MAX) &
         CALL WRNDIE(-5,'<INTSEL>', &
         'Assigned more arrays than there were!')
    !
    RETURN
  END SUBROUTINE INTSEL
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTSL2(COLS,ROWS,SEL,NPRM,PRMLST,NSEC,SECLST, &
       MAXDAT,MAXCOR,LFFT)
    !
    !     this routine fills the SEL array (of length 4*NAT)
    !     the first row holds the offset for primary atoms (NPRM,PRMLST) in
    !         the data structure and the third in the correl structure
    !     the second row holds a pointer to the first row for a secondary
    !         atom in the data and the fourth in the correl structure
    !
    !     so when rows 1+2 are added one should get the first row (X-comp)
    !     for a given primary-secondary combination) in the data structure
    !     (while 3+4 give the first correlation for this pair in the correl
    !     structure)
    !     
    !     (used for: DDIP)
    !
    !     Tibor Rudas 20. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER COLS,ROWS,SEL(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER MAXDAT,MAXCOR
    LOGICAL LFFT
    !
    INTEGER I,J,K,PT1,PT2,PT3,PT4
    LOGICAL OK

    SEL(1:rows,1:4)=-1

    !     first the primary atom pointers (row 1 and 3)
    DO I=1,NPRM
       PT1=(I-1)*NSEC*4
       PT3=(I-1)*NSEC*3
       K=PRMLST(I)
       SEL(K,1)=PT1
       SEL(K,3)=PT3
    ENDDO
    !
    !     now the secondary atom pointers (row 2 and 4)
    DO I=1,NSEC
       PT2=((I-1)*4)+1
       PT4=((I-1)*3)+1
       K=SECLST(I)
       SEL(K,2)=PT2
       SEL(K,4)=PT4
    ENDDO
    !
    !     (we should _never_ get here, but let's be paraoid)
    OK=((PT1+PT2) > MAXDAT)
    IF(.NOT.LFFT) OK=(OK.OR.((PT3+PT4) > MAXCOR))
    IF(OK) &
         CALL WRNDIE(-5,'<INTSL2>', &
         'Assigned more arrays than there were!')
    !
    RETURN
  END SUBROUTINE INTSL2
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTSL3(SEL,NAT,NLST,MULT1,MULT2,ATLST)
    !
    !     this routine fills the SEL array (of length NAT) with 0 for every
    !     atom which is not in ATLST and with a pointer to its data/correl
    !     series assuming that they both are MULT1 columns wide for one atom
    !     and MULT2 for the second row
    !     (used for: SATX)
    !
    !     Tibor Rudas Jun 1st 2004 - June 2006
    !
  use exfunc
    !
    INTEGER NAT,SEL(NAT,2),NLST,MULT1,MULT2,ATLST(*)
    !
    INTEGER I,K,PT1,PT2
    !
    SEL(1:nat,1)=0
    SEL(1:nat,2)=0
    !
    PT1=1
    PT2=1
    DO I=1,NLST
       K=ATLST(I)
       SEL(K,1)=PT1
       SEL(K,2)=PT2
       PT1=PT1+MULT1
       PT2=PT2+MULT2
    ENDDO
    !
    RETURN
  END SUBROUTINE INTSL3
  !
#if KEY_SHELL==1 /*shell*/
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILSAT(NAT,ACT,COLS,ROWS,DATS,SL,DATLST, &
       NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
    !
    !     this routine fills position ACT of the DATS array (which has ROWS
    !     rows and COLS columns) with 0 or 1 depending if the corresponding
    !     atom is in shell SL
    !
    !     we do this by first writing 0 to all data points and then looking
    !     for all atoms in shell SL if we need to enter a 1 (i.e. the
    !     corresponding entry in DATLST is not 0)
    !
    !     NSHL,NASHL,NSHBLK,SHLLST AND SHBLKL are the usual SHELL paramters
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER NAT,ACT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER SL,DATLST(*)
    INTEGER NSHL,NASHL(*),NSHBLK,SHLLST(NAT,NSHL),SHBLKL(*)
    !
    INTEGER I,K
    !
    DO I=1,ROWS
       DATS(I,ACT)=0
    ENDDO
    !
    IF(SL > 0) THEN
       DO I=1,NASHL(SL)
          K=SHLLST(I,SL)
          IF(DATLST(K) > 0) &
               DATS(DATLST(K),ACT)=1.0D0
       ENDDO
    ELSE IF(SL == -1) THEN
       !        BULK
       DO I=1,NSHBLK
          K=SHBLKL(I)
          IF(DATLST(K) > 0) &
               DATS(DATLST(K),ACT)=1.0D0
       ENDDO
    ELSE
       CALL WRNDIE(-5,'<FILSAT>', &
            'Shell error.')
    ENDIF
    !
    RETURN
  END SUBROUTINE FILSAT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILSAX(NAT,ACT,COLS,ROWS,DATS,SL,SL2,DATLST, &
       NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
    !
    !     this routine fills position ACT of the DATS array (which has ROWS
    !     rows and COLS columns) with 0 or 1 depending if the corresponding
    !     atom is in shell SL or SL2 to calculate cross corr func.
    !
    !     series: SATX
    !
    !     NSHL,NASHL,NSHBLK,SHLLST AND SHBLKL are the usual SHELL paramters
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER NAT,ACT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER SL,SL2,DATLST(NAT,2)
    INTEGER NSHL,NASHL(*),NSHBLK,SHLLST(NAT,NSHL),SHBLKL(*)
    !
    INTEGER I,K,KK,L
    INTEGER SLS(2),INC(2)
    !
    DO I=1,ROWS
       DATS(I,ACT)=0
    ENDDO
    !
    SLS(1) = SL
    SLS(2) = SL2
    INC(1) = 0
    INC(2) = 1
    !
    DO L=1,2
       IF(SLS(L) > 0) THEN
          DO I=1,NASHL(SLS(L))
             K=SHLLST(I,SLS(L))
             KK=DATLST(K,1)
             IF(KK > 0) &
                  DATS(KK+INC(L),ACT)=1.0D0
          ENDDO
       ELSE IF(SLS(L) == -1) THEN
          !           BULK
          DO I=1,NSHBLK
             K=SHBLKL(I)
             KK=DATLST(K,1)
             IF(KK > 0) &
                  DATS(KK+INC(L),ACT)=1.0D0
          ENDDO
       ELSE
          CALL WRNDIE(-5,'<FILSAX>', &
               'Shell error.')
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE FILSAX
  !
  !----------------------------------------------------------------------
  SUBROUTINE FILSDD(ACT,NAT,NIT,IMATTR, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST,NSEC,SECLST, &
       SELCOL,SELROW,SELLST, &
       SHELL,NSHL,NASHL,NSHBLK,SHLLST,SHBLKL, &
       TMPFLG,DOIMAGES)
    !
    !     Just like FILDDP but only for secondary atoms which are in shell
    !     SHELL.
    !
    !     NSHL,NASHL,NSHBLK,SHLLST AND SHBLKL are the usual SHELL paramters
    !
    !     Tibor Rudas 3. 03. 2003 - June 2006
    !
  use exfunc
  use coord
    !
    INTEGER ACT,NAT,NIT,IMATTR(*),COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER SELCOL,SELROW,SELLST(SELROW,SELCOL),SHELL
    INTEGER NSHL,NASHL(*),NSHBLK,SHLLST(NAT,NSHL),SHBLKL(*),TMPFLG(*)
    LOGICAL DOIMAGES
    !
    !
    INTEGER I,J,K,KK
    INTEGER N1,N2,NI
    real(chm_real) X1,Y1,Z1
    real(chm_real) DX,DY,DZ,LEN
    !
    !     now fill all prim-sec pairs in PRIMARY atom space
    DO I=1,NPRM
       N1=PRMLST(I)
       X1=X(N1)
       Y1=Y(N1)
       Z1=Z(N1)
       K=SELLST(N1,1)
       IF(SHELL > 0) THEN
          DO J=1,NASHL(SHELL)
             N2=SHLLST(J,SHELL)
             !              fill the vector and r^2 into DATS for quicker comparison
             !              when doing images. renormalization follows at the end
             !              if there is a ii combination-fill with zero
             IF(SELLST(N2,1) >= 0) THEN
                KK=K+SELLST(N2,2)
                IF(N1 /= N2) THEN
                   DX=X1-X(N2)
                   DY=Y1-Y(N2)
                   DZ=Z1-Z(N2)
                   DATS(KK,  ACT)=DX
                   DATS(KK+1,ACT)=DY
                   DATS(KK+2,ACT)=DZ
                   DATS(KK+3,ACT)=(DX*DX)+(DY*DY)+(DZ*DZ)
                ELSE
                   DATS(KK,  ACT)=0.0D0
                   DATS(KK+1,ACT)=0.0D0
                   DATS(KK+2,ACT)=0.0D0
                   DATS(KK+3,ACT)=0.0D0
                ENDIF
             ENDIF
          ENDDO
       ELSE IF(SHELL == -1) THEN
          !           BULK
          DO J=1,NSHBLK
             N2=SHBLKL(J)
             IF(SELLST(N2,1) >= 0) THEN
                KK=K+SELLST(N2,2)
                IF(N1 /= N2) THEN
                   DX=X1-X(N2)
                   DY=Y1-Y(N2)
                   DZ=Z1-Z(N2)
                   DATS(KK,  ACT)=DX
                   DATS(KK+1,ACT)=DY
                   DATS(KK+2,ACT)=DZ
                   DATS(KK+3,ACT)=(DX*DX)+(DY*DY)+(DZ*DZ)
                ELSE
                   DATS(KK,  ACT)=0.0D0
                   DATS(KK+1,ACT)=0.0D0
                   DATS(KK+2,ACT)=0.0D0
                   DATS(KK+3,ACT)=0.0D0
                ENDIF
             ENDIF
          ENDDO
       ELSE
          CALL WRNDIE(-5,'<FILSAT>', &
               'Shell error.')
       ENDIF
    ENDDO
    !
    !     and now try minimum image for all atoms in IMAGE space
    IF(DOIMAGES) THEN
       !
       !        prepare a lookuptable for all atoms in the shell
       DO I=1,NAT
          TMPFLG(I)=-1
       ENDDO
       IF(SHELL > 0) THEN
          DO I=1,NASHL(SHELL)
             K=SHLLST(I,SHELL)
             TMPFLG(K)=1
          ENDDO
       ELSE IF(SHELL == -1) THEN
          !           BULK
          DO I=1,NSHBLK
             K=SHBLKL(I)
             TMPFLG(K)=2
          ENDDO
       ENDIF
       !
       DO I=(NAT+1),NIT
          NI=IMATTR(I)
          !
          IF(SELLST(NI,1) >= 0) THEN
             !              i.e. I is an image of a primary atom
             X1=X(I)
             Y1=Y(I)
             Z1=Z(I)
             K=SELLST(NI,1)
             IF(SHELL > 0) THEN
                DO J=1,NASHL(SHELL)
                   N2=SHLLST(J,SHELL)
                   IF(SELLST(N2,1) >= 0) THEN
                      KK=K+SELLST(N2,2)
                      IF(N1 /= N2) THEN
                         DX=X1-X(N2)
                         DY=Y1-Y(N2)
                         DZ=Z1-Z(N2)
                         LEN=(DX*DX)+(DY*DY)+(DZ*DZ)
                      ENDIF
                      IF(LEN < DATS(KK+3,ACT)) THEN
                         DATS(KK  ,ACT)=DX
                         DATS(KK+1,ACT)=DY
                         DATS(KK+2,ACT)=DZ
                         DATS(KK+3,ACT)=LEN
                      ENDIF
                   ENDIF
                ENDDO
             ELSE IF(SHELL == -1) THEN
                !                 BULK
                DO J=1,NSHBLK
                   N2=SHBLKL(J)
                   IF(SELLST(N2,1) >= 0) THEN
                      KK=K+SELLST(N2,2)
                      IF(N1 /= N2) THEN
                         DX=X1-X(N2)
                         DY=Y1-Y(N2)
                         DZ=Z1-Z(N2)
                         LEN=(DX*DX)+(DY*DY)+(DZ*DZ)
                      ENDIF
                      IF(LEN < DATS(KK+3,ACT)) THEN
                         DATS(KK  ,ACT)=DX
                         DATS(KK+1,ACT)=DY
                         DATS(KK+2,ACT)=DZ
                         DATS(KK+3,ACT)=LEN
                      ENDIF
                   ENDIF
                ENDDO
             ENDIF
          ENDIF
          !
          IF(TMPFLG(NI) > 0) THEN
             !              i.e. I is an image of an atom in SHELL
             IF(SELLST(NI,2) >= 0) THEN
                !                 and it is also a secondary atom
                X1=X(I)
                Y1=Y(I)
                Z1=Z(I)
                K=SELLST(NI,2)
                DO J=1,NPRM
                   N1=PRMLST(J)
                   IF(N1 /= NI) THEN
                      KK=SELLST(N1,1)+K
                      DX=X(N1)-X1
                      DY=Y(N1)-Y1
                      DZ=Z(N1)-Z1
                      LEN=(DX*DX)+(DY*DY)+(DZ*DZ)
                      IF(LEN < DATS(KK+3,ACT)) THEN
                         DATS(KK  ,ACT)=DX
                         DATS(KK+1,ACT)=DY
                         DATS(KK+2,ACT)=DZ
                         DATS(KK+3,ACT)=LEN
                      ENDIF
                   ENDIF
                ENDDO
             ENDIF
          ENDIF
          !
       ENDDO
    ENDIF
    !
    !     and finally: renormalize the vector component to form a unit vector
    !     and compute the distance
    DO I=1,NPRM
       N1=PRMLST(I)
       DO J=1,NSEC
          N2=SECLST(J)
          IF(N1 /= N2) THEN
             K=SELLST(N1,1)+SELLST(N2,2)
             LEN=DSQRT(DATS(KK+3,ACT))
             DATS(KK,  ACT)=(DATS(KK,  ACT)/LEN)
             DATS(KK+1,ACT)=(DATS(KK+1,ACT)/LEN)
             DATS(KK+2,ACT)=(DATS(KK+2,ACT)/LEN)
             DATS(KK+3,ACT)=1.0D0/(LEN**3)
          ENDIF
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE FILSDD
  !
  !----------------------------------------------------------------------
  SUBROUTINE FILMRT(NATM,ATLST,NAT,SHELL, &
       NHIST,HISTO,NREC,RECENT,NNOW,NOW, &
       NSHL,NASHL,NSHBLK,SHLLST,SHBLKL)
    !
    !     this routine processes one frame for a MRTI (mean residence time)
    !     time series.
    !     it first fills NOW with 0/1 for all atoms in shell SHELL and then
    !     does a lookup for all NAT atomnumbers in ATLST.
    !     if NOW == 0 and RECENT  > 0 :
    !            atom was for RECENT timesteps in this shell but is not now,
    !            so increase histogram HIST(RECENT(atom)) by 1 and reset
    !            RECENT(atom) = 0 (atom begins a "new life" if it reenters
    !            the shell)
    !     if NOW == 1 and RECENT >= 0 :
    !            atom enters/stays in shell -> increase RECENT
    !    (if NOW == 0 and RECENT == 0 :
    !            do nothing)
    !
    !     NSHL,NASHL,NSHBLK,SHLLST AND SHBLKL are the usual SHELL paramters
    !
    !     Tibor Rudas 16. 03. 2004 - June 2006
    !
  use exfunc
    !
    INTEGER NATM,ATLST(*),NAT,SHELL
    INTEGER NHIST,HISTO(*),NREC,RECENT(*),NNOW,NOW(*)
    INTEGER NSHL,NASHL(*),NSHBLK,SHLLST(NATM,NSHL),SHBLKL(*)
    !
    INTEGER I,K
    !
    !     first: zero out NOW and fill in appropriate 1s
    NOW(1:NNOW) = 0
    DO I=1,NASHL(SHELL)
       NOW(SHLLST(I,SHELL))=1
    ENDDO
    !
    !     and now check for all relevant atoms: content of NOW, then
    !     content of RECENT and act accordingly
    DO I=1,NAT
       K=ATLST(I)
       IF(NOW(K) == 1) THEN
          !           atom remains in or enters shell
          RECENT(K)=RECENT(K)+1
       ELSE
          IF(RECENT(K) > 0) THEN
             !              i.e. atom was in a shell up to now -> increase histo and
             !                                 clear counter
             HISTO(RECENT(K))=HISTO(RECENT(K))+1
             RECENT(K)=0
          ENDIF
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE FILMRT
  !
#endif /* (shell)*/
  !----------------------------------------------------------------------
  SUBROUTINE FILDDP(ACT,NAT,NIT,IMATTR, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST,NSEC,SECLST, &
       SELCOL,SELROW,SELLST,DOIMAGES)
    !
    !     this routine fills position ACT of the DATS array (which has ROWS
    !     rows and COLS columns). For each PRiMary -SECondary atom combination
    !     the 3 components of the unit connection vector and their distance
    !     are recorded. If DOIMAGES is .TRUE. images are also sampled to assure
    !     minimum image conditions and avoid "jumps" in the vector.
    !
    !     This data is necessary to compute dipole-dipole couplings
    !
    !     PRMLST of length NPRM holds the primary atom numbers and SECLST of
    !     length NSEC the secondary atoms.
    !     SELLST is a 4-dim array of which only rows 1 & 2 are used here. The
    !     first row holds the primary atom offset in DATS and the second the
    !     secondary atom offset from the start of the primary atom data segment.
    !     So 1+2 should point to the X-component of this pair which is followed
    !     by Y, Z and R.
    !
    !     Tibor Rudas 21. 02. 2003 - June 2006
    !
  use exfunc
  use coord
    !
    INTEGER ACT,NAT,NIT,IMATTR(*),COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER SELCOL,SELROW,SELLST(SELROW,SELCOL)
    LOGICAL DOIMAGES
    !
    !
    INTEGER I,J,K,KK
    INTEGER N1,N2,NI
    real(chm_real) X1,Y1,Z1
    real(chm_real) DX,DY,DZ,LEN
    !
    !     now fill all prim-sec pairs in PRIMARY atom space
    DO I=1,NPRM
       N1=PRMLST(I)
       X1=X(N1)
       Y1=Y(N1)
       Z1=Z(N1)
       K=SELLST(N1,1)
       DO J=1,NSEC
          !           fill the vector and r^2 into DATS for quicker comparison
          !           when doing images. renormalization follows at the end
          !           if there is a ii combination-fill with zero
          N2=SECLST(J)
          KK=K+SELLST(N2,2)
          IF(N1 /= N2) THEN
             DX=X1-X(N2)
             DY=Y1-Y(N2)
             DZ=Z1-Z(N2)
             DATS(KK,  ACT)=DX
             DATS(KK+1,ACT)=DY
             DATS(KK+2,ACT)=DZ
             DATS(KK+3,ACT)=(DX*DX)+(DY*DY)+(DZ*DZ)
          ELSE
             DATS(KK,  ACT)=0.0D0
             DATS(KK+1,ACT)=0.0D0
             DATS(KK+2,ACT)=0.0D0
             DATS(KK+3,ACT)=0.0D0
          ENDIF
       ENDDO
    ENDDO
    !
    !     and now try minimum image for all atoms in IMAGE space
    IF(DOIMAGES) THEN
       DO I=(NAT+1),NIT
          NI=IMATTR(I)
          !
          IF(SELLST(NI,1) >= 0) THEN
             !              i.e. I is an image of a primary atom
             X1=X(I)
             Y1=Y(I)
             Z1=Z(I)
             K=SELLST(NI,1)
             DO J=1,NSEC
                N2=SECLST(J)
                IF(NI /= N2) THEN
                   KK=K+SELLST(N2,2)
                   DX=X1-X(N2)
                   DY=Y1-Y(N2)
                   DZ=Z1-Z(N2)
                   LEN=(DX*DX)+(DY*DY)+(DZ*DZ)
                   IF(LEN < DATS(KK+3,ACT)) THEN
                      DATS(KK  ,ACT)=DX
                      DATS(KK+1,ACT)=DY
                      DATS(KK+2,ACT)=DZ
                      DATS(KK+3,ACT)=LEN
                   ENDIF
                ENDIF
             ENDDO
          ENDIF
          !
          IF(SELLST(NI,2) >= 0) THEN
             !              i.e. I is an image of a secondary atom
             X1=X(I)
             Y1=Y(I)
             Z1=Z(I)
             K=SELLST(NI,2)
             DO J=1,NPRM
                N1=PRMLST(J)
                IF(N1 /= NI) THEN
                   KK=SELLST(N1,1)+K
                   DX=X(N1)-X1
                   DY=Y(N1)-Y1
                   DZ=Z(N1)-Z1
                   LEN=(DX*DX)+(DY*DY)+(DZ*DZ)
                   IF(LEN < DATS(KK+3,ACT)) THEN
                      DATS(KK,  ACT)=DX
                      DATS(KK+1,ACT)=DY
                      DATS(KK+2,ACT)=DZ
                      DATS(KK+3,ACT)=LEN
                   ENDIF
                ENDIF
             ENDDO
          ENDIF
          !
       ENDDO
    ENDIF
    !
    !     and finally: renormalize the vector component to form a unit vector
    !     and compute the distance
    DO I=1,NPRM
       N1=PRMLST(I)
       K=SELLST(N1,1)
       DO J=1,NSEC
          N2=SECLST(J)
          IF(N1 /= N2) THEN
             KK=K+SELLST(N2,2)
             LEN=DSQRT(DATS(KK+3,ACT))
             DATS(KK,  ACT)=(DATS(KK,  ACT)/LEN)
             DATS(KK+1,ACT)=(DATS(KK+1,ACT)/LEN)
             DATS(KK+2,ACT)=(DATS(KK+2,ACT)/LEN)
             DATS(KK+3,ACT)=1.0D0/(LEN**3)
          ENDIF
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE FILDDP
  !
  !----------------------------------------------------------------------
  SUBROUTINE FILVAC(ACT,NAT,SERVAL, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST, &
       SELCOL,SELROW,SELLST)
    !
    !     this routine fills position ACT of the DATS array (which has ROWS
    !     rows and COLS columns). For each atom in PRMLST the velocity vector
    !     (which is simply x/y/z when reading a vel-traj) are stored and
    !     correlated.
    !
    !     Tibor Rudas Jun 2004 - June 2006
    !
  use exfunc
  use coord
  use psf
    !
    INTEGER ACT,NAT,COLS,ROWS
    real(chm_real) SERVAL
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*)
    INTEGER SELCOL,SELROW,SELLST(SELROW,SELCOL)
    !
    !
    INTEGER I,N1,K
    real(chm_real) CHRG
    !
    DO I=1,NPRM
       N1=PRMLST(I)
       K=SELLST(N1,1)
       CHRG=1.0D0
       IF(SERVAL > 0) CHRG=CG(N1)
       DATS(K,  ACT)=X(N1)*CHRG
       DATS(K+1,ACT)=Y(N1)*CHRG
       DATS(K+2,ACT)=Z(N1)*CHRG
    ENDDO
    !
    RETURN
  END SUBROUTINE FILVAC
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILWRO(ACT,NAT,SERVAL, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST, &
       SELCOL,SELROW,SELLST)
    !
    !     this routine fills position ACT of the DATS array (which has ROWS
    !     rows and COLS columns). 
    !     
    !     
    !
    !     Tibor Rudas Sep. 2004 - June 2006
    !
  use exfunc
  use coord
  use psf
    !
    INTEGER ACT,NAT,COLS,ROWS
    real(chm_real) SERVAL
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*)
    INTEGER SELCOL,SELROW,SELLST(SELROW,SELCOL)
    !
    !
    INTEGER I,N1,K
    real(chm_real) XD,YD,ZD,LEND
    LOGICAL QNORM
    !
    QNORM=.FALSE.
    IF(SERVAL > 0.0D0) QNORM=.TRUE.
    !
    DO I=1,NPRM
       N1=PRMLST(I)
       K=SELLST(N1,1)
       CALL GWDIP(N1,X,Y,Z,CG,XD,YD,ZD,LEND,QNORM)
       DATS(K,  ACT)=XD
       DATS(K+1,ACT)=YD
       DATS(K+2,ACT)=ZD
    ENDDO
    !
    RETURN
  END SUBROUTINE FILWRO
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE ZERSDD(ACT,COLS,ROWS,DATS,NPRM,PRMLST,NSEC,SECLST, &
       SELCOL,SELROW,SELLST)
    !
    !     Zeroes out all rows in the ACTth column (not all rows will be written
    !     to, only the ones with atoms wich may be in the chosen shell).
    !
    !     Tibor Rudas 4. 03. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER ACT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER SELCOL,SELROW,SELLST(SELROW,SELCOL)
    !
    INTEGER I,J,K,KK
    INTEGER N1,N2,NI
    !
    DO I=1,NPRM
       N1=PRMLST(I)
       K=SELLST(N1,1)
       DO J=1,NSEC
          N2=SECLST(J)
          KK=K+SELLST(N2,2)
          DATS(KK,  ACT)=0.0D0
          DATS(KK+1,ACT)=0.0D0
          DATS(KK+2,ACT)=0.0D0
          DATS(KK+3,ACT)=0.0D0
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE ZERSDD
  !
  !----------------------------------------------------------------------
#if KEY_SHELL==1 /*shell*/
  !
  SUBROUTINE SATWIN(ACT,NDAT,COLS,ROWS,DATS,COLS2,CORRS)
    !
    !     this routine does the windowing autocorrelation function for
    !     DATS(COLS,ROWS) represents the data and CORRS(COLS2,ROWS) holds
    !     the correlation functions. The data array holds NDAT data points
    !     up to now
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER ACT,NDAT,COLS,ROWS
    INTEGER COLS2
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) CORRS(ROWS,COLS2)
    !
    INTEGER I,AC2,MAX,ATM
    !
    MAX=COLS2
    IF(NDAT < COLS2) MAX=NDAT
    DO I=1,MAX
       AC2=ACT-I+1
       IF(AC2 < 1) AC2=AC2+NDAT
       DO ATM=1,ROWS
          CORRS(ATM,I)=CORRS(ATM,I)+(DATS(ATM,ACT)*DATS(ATM,AC2))
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SATWIN
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SATFFT(NDAT,COLS,ROWS,DATS,NCORR)
    !
    !     this routine does fft correlation of all single ROW data sets
    !     for a SATM time series
    !     (the row arrangement of the data is unfavourable but preferable
    !     in the windowing process which needs to be done in each step)
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use memory
    INTEGER NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR
    !
    !
    INTEGER I,NTOT2,LFA
    real(chm_real),allocatable,dimension(:) :: SFA,SFB,SFC,SFD
    !
    !     find number of points we want to keep
    !     (highest power of 2 lower than half the data points)
    IF(NCORR <= 0) THEN
       NCORR=2
       DO WHILE(NCORR <= NDAT)
          NCORR=NCORR*2
       ENDDO
       NCORR=NCORR/4
    ENDIF
    !
    !     find the length of the fft arrays
    NTOT2=2
    DO WHILE(NTOT2 < (NCORR+NDAT))
       NTOT2=NTOT2*2
    ENDDO
    !
    !     allocate space
    LFA=NTOT2
    call chmalloc('ancsol.src','SATFFT','SFA',LFA,crl=SFA)
    call chmalloc('ancsol.src','SATFFT','SFB',LFA,crl=SFB)
    call chmalloc('ancsol.src','SATFFT','SFC',LFA,crl=SFC)
    call chmalloc('ancsol.src','SATFFT','SFD',LFA,crl=SFD)
    !
    !     and now call the single series fft for each set
    DO I=1,ROWS
       CALL SATFF1(I,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
            SFA,SFB,SFC,SFD)
    ENDDO
    call chmdealloc('ancsol.src','SATFFT','SFA',LFA,crl=SFA)
    call chmdealloc('ancsol.src','SATFFT','SFB',LFA,crl=SFB)
    call chmdealloc('ancsol.src','SATFFT','SFC',LFA,crl=SFC)
    call chmdealloc('ancsol.src','SATFFT','SFD',LFA,crl=SFD)
    RETURN
  END SUBROUTINE SATFFT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SATFF1(ACT,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
       FA,FB,FC,FD)
    !
    !     this routine does fft correlation of one single ROW data set
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER ACT,NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NTOT2
    real(chm_real) FA(*),FB(*),FC(*),FD(*)
    !
    INTEGER NDATP1,I
    !
    NDATP1=NDAT+1
    !
    !     fill data array
    DO I=1,NDAT
       FA(I)=DATS(ACT,I)
       FB(I)=0.0D0
    ENDDO
    !     zero fill
    DO I=NDATP1,NTOT2
       FA(I)=0.0D0
       FB(I)=0.0D0
    ENDDO
    !
    !     fft transform
    CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
    !
    !     fill result arrays
    DO I=1,NTOT2
       FC(I)=(FA(I)*FA(I))+(FB(I)*FB(I))
       FD(I)=0.0D0
    ENDDO
    !
    !     call inverse fft
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    !     and now fill resulting autocorrelation function into the DATS
    !     array (saves space)
    DO I=1,NDAT
       DATS(ACT,I)=((FC(I)/NTOT2)/(NDATP1-I))
    ENDDO
    !
    RETURN
  END SUBROUTINE SATFF1
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SAXWIN(NAT,ACT,NDAT,NSEC,SECLST, &
       COLS,ROWS,DATS, &
       COLS2,ROWS2,CORRS,DATLST)
    !
    !     this routine does the windowing autocorrelation function for
    !     DATS(COLS,ROWS) represents the data and CORRS(COLS2,ROWS2) holds
    !     the correlation functions. The data array holds NDAT data points
    !     up to now
    !     (for this series the ordering of data/corrs is unfavourable since
    !     we have to go row by row... could be improved)
    !
    !     series: SATX
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER NAT,ACT,NDAT,NSEC,SECLST(*),COLS,ROWS
    INTEGER COLS2,ROWS2
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) CORRS(ROWS2,COLS2)
    INTEGER DATLST(NAT,2)
    !
    INTEGER I,J,K,KD,KC,AC2,MAX,ATM
    real(chm_real) XTMP
    !
    MAX=COLS2
    IF(NDAT < COLS2) MAX=NDAT
    !
    DO J=1,NSEC
       K=SECLST(J)
       KD=DATLST(K,1)
       KC=DATLST(K,2)
       XTMP=DATS(KD+1,ACT)
       !
       DO I=1,MAX
          AC2=ACT-I+1
          IF(AC2 < 1) AC2=AC2+NDAT
          !tr this would be: looking 'backward' from both new points and
          !tr taking an average
          !tr CORRS(KC,I)=CORRS(KC,I)+(DATS(KD,ACT)*DATS(KD+1,AC2))*0.5D0
          !tr CORRS(KC,I)=CORRS(KC,I)+(DATS(KD+1,ACT)*DATS(KD,AC2))*0.5D0
          !tr this is analogous to corfun.src: look NCORR steps backward
          !tr from the latest point in the second series
          CORRS(KC,I)=CORRS(KC,I)+(DATS(KD,AC2)*XTMP)
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SAXWIN
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SAXFFT(NAT,NDAT,NSEC,SECLST, &
       COLS,ROWS,DATS,NCORR,SELLST)
    !
    !     this routine does fft correlation of all single ROW data sets
    !     for a SATM time series
    !     (the row arrangement of the data is unfavourable but preferable
    !     in the windowing process which needs to be done in each step)
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use memory
    !
    real(chm_real),allocatable,dimension(:),target :: space_SF
    INTEGER NAT,NDAT,NSEC,SECLST(*),COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,SELLST(NAT,2)
    !
    !
    INTEGER I,NTOT2,LFA,K,KK,FIL
    real(chm_real),pointer,dimension(:) :: SFA,SFB,SFC,SFD,SFE,SFF
    !
    !     find number of points we want to keep
    !     (highest power of 2 lower than half the data points)
    IF(NCORR <= 0) THEN
       NCORR=2
       DO WHILE(NCORR <= NDAT)
          NCORR=NCORR*2
       ENDDO
       NCORR=NCORR/4
    ENDIF
    !
    !     find the length of the fft arrays
    NTOT2=2
    DO WHILE(NTOT2 < (NCORR+NDAT))
       NTOT2=NTOT2*2
    ENDDO
    !
    !     allocate space
    LFA=NTOT2
    call chmalloc('ancsol.src','SAXFFT','space_sf',6*LFA,crl=space_sf)
    SFA => space_sf(0*lfa+1 : 1*lfa)
    SFB => space_sf(1*lfa+1 : 2*lfa)
    SFC => space_sf(2*lfa+1 : 3*lfa)
    SFD => space_sf(3*lfa+1 : 4*lfa)
    SFE => space_sf(4*lfa+1 : 5*lfa)
    SFF => space_sf(5*lfa+1 : 6*lfa)
    !
    !     and now call the single series fft for each set
    FIL=1
    DO I=1,NSEC
       K=SECLST(I)
       KK=SELLST(K,1)
       CALL SAXFF1(KK,FIL,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
            SFA,SFB,SFC,SFD, &
            SFE,SFF)
       FIL=FIL+1
    ENDDO
    !
    call chmdealloc('ancsol.src','SAXFFT','space_sf',6*LFA,crl=space_sf)
    RETURN
  END SUBROUTINE SAXFFT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SAXFF1(ACT,FIL,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
       FA,FB,FC,FD,FE,FF)
    !
    !     this routine does fft correlation of one single ROW data set
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER ACT,FIL,NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NTOT2
    real(chm_real) FA(*),FB(*),FC(*),FD(*),FE(*),FF(*)
    !
    INTEGER NDATP1,I
    !
    NDATP1=NDAT+1
    !
    !     fill data arrays
    DO I=1,NDAT
       FA(I)=DATS(ACT,I)
       FB(I)=0.0D0
       FC(I)=DATS(ACT+1,I)
       FD(I)=0.0D0
    ENDDO
    !     zero fill
    DO I=NDATP1,NTOT2
       FA(I)=0.0D0
       FB(I)=0.0D0
       FC(I)=0.0D0
       FD(I)=0.0D0
    ENDDO
    !
    !     fft transform
    CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,1)
    !
    !     fill result arrays
    DO I=1,NTOT2
       FE(I)=(FA(I)*FC(I))+(FB(I)*FD(I))
       FF(I)=(FA(I)*FD(I))-(FB(I)*FC(I))
    ENDDO
    !
    !     call inverse fft
    CALL correl_FFT(FE,FF,NTOT2,NTOT2,NTOT2,-1)
    !
    !     and now fill resulting autocorrelation function into the DATS
    !     array (saves space)
    DO I=1,NDAT
       DATS(FIL,I)=((FE(I)/NTOT2)/(NDATP1-I))
    ENDDO
    !
    RETURN
  END SUBROUTINE SAXFF1
  !
#endif /* (shell)*/
  !----------------------------------------------------------------------
  !
  SUBROUTINE DDPWIN(ACT,NDAT,NPRM,PRMLST,NSEC,SECLST, &
       COLS,ROWS,DATS, &
       COLS2,ROWS2,CORRS, &
       COLS3,ROWS3,SELLST)
    !
    !
    !     this routine does the windowing autocorrelation function for
    !     DATS(COLS,ROWS) represents the data and CORRS(COLS2,ROWS2) holds
    !     the correlation functions. The data array holds NDAT data points
    !     up to now
    !
    !     SELLST is a 4-dim array. The first row holds the primary atom
    !     offset in DATS and the second the secondary atom offset from the
    !     start of the primary atom data segment.  So 1+2 should point to
    !     the X-component of this pair which is followed by Y, Z and R.
    !     Rows 3 and 4 hold the same for the CORRS array.
    !
    !     Tibor Rudas 21. 02. 2003 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,NDAT,NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER COLS,ROWS
    INTEGER COLS2,ROWS2
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) CORRS(ROWS2,COLS2)
    INTEGER COLS3,ROWS3,SELLST(ROWS3,COLS3)
    !
    !
    INTEGER I,AC2,MAX
    INTEGER II,JJ,N1,N2,K,L
    real(chm_real) XTMP,XTMP2
    !
    MAX=COLS2
    IF(NDAT < COLS2) MAX=NDAT
    DO I=1,MAX
       AC2=ACT-I+1
       IF(AC2 < 1) AC2=AC2+NDAT
       DO II=1,NPRM
          N1=PRMLST(II)
          DO JJ=1,NSEC
             N2=SECLST(JJ)
             K=SELLST(N1,1)+SELLST(N2,2)
             L=SELLST(N1,3)+SELLST(N2,4)
             XTMP=( DATS(K  ,ACT) * DATS(K  ,AC2) )+ &
                  ( DATS(K+1,ACT) * DATS(K+1,AC2) )+ &
                  ( DATS(K+2,ACT) * DATS(K+2,AC2) )
             XTMP=(ONEPT5*(XTMP*XTMP))-HALF
             XTMP2=DATS(K+3,ACT)*DATS(K+3,AC2)
             CORRS(L  ,I)=CORRS(L,I)+(XTMP*XTMP2)
             CORRS(L+1,I)=CORRS(L+1,I)+XTMP2
             CORRS(L+2,I)=CORRS(L+2,I)+XTMP
          ENDDO
       ENDDO
    ENDDO
    !
    !
    RETURN
  END SUBROUTINE DDPWIN
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE DDPFFT(NDAT,COLS,ROWS,DATS, &
       NPRM,PRMLST,NSEC,SECLST, &
       COLS2,ROWS2,SELLST,NCORR)
    !
    !     this routine does fft correlation of all 4 column data sets
    !     for a DDIP time series
    !     (the row arrangement of the data is unfavourable but preferable
    !     in the windowing process which needs to be done in each step)
    !
    !     Tibor Rudas 21. 02. 2003 - June 2006
    !
  use memory
    !
    INTEGER NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER COLS2,ROWS2,SELLST(ROWS2,COLS2)
    INTEGER NCORR
    !
    !
    INTEGER I,J,K,N1,N2,NTOT2,LFA,LFA2
    real(chm_real),allocatable,dimension(:),target :: space_SF
    real(chm_real),allocatable,dimension(:),target :: space_c
    real(chm_real),pointer,dimension(:) :: SFA,SFB,SFC,SFD,C1,C2,C3
    integer PT
    !
    !     find number of points we want to keep
    !     (highest power of 2 lower than half the data points)
    IF(NCORR <= 0) THEN
       NCORR=2
       DO WHILE(NCORR <= NDAT)
          NCORR=NCORR*2
       ENDDO
       NCORR=NCORR/4
    ENDIF
    !
    !     find the length of the fft arrays
    NTOT2=2
    DO WHILE(NTOT2 < (NCORR+NDAT))
       NTOT2=NTOT2*2
    ENDDO
    !
    !     allocate space
    LFA=NTOT2
    call chmalloc('ancsol.src','ddpFFT','space_sf',4*LFA,crl=space_sf)
    SFA => space_sf(0*lfa+1 : 1*lfa)
    SFB => space_sf(1*lfa+1 : 2*lfa)
    SFC => space_sf(2*lfa+1 : 3*lfa)
    SFD => space_sf(3*lfa+1 : 4*lfa)
    !
    LFA2=NCORR
    call chmalloc('ancsol.src','SAXFFT','space_c',3*LFA2,crl=space_c)
    c1 => space_c(0*lfa2+1 : 1*lfa2)
    c2 => space_c(1*lfa2+1 : 2*lfa2)
    c3 => space_c(2*lfa2+1 : 3*lfa2)

    !     and now call the single series fft for each set
    PT=1
    DO I=1,NPRM
       N1=PRMLST(I)
       DO J=1,NSEC
          N2=SECLST(J)
          K=SELLST(N1,1)+SELLST(N2,2)
          CALL DDPFF1(K,PT,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
               SFA,SFB,SFC,SFD, &
               C1,C2,C3)
          PT=PT+3
       ENDDO
    ENDDO
    call chmdealloc('ancsol.src','ddpFFT','space_sf',4*LFA,crl=space_sf)
    call chmdealloc('ancsol.src','ddpFFT','space_c',3*LFA2,crl=space_c)
    RETURN
  END SUBROUTINE DDPFFT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE DDPFF1(ACT,FIL,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
       FA,FB,FC,FD,C1,C2,C3)
    !
    !     this routine does fft correlation of one 4 ROW data set
    !
    !     Tibor Rudas 21. 02. 2003 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,FIL,NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NTOT2
    real(chm_real) FA(*),FB(*),FC(*),FD(*)
    real(chm_real) C1(*),C2(*),C3(*)
    !
    !
    INTEGER NDATP1,I,K,K1,K2
    real(chm_real) SQRT2
    !
    NDATP1=NDAT+1
    SQRT2=SQRT(2.0D0)
    !
    !     NEEDED: here we accumulate fourier results in FC for part I
    !     zero out result array
    DO I=1,NTOT2
       FC(I)=0.0D0
       FD(I)=0.0D0
    ENDDO
    !
    !
    !     I: first part of exact result into C1
    !
    !     now loop over all components: K1 is one of x/y/z K2 another
    !     FA always holds the self-term and FB the cross-term
    !
    DO K=1,3
       K1=K-1
       K2=K1+1
       IF(K1 == 2) K2=0
       !        fill data array
       DO I=1,NDAT
          FA(I)=DATS(ACT+K1,I)*DATS(ACT+K1,I)*DATS(ACT+3,I)
          FB(I)=SQRT2*DATS(ACT+K1,I)*DATS(ACT+K2,I)*DATS(ACT+3,I)
       ENDDO
       !        zero fill
       DO I=NDATP1,NTOT2
          FA(I)=0.0D0
          FB(I)=0.0D0
       ENDDO
       !
       !        fft transform
       CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
       !
       !        fill result arrays
       DO I=1,NTOT2
          FC(I)=FC(I)+(FA(I)*FA(I))+(FB(I)*FB(I))
       ENDDO
    ENDDO
    !
    !     call inverse fft
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    !     and now fill the preliminary result into C1
    DO I=1,NCORR
       C1(I)=((ONEPT5*FC(I)/NTOT2)/(NDATP1-I))
    ENDDO
    !
    !
    !     II: < 1/r_ij^3(0) 1/r_ij^3(t) > into C2
    !
    !     fill data array
    DO I=1,NDAT
       FA(I)=DATS(ACT+3,I)
       FB(I)=0.0D0
    ENDDO
    !     zero fill
    DO I=NDATP1,NTOT2
       FA(I)=0.0D0
       FB(I)=0.0D0
    ENDDO
    !
    !     fft transform
    CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
    !
    !     fill result arrays
    DO I=1,NTOT2
       FC(I)=(FA(I)*FA(I))+(FB(I)*FB(I))
       FD(I)=0.0D0
    ENDDO
    !
    !     call inverse fft
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    !     and now fill the result into C2
    DO I=1,NCORR
       C2(I)=((FC(I)/NTOT2)/(NDATP1-I))
    ENDDO
    !
    !
    !     III: P2(theta) into C3
    !
    !     zero out result array
    DO I=1,NTOT2
       FC(I)=0.0D0
       FD(I)=0.0D0
    ENDDO
    !
    !     now loop over all components: K1 is one of x/y/z K2 another
    !     FA always holds the self-term and FB the cross-term
    !
    DO K=1,3
       K1=K-1
       K2=K1+1
       IF(K1 == 2) K2=0
       !        fill data array
       DO I=1,NDAT
          FA(I)=DATS(ACT+K1,I)*DATS(ACT+K1,I)
          FB(I)=SQRT2*DATS(ACT+K1,I)*DATS(ACT+K2,I)
       ENDDO
       !        zero fill
       DO I=NDATP1,NTOT2
          FA(I)=0.0D0
          FB(I)=0.0D0
       ENDDO
       !
       !        fft transform
       CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
       !
       !        fill result arrays
       DO I=1,NTOT2
          FC(I)=FC(I)+(FA(I)*FA(I))+(FB(I)*FB(I))
       ENDDO
    ENDDO
    !
    !     call inverse fft
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    !     and now fill the preliminary result into C1
    DO I=1,NCORR
       C3(I)=((ONEPT5*FC(I)/NTOT2)/(NDATP1-I))-HALF
    ENDDO
    !
    !
    !     IV: refill the results from C1, C2 and C3 into DATS
    !
    !     to allow for easier collection of the data we refill
    !     the results into the DATS structure but 3-row wide beginning
    !     with FIL
    !
    DO I=1,NCORR
       DATS(FIL  ,I)=C1(I)-(HALF*C2(I))
       DATS(FIL+1,I)=C2(I)
       DATS(FIL+2,I)=C3(I)
    ENDDO
    !
    !
    RETURN
  END SUBROUTINE DDPFF1
  !
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE WIN3D1(ACT,NDAT,NPRM,PRMLST,SERVAL, &
       COLS,ROWS,DATS, &
       COLS2,ROWS2,CORRS, &
       COLS3,ROWS3,SELLST)
    !
    !
    !     this routine does the windowing autocorrelation function for
    !     a 3dimensional timeseries in DATS(COLS,ROWS) into CORRS(COLS2,ROWS2).
    !     SELLST holds the pointers to the data/correl pointers
    !     up to now
    !
    !     Tibor Rudas Jun 2004 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,NDAT,NPRM,PRMLST(*)
    real(chm_real) SERVAL
    INTEGER COLS,ROWS
    INTEGER COLS2,ROWS2
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) CORRS(ROWS2,COLS2)
    INTEGER COLS3,ROWS3,SELLST(ROWS3,COLS3)
    !      INTEGER COLS3,ROWS3,SELLST(COLS3,ROWS3)
    !
    !
    INTEGER I,AC2,MAX
    INTEGER II,N1,KD,KC
    LOGICAL QP2
    real(chm_real) XTMP
    !
    MAX=COLS2
    IF(NDAT < COLS2) MAX=NDAT
    !
    !     this IF could be placed in the CORRS(KC...= line only
    !     but then it would be evaluated MAX x NPRM times...
    IF(SERVAL > ONE) THEN
       !        process p2
       DO I=1,MAX
          AC2=ACT-I+1
          IF(AC2 < 1) AC2=AC2+NDAT
          DO II=1,NPRM
             N1=PRMLST(II)
             KD=SELLST(N1,1)
             KC=SELLST(N1,2)
             XTMP=( DATS(KD  ,ACT) * DATS(KD  ,AC2) )+ &
                  ( DATS(KD+1,ACT) * DATS(KD+1,AC2) )+ &
                  ( DATS(KD+2,ACT) * DATS(KD+2,AC2) )
             !tr               CORRS(KC,I)=CORRS(KC,I)+(ONEPT5*XTMP*XTMP-HALF)
             !              new: see comment in WIN3D2
             CORRS(KC,I)=CORRS(KC,I)+(XTMP*XTMP)
          ENDDO
       ENDDO
    ELSE
       !        process p1
       DO I=1,MAX
          AC2=ACT-I+1
          IF(AC2 < 1) AC2=AC2+NDAT
          DO II=1,NPRM
             N1=PRMLST(II)
             KD=SELLST(N1,1)
             KC=SELLST(N1,2)
             XTMP=( DATS(KD  ,ACT) * DATS(KD  ,AC2) )+ &
                  ( DATS(KD+1,ACT) * DATS(KD+1,AC2) )+ &
                  ( DATS(KD+2,ACT) * DATS(KD+2,AC2) )
             CORRS(KC,I)=CORRS(KC,I)+XTMP
          ENDDO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE WIN3D1
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE WIN3D2(ACT,NDAT,NPRM,PRMLST,SERVAL, &
       COLS,ROWS,DATS, &
       COLS2,ROWS2,CORRS, &
       COLS3,ROWS3,SELLST)
    !
    !
    !     this routine does the windowing autocorrelation function for
    !     a 3dimensional timeseries in DATS(COLS,ROWS) into CORRS(COLS2,ROWS2).
    !     SELLST holds the pointers to the data/correl pointers
    !     up to now
    !     NEW: CORRS is now 3dim to enable accounting for steps in/out of a
    !          shell
    !
    !     Tibor Rudas Jan 2005 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,NDAT,NPRM,PRMLST(*)
    real(chm_real) SERVAL
    INTEGER COLS,ROWS
    INTEGER COLS2,ROWS2
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) CORRS(ROWS2,COLS2)
    INTEGER COLS3,ROWS3,SELLST(ROWS3,COLS3)
    !      INTEGER COLS3,ROWS3,SELLST(COLS3,ROWS3)
    !
    !
    INTEGER I,AC2,MAX
    INTEGER II,N1,KD,KC
    LOGICAL QP2
    real(chm_real) XTMP
    !
    !     only do this if the first (fixed) point is not ZERO....
    if( &
       (DATS(KD  ,ACT) /= 0.0D0).OR. &
            (DATS(KD+1,ACT) /= 0.0D0).OR. &
            (DATS(KD+2,ACT) /= 0.0D0)) THEN
       MAX=COLS2
       IF(NDAT < COLS2) MAX=NDAT
       !
       !     this IF could be placed in the CORRS(KC...= line only
       !     but then it would be evaluated MAX x NPRM times...
       IF(SERVAL > ONE) THEN
          !           process p2
          DO I=1,MAX
             AC2=ACT-I+1
             IF(AC2 < 1) AC2=AC2+NDAT
             DO II=1,NPRM
                N1=PRMLST(II)
                KD=SELLST(N1,1)
                KC=SELLST(N1,2)
                !                 and now check the momentary secondary point
                if( &
                   (DATS(KD  ,AC2) /= 0.0D0).OR. &
                        (DATS(KD+1,AC2) /= 0.0D0).OR. &
                        (DATS(KD+2,AC2) /= 0.0D0)) THEN
                   XTMP=( DATS(KD  ,ACT) * DATS(KD  ,AC2) )+ &
                        ( DATS(KD+1,ACT) * DATS(KD+1,AC2) )+ &
                        ( DATS(KD+2,ACT) * DATS(KD+2,AC2) )
                   !tr  CORRS(KC,I)=CORRS(KC,I)+(ONEPT5*XTMP*XTMP-HALF)
                   !    new: sum cos(alpha)^2 and do the 3/2 ... part after
                   !    normalization since the normalizing factor is
                   !    different for each bin
                   CORRS(KC,I)=CORRS(KC,I)+(XTMP*XTMP)
                   CORRS(KC+1,I)=CORRS(KC+1,I)+1
                ENDIF
             ENDDO
          ENDDO
       ELSE
          !           process p1
          DO I=1,MAX
             AC2=ACT-I+1
             IF(AC2 < 1) AC2=AC2+NDAT
             DO II=1,NPRM
                N1=PRMLST(II)
                KD=SELLST(N1,1)
                KC=SELLST(N1,2)
                if( &
                   (DATS(KD  ,AC2) /= 0.0D0).OR. &
                        (DATS(KD+1,AC2) /= 0.0D0).OR. &
                        (DATS(KD+2,AC2) /= 0.0D0)) THEN
                   XTMP=( DATS(KD  ,ACT) * DATS(KD  ,AC2) )+ &
                        ( DATS(KD+1,ACT) * DATS(KD+1,AC2) )+ &
                        ( DATS(KD+2,ACT) * DATS(KD+2,AC2) )
                   CORRS(KC,I)=CORRS(KC,I)+XTMP
                   CORRS(KC+1,I)=CORRS(KC+1,I)+1
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE WIN3D2
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FFT3D1(NAT,NDAT,NPRM,PRMLST,SERVAL, &
       COLS,ROWS,DATS,NCORR,SELLST)
    !
    !     this routine does fft correlation of a VACF series
    !
    !     Tibor Rudas Jun 2004 - June 2006
    !
  use memory
    !
    real(chm_real),allocatable,dimension(:),target :: space_SF
    INTEGER NAT,NDAT,NPRM,PRMLST(*),COLS,ROWS
    real(chm_real) SERVAL
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,SELLST(NAT,2)
    !
    !
    INTEGER I,K,N1,NTOT2,LFA
    real(chm_real),dimension(:),pointer :: SFA,SFB,SFC,SFD
    INTEGER PT
    !
    !     find number of points we want to keep
    !     (highest power of 2 lower than half the data points)
    IF(NCORR <= 0) THEN
       NCORR=2
       DO WHILE(NCORR <= NDAT)
          NCORR=NCORR*2
       ENDDO
       NCORR=NCORR/4
    ENDIF
    !
    !     find the length of the fft arrays
    NTOT2=2
    DO WHILE(NTOT2 < (NCORR+NDAT))
       NTOT2=NTOT2*2
    ENDDO
    !
    !     allocate space
    LFA=NTOT2
    call chmalloc('ancsol.src','ddpFFT','space_sf',4*LFA,crl=space_sf)
    SFA => space_sf(0*lfa+1 : 1*lfa)
    SFB => space_sf(1*lfa+1 : 2*lfa)
    SFC => space_sf(2*lfa+1 : 3*lfa)
    SFD => space_sf(3*lfa+1 : 4*lfa)

    !     and now call the single series fft for each set
    PT=1
    DO I=1,NPRM
       N1=PRMLST(I)
       K=SELLST(N1,1)
       IF(SERVAL > 1.0D0) THEN
          !           P2
          CALL FFT3D12(K,PT,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
               SFA,SFB,SFC,SFD)
       ELSE
          !           P1
          CALL FFT3D11(K,PT,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
               SFA,SFB,SFC,SFD)
       ENDIF
       PT=PT+1
    ENDDO
    call chmdealloc('ancsol.src','ddpFFT','space_sf',4*LFA,crl=space_sf)
    !
    RETURN
  END SUBROUTINE FFT3D1
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FFT3D11(ACT,FIL,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
       FA,FB,FC,FD)
    !
    !     this routine does fft correlation of one VAC set (in-place, filling
    !     FIL)
    !
    !     Tibor Rudas Jun 2004 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,FIL,NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NTOT2
    real(chm_real) FA(*),FB(*),FC(*),FD(*)
    !
    !
    INTEGER NDATP1,I,MVC,DMVC
    !
    NDATP1=NDAT+1
    !
    DO I=1,NTOT2
       FC(I)=ZERO
       FD(I)=ZERO
    ENDDO
    !
    !     see correl/corfun.src for details
    !
    DO MVC=1,3,2
       !
       DMVC=MVC-1
       DO I=1,NDAT
          FA(I)=DATS(ACT+DMVC,I)
       ENDDO
       IF(MVC < 3) THEN
          DO I=1,NDAT
             FB(I)=DATS(ACT+1,I)
          ENDDO
       ELSE
          DO I=1,NDAT
             FB(I)=ZERO
          ENDDO
       ENDIF
       !
       DO I=NDATP1,NTOT2
          FA(I)=ZERO
          FB(I)=ZERO
       ENDDO
       !
       CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
       !     
       DO I=1,NTOT2
          FC(I)=FC(I)+FA(I)*FA(I)+FB(I)*FB(I)
       ENDDO
       !
    ENDDO
    !
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    DO I=1,NDAT
       DATS(FIL,I)=((FC(I)/NTOT2)/(NDATP1-I))
    ENDDO
    !
    RETURN
  END SUBROUTINE FFT3D11
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FFT3D12(ACT,FIL,NDAT,COLS,ROWS,DATS,NCORR,NTOT2, &
       FA,FB,FC,FD)
    !
    !     this routine does fft correlation of one VAC set (in-place, filling
    !     FIL)
    !     do P2
    !
    !     Tibor Rudas Oct 2004 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER ACT,FIL,NDAT,COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NTOT2
    real(chm_real) FA(*),FB(*),FC(*),FD(*)
    !
    !
    INTEGER NDATP1,I,MVC,DMVC,DMVC2
    real(chm_real) SQRT2
    !
    SQRT2 = SQRT(TWO)
    NDATP1=NDAT+1
    !
    DO I=1,NTOT2
       FC(I)=ZERO
       FD(I)=ZERO
    ENDDO
    !
    !     see correl/corfun.src for details
    !
    DO MVC=1,3
       !
       !        FOR P2
       !        MVC=1, FA=X*X, FB=sqrt(2)*X*Y
       !        MVC=2, FA=Y*Y, FB=sqrt(2)*Y*Z;
       !        MVC=3, FA=Z*Z, FB=sqrt(2)*Z*X
       DMVC=MVC-1
       DMVC2=MVC
       IF(MVC == 3) DMVC2=0
       DO I=1,NDAT
          FA(I)=DATS(ACT+DMVC,I)*DATS(ACT+DMVC,I)
          FB(I)=SQRT2*DATS(ACT+DMVC,I)*DATS(ACT+DMVC2,I)
       ENDDO
       !
       DO I=NDATP1,NTOT2
          FA(I)=ZERO
          FB(I)=ZERO
       ENDDO
       !
       CALL correl_FFT(FA,FB,NTOT2,NTOT2,NTOT2,1)
       !     
       DO I=1,NTOT2
          FC(I)=FC(I)+FA(I)*FA(I)+FB(I)*FB(I)
       ENDDO
       !
    ENDDO
    !
    CALL correl_FFT(FC,FD,NTOT2,NTOT2,NTOT2,-1)
    !
    DO I=1,NDAT
       DATS(FIL,I)=((ONEPT5*FC(I)/NTOT2)/(NDATP1-I))-HALF
    ENDDO
    !
    RETURN
  END SUBROUTINE FFT3D12
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SUMSAT(MXNTOT,RESLT,COLS,ROWS,DATS,NCORR)
    !
    !     Sums up the single results which are stored in DATS into RESLT
    !     for SATM (1 set per atom)
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER MXNTOT
    real(chm_real) RESLT(MXNTOT,*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR
    !
    INTEGER I,J,K
    !
    !     I goes up to NCORR data points which we want to keep
    DO I=1,NCORR
       RESLT(I,1)=0.0D0
       !        and J loops over all rows (atoms) collected
       DO J=1,ROWS
          RESLT(I,1)=RESLT(I,1)+DATS(J,I)
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE SUMSAT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SUMSAX(MXNTOT,RESLT,COLS,ROWS,DATS,NCORR, &
       NSEC,START,SKIP,FACTOR,SUMIT,FACTR2)
    !
    !     Sums up the single results which are stored in DATS into RESLT
    !     for SATX (1 set per atom)
    !
    !     Tibor Rudas 2. 06. 2004 - June 2006
    !
  use exfunc
    !
    INTEGER MXNTOT
    real(chm_real) RESLT(MXNTOT,*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NSEC,START,SKIP
    real(chm_real) FACTOR,FACTR2
    LOGICAL SUMIT
    !
    INTEGER I,J,K,PT
    !
    !     I goes up to NCORR data points which we want to keep
    DO I=1,NCORR
       RESLT(I,1)=0.0D0
       !        and J loops over all rows (atoms) collected
       PT=START
       DO J=1,NSEC
          RESLT(I,1)=RESLT(I,1)+DATS(PT,I)
          PT=PT+SKIP
       ENDDO
    ENDDO
    !
    !     normalization
    DO I=1,NCORR
       RESLT(I,1)=RESLT(I,1)*FACTOR
    ENDDO
    !
    !     simple integration by sum (if requested)
    !     in the next series
    IF(SUMIT) THEN
       RESLT(1,2)=RESLT(1,1)*0.5D0*FACTR2
       DO I=2,(NCORR-1)
          RESLT(I,2)=RESLT(I-1,2)+RESLT(I,1)*FACTR2
       ENDDO
       RESLT(NCORR,2)=RESLT(NCORR-1,2)+RESLT(NCORR,1)*FACTR2*0.5D0
    ENDIF
    !
    RETURN
  END SUBROUTINE SUMSAX
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SUMMRT(MXNTOT,RESLT,DATS,NCORR)
    !
    !     Sums up the single results which are stored in DATS into RESLT
    !     for MRTI (1 histogramm)
    !
    !     Tibor Rudas 16. 03. 2004 - June 2006
    !
  use exfunc
    !
    INTEGER MXNTOT
    real(chm_real) RESLT(MXNTOT,*)
    INTEGER DATS(*)
    INTEGER NCORR
    !
    INTEGER I
    !
    !     I goes up to NCORR data points which we want to keep
    !     bear in mind that we never fill RESLT(1) which would represent
    !     a "residence time" of 0 steps
    DO I=2,NCORR
       RESLT(I,1)=DBLE(DATS(I-1))
    ENDDO
    !
    RETURN
  END SUBROUTINE SUMMRT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SUMDDP(MXNTOT,RESLT, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST, &
       NSEC,SECLST, &
       COLS2,ROWS2,SELLST, &
       NCORR)
    !
    !     Sums up the single results which are stored in DATS into RESLT
    !     for DDIP (3 sets per atom)
    !
    !     Tibor Rudas 21. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER MXNTOT
    real(chm_real) RESLT(MXNTOT,*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER COLS2,ROWS2,SELLST(ROWS2,COLS2)
    INTEGER NCORR
    !
    INTEGER I,J,L,K,KK,N1,N2,PT
    !
    !
    PT=1
    !     I/N1 loops over all primary atoms
    DO I=1,NPRM
       N1=PRMLST(I)
       !        K is the offset for this primary atom (DAT/COR should be the same
       !          after the FFT: both 3 cols per atom-pair)
       K=SELLST(N1,3)
       !        J loops over all data points
       DO J=1,NCORR
          RESLT(J,PT  )=0.0D0
          RESLT(J,PT+1)=0.0D0
          RESLT(J,PT+2)=0.0D0
          !           and L/N2 finally loops over all secondary atoms
          DO L=1,NSEC
             N2=SECLST(L)
             KK=K+SELLST(N2,4)
             RESLT(J,PT  )=RESLT(J,PT  )+DATS(KK  ,J)
             RESLT(J,PT+1)=RESLT(J,PT+1)+DATS(KK+1,J)
             RESLT(J,PT+2)=RESLT(J,PT+2)+DATS(KK+2,J)
          ENDDO
       ENDDO
       PT=PT+3
    ENDDO
    !
    RETURN
  END SUBROUTINE SUMDDP
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SUMC1D(MXNTOT,RESLT,SERVAL,COLS,ROWS,DATS,NCORR, &
       NRAT,START,SKIP,FACTOR,SUMIT,FACTR2)
    !
    !     Sums up the results which are stored in DATS into RESLT
    !     for a 1D correlation function. NRAT gives the nr of atoms (i.e.
    !     the number of correlations to sum up), START the index of the
    !     first correlation function and SKIP the spacing between the single
    !     functions. FACTOR is a normalization factor. If SUMIT is .true.
    !     corr(x)*FACTR2 will be summed into the _next_ RESLT slot giving
    !     a crude integral of the function.
    !
    !     Tibor Rudas Sep. 2004 - June 2006
    !
  use exfunc
  use number
    !
    INTEGER MXNTOT
    real(chm_real) RESLT(MXNTOT,*),SERVAL
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    INTEGER NCORR,NRAT,START,SKIP
    real(chm_real) FACTOR,FACTR2
    LOGICAL SUMIT
    !
    !
    INTEGER I,J,K,PT
    !
    DO I=1,NCORR
       RESLT(I,1)=0.0D0
       !        and J loops over all rows (atoms) collected, PT gives the slot
       !        for this atom
       PT=START
       DO J=1,NRAT
          RESLT(I,1)=RESLT(I,1)+DATS(PT,I)
          PT=PT+SKIP
       ENDDO
       !        normalization
       RESLT(I,1)=RESLT(I,1)*FACTOR
    ENDDO
    !
    !
    !     simple integration by sum (if requested)
    !     in the next series
    IF(SUMIT) THEN
       RESLT(1,2)=RESLT(1,1)*0.5D0*FACTR2
       DO I=2,(NCORR-1)
          RESLT(I,2)=RESLT(I-1,2)+(RESLT(I,1)*FACTR2)
       ENDDO
       RESLT(NCORR,2)=RESLT(NCORR-1,2)+(RESLT(NCORR,1)*FACTR2*0.5D0)
    ENDIF
    !
    RETURN
  END SUBROUTINE SUMC1D
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE NRMWIN(NTOT,COLS,ROWS,CORS,NCORR)
    !
    !     Normalizes all correlation functions in CORS up to NCORR
    !     data points when NTOT points have been sampled in total.
    !     (CAUTION: 'normalization' only with respect to the different
    !     number of samples in each bin NOT with respect to C(0)=1.0 !)
    !
    !     Tibor Rudas 14. 02. 2003 - June 2006
    !
  use exfunc
    !
    INTEGER NTOT,COLS,ROWS
    real(chm_real) CORS(ROWS,COLS)
    INTEGER NCORR
    !
    INTEGER I,J,NRM
    !
    DO I=1,COLS
       NRM=NTOT-I+1
       DO J=1,ROWS
          CORS(J,I)=CORS(J,I)/NRM
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE NRMWIN
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE MAXDDP(MXNTOT,UNOUT,NROUT, &
       IDX,VALS,IATM,TMPVAL, &
       COLS,ROWS,DATS, &
       NPRM,PRMLST,NSEC,SECLST, &
       COLS2,ROWS2,SELLST, &
       NCORR,DELTA)
    !
    !     does output of the NROUT maxmium contributions to in DATS
    !     assuming NPRM primary and NSEC secondary atoms (in xxxLST)
    !     SELLST is the same as in FILDDP and SUMDDP etc.
    !     output is done in UNOUT
    !
    !     Tibor Rudas MAY 2004 - June 2006
    !
  use exfunc
  use chutil,only:atomid
    !
    INTEGER MXNTOT,UNOUT,NROUT,IDX(*),IATM(*)
    real(chm_real) VALS(*),TMPVAL(*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) DELTA
    INTEGER NPRM,PRMLST(*),NSEC,SECLST(*)
    INTEGER COLS2,ROWS2,SELLST(ROWS2,COLS2)
    INTEGER NCORR
    !
    INTEGER I,J,L,K,KK,N1,N2,NOUT,PT
    character(len=4) SEGID,RESID,RESNAM,ATNAM
    !
    !
    NOUT=NROUT
    IF(NROUT > NSEC) NOUT=NSEC
    !     I/N1 loops over all primary atoms
    DO I=1,NPRM
       N1=PRMLST(I)
       !        K is the offset for this primary atom (DAT/COR should be the same
       !          after the FFT: both 3 cols per atom-pair)
       K=SELLST(N1,3)
       !
       !        now do all for this one primary atom:
       !        - fill the starting value into VALS
       !        - get the sorted indexes into IDX (biggest first ???)
       !        - output the largest NROUT
       !
       !        J/N2 loops over all secondary atoms
       DO J=1,NSEC
          N2=SECLST(J)
          KK=K+SELLST(N2,4)
          VALS(J)=DATS(KK,1)
          IATM(J)=N2
       ENDDO
       CALL INDIXF(NSEC,VALS,IDX)
       !
       !        do output:
       CALL ATOMID(N1,SEGID,RESID,RESNAM,ATNAM)
       WRITE(UNOUT,850) NOUT,SEGID,RESID,RESNAM,ATNAM
850    FORMAT('# ', I5, ' maximum values for ', 4(1X,A4))
       !
       DO L=1,NOUT
          N2=IATM(IDX(L))
          CALL ATOMID(N2,SEGID,RESID,RESNAM,ATNAM)
          WRITE(UNOUT,851) L,SEGID,RESID,RESNAM,ATNAM
       ENDDO
851    FORMAT('# ', I5, ' : ', 4(1X,A4))
       !
       !        I rearrange the data into a temporary array (i know this could be
       !        done in an implicit loop with WRITE but the index-lookup-lookup
       !        might become a littel unreadable (and this routine should _not_ be
       !        the bottleneck...)
       !        try to use 'DUMB TIME' like output (i.e. 1st col = time)
       DO J=1,NCORR
          PT=1
          DO L=1,NOUT
             N2=IATM(IDX(L))
             KK=K+SELLST(N2,4)
             TMPVAL(PT  )=DATS(KK  ,J)
             TMPVAL(PT+1)=DATS(KK+1,J)
             TMPVAL(PT+2)=DATS(KK+2,J)
             PT=PT+3
          ENDDO
          WRITE(UNOUT,852) (DELTA*(J-1)),(TMPVAL(L),L=1,(3*NOUT))
       ENDDO
       ! 852     FORMAT(300(1X,G22.15))
852    FORMAT(300(1X,G12.5))
    ENDDO
    !
    RETURN
  END SUBROUTINE MAXDDP
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE MAXSAT(UNOUT,NROUT, &
       IDX,VALS,IATM,TMPVAL, &
       COLS,ROWS,DATS, &
       NAT,ATLST,SELLST, &
       NCORR,DELTA)
    !
    !     does output of the NROUT maxmium contributions to in DATS
    !     assuming NPRM primary and NSEC secondary atoms (in xxxLST)
    !     SELLST is the same as in FILDDP and SUMDDP etc.
    !     output is done in UNOUT
    !
    !     Tibor Rudas MAY 2004 - June 2006
    !
  use exfunc
  use chutil,only:atomid
    !
    INTEGER UNOUT,NROUT,IDX(*),IATM(*)
    real(chm_real) VALS(*),TMPVAL(*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) DELTA
    INTEGER NAT,ATLST(*),SELLST(*)
    INTEGER NCORR
    !
    INTEGER I,J,N1,NOUT
    character(len=4) SEGID,RESID,RESNAM,ATNAM
    !
    !
    NOUT=NROUT
    IF(NROUT > NAT) NOUT=NAT
    !     I/N1 loops over all primary atoms
    DO I=1,NAT
       N1=ATLST(I)
       VALS(I)=DATS(SELLST(N1),1)
       IATM(I)=N1
    ENDDO
    CALL INDIXF(NAT,VALS,IDX)
    !
    !     do output:
    WRITE(UNOUT,850) NOUT
850 FORMAT('# ', I5, ' maximum values for SAT timeseries')
    !
    DO I=1,NOUT
       N1=IATM(IDX(I))
       CALL ATOMID(N1,SEGID,RESID,RESNAM,ATNAM)
       WRITE(UNOUT,851) I,SEGID,RESID,RESNAM,ATNAM
    ENDDO
851 FORMAT('# ', I5, ' : ', 4(1X,A4))
    !
    !     I rearrange the data into a temporary array (...)
    !     try to use 'DUMB TIME' like output (i.e. 1st col = time)
    DO I=1,NCORR
       DO J=1,NOUT
          N1=IATM(IDX(J))
          TMPVAL(J)=DATS(SELLST(N1),I)
       ENDDO
       WRITE(UNOUT,852) (DELTA*(I-1)),(TMPVAL(J),J=1,NOUT)
    ENDDO
852 FORMAT(300(1X,G12.5))
    !
    RETURN
  END SUBROUTINE MAXSAT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE MAXSAX(UNOUT,NROUT, &
       IDX,VALS,IATM,TMPVAL, &
       COLS,ROWS,DATS, &
       NAT,ATLST,SELLST, &
       NCORR,DELTA,NATSEL)
    !
    !     does output of the NROUT maxmium contributions to in DATS
    !     assuming NPRM primary and NSEC secondary atoms (in xxxLST)
    !     SELLST is the same as in FILDDP and SUMDDP etc.
    !     output is done in UNOUT
    !
    !     Tibor Rudas JUN 2004 - June 2006
    !
  use exfunc
  use chutil,only:atomid
    !
    INTEGER UNOUT,NROUT,IDX(*),IATM(*)
    real(chm_real) VALS(*),TMPVAL(*)
    INTEGER COLS,ROWS
    real(chm_real) DATS(ROWS,COLS)
    real(chm_real) DELTA
    INTEGER NAT,ATLST(*),NATSEL,SELLST(NATSEL,2)
    INTEGER NCORR
    !
    INTEGER I,J,N1,NOUT
    character(len=4) SEGID,RESID,RESNAM,ATNAM
    !
    !
    NOUT=NROUT
    IF(NROUT > NAT) NOUT=NAT
    !     I/N1 loops over all primary atoms
    DO I=1,NAT
       N1=ATLST(I)
       VALS(I)=DATS(SELLST(N1,2),1)
       IATM(I)=N1
    ENDDO
    CALL INDIXF(NAT,VALS,IDX)
    !
    !     do output:
    WRITE(UNOUT,850) NOUT
850 FORMAT('# ', I5, ' maximum values for SAT timeseries')
    !
    DO I=1,NOUT
       N1=IATM(IDX(I))
       CALL ATOMID(N1,SEGID,RESID,RESNAM,ATNAM)
       WRITE(UNOUT,851) I,SEGID,RESID,RESNAM,ATNAM
    ENDDO
851 FORMAT('# ', I5, ' : ', 4(1X,A4))
    !
    !     I rearrange the data into a temporary array (...)
    !     try to use 'DUMB TIME' like output (i.e. 1st col = time)
    DO I=1,NCORR
       DO J=1,NOUT
          N1=IATM(IDX(J))
          TMPVAL(J)=DATS(SELLST(N1,2),I)
       ENDDO
       WRITE(UNOUT,852) (DELTA*(I-1)),(TMPVAL(J),J=1,NOUT)
    ENDDO
852 FORMAT(300(1X,G12.5))
    !
    RETURN
  END SUBROUTINE MAXSAX
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INDIXF(N,ARRIN,INDX)
    !
    ! From "Numerical Recipes", based on QuickSort.
    ! Indexes an array ARRIN(1:N), i.e., outputs the array INDX(1:N) such that
    ! ARRIN(INDX(j)) is in descending order for j=1,2,...,N.
    ! The input quantities N and ARRIN are not changed.
    !
    !     copied from util/sort.src tr 10 may 2004 and modified for real(chm_real)
    !     should be replaced by some free implementation
    !
    ! global
    ! passed
    real(chm_real) ARRIN(*)
    INTEGER INDX(*),N
    ! local
    real(chm_real) Q
    INTEGER I,J,L,IR,INDXT
    !
    ! begin
    !
    DO J=1,N
       INDX(J)=J
       !
    enddo
    L=N/2+1
    IR=N
    !
    do while(.true.)
       IF(L > 1)THEN
          L=L-1
          INDXT=INDX(L)
          Q=ARRIN(INDXT)
       ELSE
          INDXT=INDX(IR)
          Q=ARRIN(INDXT)
          INDX(IR)=INDX(1)
          IR=IR-1
          IF(IR == 1)THEN
             INDX(1)=INDXT
             RETURN
          ENDIF
       ENDIF
       I=L
       J=L+L
       do while(J <= IR)
          IF(J < IR)THEN
             !trd            IF(ARRIN(INDX(J)) < ARRIN(INDX(J+1))) J=J+1
             IF(ARRIN(INDX(J)) > ARRIN(INDX(J+1))) J=J+1
          ENDIF
          !trd         IF(Q < ARRIN(INDX(J)))THEN
          IF(Q > ARRIN(INDX(J)))THEN
             INDX(I)=INDX(J)
             I=J
             J=J+J
          ELSE
             J=IR+1
          ENDIF
       ENddo
       INDX(I)=INDXT
    enddo
    !
    !----------------------------------------------------------------------
    !
    return
  end SUBROUTINE INDIXF
#endif /* (corsol)*/

  SUBROUTINE NUL_ANCSOL
    RETURN
  END SUBROUTINE NUL_ANCSOL
end module ancsol_mod

