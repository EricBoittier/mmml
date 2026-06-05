!module nbutil_module
!
!contains

SUBROUTINE RENBND(BNBND,MAXJNB,NATOM,NGRP,MXJNBG)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE RESIZES THE NONBONDED DATA STRUCTURE TO
  !     ACCOMMODATE A BIGGER JNB ARRAY
  !
  !     THE EXTENEDED ELECTROSTATIC FIELD AND GRADIENT ARRAYS ARE ONLY
  !     ALLOCATED FOR THOSE NONBONDED OPTIONS WHERE THEY WILL BE NEEDED
  !
  !     Author: David States
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use inbnd
  use stream
  use datstr
  implicit none

  type(nonbondDataStructure) BNBND

  INTEGER MAXJNB,NATOM,NGRP,MXJNBG
  !
  INTEGER CHNGIN(20),CHNGLN(20)
  INTEGER NCHNG,J
  CHARACTER(len=4) WINIT
  !
  IF(.NOT.USEDDT_nbond(BNBND)) THEN
     WINIT='INIT'
     J=4
     CALL GTNBCT(WINIT,J,BNBND)
  ENDIF

  call NBGROW(bnbnd%JNB, MAXJNB)
#if KEY_IMCUBES==1
  J=NATOM*2
#else /**/
  J=NATOM
#endif 
  call NBGROW(bnbnd%INBLO, J)
  call NBGROW(bnbnd%INBLOG, NGRP)
  call NBGROW(bnbnd%JNBG, MXJNBG)

  RETURN
END SUBROUTINE RENBND

SUBROUTINE PRNBCT(BNBND)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PRINTS OUT NONBONDED CUTOFFS
  !
  !      By Bernard R. Brooks  (and others)    1983
  !
#if KEY_MMFF==1
  !      Jay Banks 23 Oct 95:
  !      Added code for processing more complicated cutoff options, from
  !      Tom Halgren's MSI/Merck version.  Increased array sizes to
  !      include new options (truncation, TAH's "modified shift"), and
  !      added them to name arrays.
#endif 
#if KEY_ACE==1
  !      not all printed non-bonded options (e.g. CDIE) are valid if ACE
  !      is used
#endif 

  use ewald,only: kappa,ksqmax,kmax,lewald
  use pme_module,only:qpme,reqcor,rewcut, &
       qfinit
  use pmeutil,only: nfft1,nfft2,nfft3,forder
#if KEY_LJPME==1
  use pmeutil,only: dfftx,dffty,dfftz,dorder, dkappa,&
                    dmax, dmaxx, dmaxy, dmaxz,&
                    qmax, qmaxx, qmaxy, qmaxz,&
                    qcpme, qljpme
#endif
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  !
  use ace_module
  use inbnd
  use exelecm
  use stream
  use grape
  use fmam
  use memory
  implicit none

  type(nonbondDataStructure) BNBND

  INTEGER NU,I,J
#if KEY_MMFF==1
  LOGICAL RSHIFT,RFSWIT,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT, &
       RMSHI,CMSHI
  INTEGER, PARAMETER :: NNAME=16
  INTEGER, PARAMETER :: LNAME=16
#else /**/
  INTEGER, PARAMETER :: NNAME=10
  INTEGER, PARAMETER :: LNAME=12
#endif 
  INTEGER, PARAMETER :: MNAME=3
  INTEGER, PARAMETER :: KNAME=1
  CHARACTER(len=8) :: UNAME(LNAME),XNAME,BLANK='        ', &
       ENAME(MNAME)=(/'EXTEnd  ','GRADient','QUADripo'/),&
       FNAME(MNAME)=(/'NOEXtnd ','        ','        '/),&
       GNAME(KNAME)=  'EWALd   ',HNAME(KNAME)='NOEWald ', &
       INAME(KNAME)=  'FMA     ', &
       CNAME(MNAME)=(/'GSHIft  ','GVSHift ','ERROR   '/)
  character(len=4) :: BNAME='BYCC'
#if KEY_ACE==1
  CHARACTER(len=7) :: ON='     ON',OFF='    OFF',ONOROF
#endif 
#if KEY_MMFF==1
  CHARACTER(len=8) :: YNAME(NNAME)=(/'ELEC    ','VDW     ','GROUps  ','CDIElec ', &
       'SHIFt   ','VATOm   ','VSHIft  ','BYCUbe  ', &
       'FSHIft  ','VFSWIt  ','EWALd   ','FMA     ', &
       'TRUNcate','VTRUncat','MSHIFT  ','DBEXP   '/)
  CHARACTER(len=8) :: ZNAME(NNAME)=(/'NOELec  ','NOVDw   ','ATOMs   ','RDIElec ', &
       'SWITch  ','VGROup  ','VSWItch ','BYGRoup ', &
       'FSWItch ','ERROR   ','ERROR   ','ERROR   ', &
       'ERROR   ','ERROR   ','ERROR   ','ERROR   '/)
#else /**/
  CHARACTER(len=8) :: YNAME(NNAME)=(/'ELEC    ','VDW     ','GROUps  ','CDIElec ', &
       'SHIFt   ','VATOm   ','VSHIft  ','BYCUbe  ', &
       'FSHIft  ','VFSWIt  '/)
  CHARACTER(len=8) :: ZNAME(NNAME)=(/'NOELec  ','NOVDw   ','ATOMs   ','RDIElec ', &
       'SWITch  ','VGROup  ','VSWItch ','BYGRoup ', &
       'FSWItch ','ERROR   '/)
#endif 
  !
  IF(PRNLEV < 2) RETURN
#if KEY_MMFF==1
  ! Set flags for electrostatics options (8 possibilities)
  RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. .NOT.LEIPS
  RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. .NOT.LEIPS
  RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. .NOT.LEIPS
  RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
       .AND. .NOT.LMSHFT .AND. .NOT.LEIPS
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. .NOT.LEIPS
  CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. .NOT.LEIPS
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. .NOT.LEIPS
  CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
       .AND. .NOT.LMSHFT .AND. .NOT.LEIPS
  CMSHI =      LCONS .AND.      LMSHFT .AND. .NOT.LEIPS
  RMSHI = .NOT.LCONS .AND.      LMSHFT .AND. .NOT.LEIPS
#endif 
  ! FILL FLAG OUTPUTS
  NU=0
  ! process name translations for force based nonbond options.
#if KEY_MMFF==1
  DO J=1,8
     I=J
     IF(J == 5) THEN      !Electrostatic Shift (or switch) option
        NU=NU+1
        IF(CSHFT.OR.RSHFT) THEN         !"Energy-based" Shift
           UNAME(NU) = YNAME(5)
        ELSE IF(CSHIFT.OR.RSHIFT) THEN  !"Force-based" Shift
           UNAME(NU) = YNAME(9)
        ELSE IF(CSWIT.OR.RSWIT) THEN    !"Energy-based" Switch
           UNAME(NU) = ZNAME(5)
        ELSE IF(CFSWIT.OR.RFSWIT) THEN  !"Force-based" Switch
           UNAME(NU) = ZNAME(9)
         ELSE IF(RMSHI.OR.CMSHI) THEN    !"Modified" shift
            UNAME(NU) = YNAME(15)
         ELSE IF(LEIPS) THEN    !  Ele IPS
            UNAME(NU) = 'EIPS'
         ELSE IF(BNBND%LNBOPT(1)) THEN
           CALL WRNDIE(-5,'<PRNBCT>', &
                'No electrostatic cutoff method defined')
        ENDIF
     ELSE
        !
        IF(J == 7 .AND. BNBND%LNBOPT(10)) I=10  !VDW Force Switch
        IF(J == 7 .AND. BNBND%LNBOPT(14)) I=14 !VDW Truncation

        IF(BNBND%LNBOPT(I)) THEN   ! TAH/MERCK
           XNAME=YNAME(I)
        ELSE
           XNAME=ZNAME(J)
        ENDIF
        IF(J == 7 .AND. LVIPS) XNAME='VIPS'       !VDW IPS

        IF((J == 8).AND.(LBYCC)) XNAME = BNAME
#if KEY_IMCUBES==1
        IF((J == 8).AND.(LBYCBIM)) XNAME = 'BYCB'      
#endif
        IF(XNAME /= BLANK) THEN
           NU=NU+1
           IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
           UNAME(NU)=XNAME
        ENDIF
     ENDIF
  ENDDO
  ! FILL TRUNCATION FLAGS
  IF(BNBND%LNBOPT(13)) THEN
     XNAME=YNAME(13)
     IF(XNAME /= BLANK) THEN
        NU=NU+1
        IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
        UNAME(NU)=XNAME
     ENDIF
  ENDIF
#else /**/
  DO J=1,8
     I=J
     IF(J == 5 .AND. BNBND%LNBOPT(9)) I=9
     IF(J == 7 .AND. BNBND%LNBOPT(10)) I=10
     IF(BNBND%LNBOPT(J)) THEN
        XNAME=YNAME(I)
     ELSE
        !GROMACS switches are handled below
        !This used to just say 
        !XNAME=ZNAME(I)
        IF ((J == 5 .AND. BNBND%LNBOPT(23)) .OR. &
             (J == 7 .AND. BNBND%LNBOPT(24))) THEN
           XNAME=BLANK
        ELSE
           XNAME=ZNAME(I)
        ENDIF
     ENDIF
     IF((J == 8).AND.(LBYCC)) XNAME = BNAME
#if KEY_IMCUBES==1
     IF((J == 8).AND.(LBYCBIM)) XNAME = 'BYCB'      
#endif
     IF(XNAME /= BLANK) THEN
        NU=NU+1
        IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
        UNAME(NU)=XNAME
     ENDIF
  ENDDO
! Look for GROMACS electrostatics and VDW shifts. These are stored in
! CNAME above, and as LNBOPT(24-25), hence the indexing from start+22+1
! to start+22+2. They should not require an increase in LNAME, as they
! are incompatible with other shift/switch options.
  DO I=1,2
     IF (BNBND%LNBOPT(22+I)) THEN
        XNAME=CNAME(I)
        IF(XNAME.NE.BLANK) THEN
           NU=NU+1
           IF(NU.GT.LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
           UNAME(NU)=XNAME
        ENDIF
     ENDIF
  ENDDO

#endif 
  ! FILL EXTENDED ELECTROSTATICS FLAGS
  XNAME=FNAME(1)
  IF(QEXTND) XNAME=ENAME(1)
  IF(XNAME /= BLANK) THEN
     NU=NU+1
     IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UNAME(NU)=XNAME
  ENDIF
  XNAME=FNAME(2)
  IF(QXGRAD) XNAME=ENAME(2)
  IF(XNAME /= BLANK) THEN
     NU=NU+1
     IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UNAME(NU)=XNAME
  ENDIF
  XNAME=FNAME(3)
  IF(QXQUAD) XNAME=ENAME(3)
  IF(XNAME /= BLANK) THEN
     NU=NU+1
     IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UNAME(NU)=XNAME
  ENDIF
  !
  XNAME=HNAME(1)
  IF(LEWALD) XNAME=GNAME(1)
  IF(XNAME /= BLANK) THEN
     NU=NU+1
     IF(NU > LNAME) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UNAME(NU)=XNAME
  ENDIF
  !
  ! FMA
  XName = Blank
  if (LFma) XName=INAme(1)
  if (XName  /=  Blank) then
     Nu = Nu + 1
     if (Nu  >  Lname) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UName(Nu)=XName
  endif
  !
  ! DBEXP
  XName = Blank
  if (LDBEXP) XName='DBEXP'
  if (XName  /=  Blank) then
     Nu = Nu + 1
     if (Nu  >  Lname) CALL WRNDIE(-5,'<PRNBCT>','NU.GT.LNAME')
     UName(Nu)=XName
  endif
  !
  IF(PRNLEV >= 2) WRITE(OUTU,2000)  (UNAME(I),I=1,NU)
2000 FORMAT (/' NONBOND OPTION FLAGS: '/(5X,7(A8,1X)))
  !
IF (QGRF) THEN   ! GRF -- Wei Chen 2015
  IF(PRNLEV >= 2) WRITE(OUTU,2311) CTOFNB,RFTEMP,EPS,RFEPSO,RFIONI,RFKAP
2311 FORMAT (' Generalized Reaction Field ',/, &
          ' CUTOFF         =',F7.3,' TEMP         =',F7.3,/, &
          ' EPSIN          =',F7.3,' EPSOUT       =',F7.3,/, &
          ' IONIC STRENGTH =',F7.3,' DEBYE LENGTH =',F7.3)
ENDIF
  !
#if KEY_ACE==1 /*ace1*/
  IF(LACE) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,2001) CUTNB,CTEXNB,CTONNB,CTOFNB, &
          WRNMIN,WRNMXD,E14FAC,NBXMOD, &
          EPSS,EPSI,RSOLV,ACEAL
2001 FORMAT (' CUTNB  =',F7.3,' CTEXNB =',F7.3, &
          ' CTONNB =',F7.3,' CTOFNB =',F7.3,/, &
          ' WMIN   =',F7.3,' WRNMXD =',F7.3, &
          ' E14FAC =',F7.3,' NBXMOD =',I7,  /, &
          ' EPSS   =',F7.3,' EPSI   =',F7.3, &
          ' RSOLV  =',F7.3,' ALPHA  =',F7.3)
     ONOROF=OFF
     IF(LIDEAL) ONOROF=ON
     IF((SCTON /= CTONNB).OR.(SCTOF.NE.CTOFNB)) THEN
        IF(SIGHYD < ZERO) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,2051)  &
                ONOROF,'atomtyp',SCTON,SCTOF
2051       FORMAT (' LIDEAL =',A7,  ' SIGMA  =',A7, &
                ' SCTON  =',F7.3,' SCTOF  =',F7.3)
        ELSE
           IF(PRNLEV >= 2) WRITE(OUTU,2052)  &
                ONOROF,1000.0*SIGHYD,SCTON,SCTOF
2052       FORMAT (' LIDEAL =',A7,  ' SIGMA  =',F7.3, &
                ' SCTON  =',F7.3,' SCTOF  =',F7.3)
        ENDIF
     ELSE
        IF(SIGHYD < ZERO) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,2053) ONOROF,'atomtyp'
2053       FORMAT (' LIDEAL =',A7,  ' SIGMA  =',A7)
        ELSE
           IF(PRNLEV >= 2) WRITE(OUTU,2054) ONOROF,1000.0*SIGHYD
2054       FORMAT (' LIDEAL =',A7,  ' SIGMA  =',F7.3)
        ENDIF
     ENDIF
     ONOROF=OFF
     IF(LQGAUS) ONOROF=ON
     IF((FSSCAL /= ONE).OR.(FISCAL.NE.ONE).OR.(LQGAUS)) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,2055) FVSCAL,FSSCAL,FISCAL,ONOROF
2055    FORMAT (' FVSCAL =',F7.3,' FSSCAL =',F7.3, &
             ' FISCAL =',F7.3,' LQGAUS =',A7)
     ELSE
        IF(FVSCAL /= ONE.and.prnlev >= 2) WRITE(OUTU,2056) FVSCAL
2056    FORMAT (' FVSCAL =',F7.3)
     ENDIF
     IF(LACE2.and.prnlev >= 2) WRITE(OUTU,2057)  &
          ON,MXBSOL,TBSOLV,TBSOLH
2057 FORMAT (' ACE2   =',A7,  ' MXBSOL =',F7.3, &
          ' TBSOLV =',F7.3,' TBSOLH =',F7.3)
     IF(LACE3.and.prnlev >= 2) WRITE(OUTU,2058) ON
2058 FORMAT (' ACE3   =',A7)
  ELSE
#endif /* (ace1)*/
     !
     IF(PRNLEV >= 2) WRITE(OUTU,2002)  &
          CUTNB,CTEXNB,CTONNB,CTOFNB,CGONNB,CGOFNB,WRNMIN,WRNMXD, &
          E14FAC,EPS,NBXMOD
2002 FORMAT (' CUTNB  =',F7.3,' CTEXNB =',F7.3, &
          ' CTONNB =',F7.3,' CTOFNB =',F7.3,/, &
          ' CGONNB =',F7.3,' CGOFNB =',F7.3,/, &
          ' WMIN   =',F7.3,' WRNMXD =',F7.3, &
          ' E14FAC =',F7.3,' EPS    =',F7.3,/, &
          ' NBXMOD =',I7)
     !
#if KEY_ACE==1
  ENDIF                     
#endif
  !
#if KEY_MMFF==1
  IF (LVTRUNC.and.prnlev >= 2) WRITE(OUTU,2005) CTVTRN
2005 FORMAT (' VDW TRUNCATION: CTVTRN =',F7.3)
#endif 
#if KEY_SOFTVDW==1
  IF (QGAMIN.and.prnlev >= 2)then
     WRITE(OUTU,2006) RGAMIN
2006 FORMAT (' VDW SOFT CORE: BEGINS AT: EMAX =',F10.2, ' kcal/mol')
     IF (qvdwexp) then
        IF(PRNLEV >= 2) WRITE(OUTU,'(a)')  &
             ' VDW SOFT CORE : EXPONENTIAL FORM '
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,'(a)')  &
             ' VDW SOFT CORE : LINEAR FORM '
     ENDIF
     IF (egamina < 0.and.prnlev >= 2) WRITE(OUTU,2007) EGAMINA
2007 FORMAT ( &
          ' ELECTROSTATIC ATRACTIVE SOFT CORE : BEGINS AT : EMIN =', &
          F10.2, ' kcal/mol')
     IF (EGAMINR > 0.and.prnlev >= 2) WRITE(OUTU,2008) EGAMINR
2008 FORMAT ( &
          ' ELECTROSTATIC  REPULSIVE SOFT CORE : BEGINS AT : EMAX =', &
          F10.2, ' kcal/mol')
     if (qelexp) then
        if(prnlev >= 2) write(outu,'(a)')  &
             ' ELECTROSTATIC SOFT CORE : EXPONENTIAL FORM '
     else
        if(prnlev >= 2) write(outu,'(a)')  &
             ' ELECTROSTATIC SOFT CORE : LINEAR FORM '
     endif
  endif
#endif 

  if (LEWALD &
#if KEY_GRAPE==1
       .OR.LGRAPE &     
#endif
     )then
     !bfix.. on charmm.org FORUM by Rick Venable
     IF (.NOT.QPME) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,2003) KAPPA,KMAX,KSQMAX
2003    FORMAT (' EWALD OPTIONS: KAPPA  =',F7.3, &
             ' KMAX   =',I7,  ' KSQMAX =',I7)
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,2010)  &
             KAPPA,REQCOR,FORDER,NFFT1,NFFT2,NFFT3
2010    FORMAT(' PME EWALD OPTIONS: KAPPA  =',F7.3, &
             '  QCOR =',F7.3,'  Bspline order =',I2,/, &
             ' FFTX=',I4,'  FFTY=',I4,'  FFTZ=',I4)
#if KEY_LJPME==1
        if(qcpme .and. (PRNLEV >= 2)) WRITE(OUTU,2012) qMAXX, qMAXY, qMAXZ
2012    FORMAT(' cPME KSUM DIMENSIONS (ELECTROSTATICS): =', I3,',', I3, ',', I3)
        IF(qljpme .and. (PRNLEV >= 2)) WRITE(OUTU,2013)  &
             DKAPPA,DORDER,DFFTX,DFFTY,DFFTZ
2013    FORMAT(' LJPME EWALD OPTIONS: DKAPPA  =',F7.3, &
             '  Bspline order =',I2,/, &
             ' DFTX=',I4,'  DFTY=',I4,'  DFTZ=',I4)
        if(qljpme .and. qcpme .and. (PRNLEV >= 2)) WRITE(OUTU,2015) dMAXX, dMAXY, dMAXZ
2015    FORMAT(' cPME KSUM DIMENSIONS (DISPERSION): =', I3,',', I3, ',', I3)
#endif
        IF (QFINIT.and.prnlev >= 2) WRITE(OUTU,2016) REWCUT
2016    FORMAT(' FINITE CUTOFF = ',F9.4)
     ENDIF
     !bfix..
#if KEY_FFTW==1
     if(prnlev >= 2)  &
          write(outu,"('                Using FFTW')")
#elif KEY_MKL==1
     if(prnlev >= 2)  &
          write(outu,"('                Using MKL')")
!     call wrndie(-5,'<nbutil>','Support for MKL not finished')
#else /**/
     if(prnlev >= 2)  &
          write(outu,"('                Using Pub FFT')")
#endif 
     if(prnlev >= 2)  &
          write(outu,"('                Real-to-Complex FFT')")
#if KEY_COLFFT==1
     if(qpme)then
        write(outu,'("                Using Column FFT ")')
     endif
#endif 

  endif

#if KEY_FMA==1
  if (LFMA.and.prnlev >= 2) WRITE(OUTU,2004) Level, Terms
2004 FORMAT (' FMA OPTIONS: LEVELS   =',I7,  ' TERMS =',I7)
#endif 
#if KEY_NOMISC==0 /*rxnfld_print*/
  IF(QRXNFL.and.prnlev >= 2) THEN
     WRITE(OUTU,2030) RXNORD,EPSEXT,RXNSHL
2030 FORMAT(' REACTIONS FIELDS WILL BE CALCULATED TO ORDER',I3/ &
          ' USING AN EXTERNAL DIELECTRIC OF',F7.2,' AND SHELL OF',F7.2)
     WRITE(OUTU,2011) RXNMOD
2011 FORMAT(' REACTION FIELD MODE: "',A,'"')
  ENDIF
#endif /* (rxnfld_print)*/
  !
  IF(PRNLEV >= 2) WRITE(OUTU,2021) NNNB,NNB14,NNNBG,NNG14
2021 FORMAT( &
       ' There are',I9,' atom  pairs and',I9,' atom  exclusions.',/, &
       ' There are',I9,' group pairs and',I9,' group exclusions.')
  RETURN
END SUBROUTINE PRNBCT

SUBROUTINE GTNBCT(COMLYN,COMLEN,BNBND)
  !-----------------------------------------------------------------------
  !     GETS THE NON-BONDED GENERATION CUTOFFS AND SWITCHING FUNCTION
  !     PARAMETERS. DEFAULTS FOR SWITCHING FUNCTION ARE RECALCULATED IF
  !     THE DISTANCE CUTOFF IS CHANGED.
  !
  !      By Bernard R. Brooks  (and others)    1983
  !
#if KEY_MMFF==1
  !      Jay Banks 23 Oct 95:
  !      Added truncation and "modified shift" options, MMFF-specific code.
#endif 
  !-----------------------------------------------------------------------

  use ewald,only: lewald,kappa,ewxmax, &
       ksqmax,kmax,kmaxx,kmaxy,kmaxz,parse_ewald,erfmod

  use pme_module,only: qpme,qfinit, &
       reqcor,rewcut,pmesh_setup,pmesh_clear
  use pmeutil,only: nfft1,nfft2,nfft3,forder
#if KEY_LJPME==1
  use pmeutil, only: DKAPPA, qljpme
#endif
#if KEY_GRAPE==1
  use grapemod, only: grapeini,grapefin                     
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use ace_module
  use contrl
  use defltsm
  use exelecm
  use image
  use inbnd
  use pbeq
  use ssbpm
  use psf
  use stream
  use string
  use fmam
  use mfmacons
  use grape
#if KEY_ACE==1
  use energym  
#endif
  use actclus_mod
#if KEY_PARALLEL==1
  use parallel       
#endif
#if KEY_MMFF==1
  use ffieldm  
#endif
  use datstr,only:freedt_nbond
  use memory
  use nbndcc_util
  use gopair, only : qgopair_upinb
#if KEY_BLOCK==1
  use block_fcm, only : qblock_excld_upinb
#endif
  use nbexcl,only:upinb
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN

  type(nonbondDataStructure) BNBND

  LOGICAL LTMP
  INTEGER NBXOLD,NBXMD,I,J
  LOGICAL LOOPS,DONE
  LOGICAL QELEC,QNOEL,QVDW,QNOVD,QGROU,QATOM,QCDIE,QRDIE
  LOGICAL QEXTD,QNOEX,QGRAD,QNOGR,QQUAD,QNOQU,FLGRF
#if KEY_GRAPE==1
  LOGICAL QGRAPE,QNOGRAP,QLIST,QNOLIST,QFMM,QNOFMM
#endif 
#if KEY_MMFF==1
  LOGICAL QTRUNC,QVTRUNC,QMSHI
#endif 
  LOGICAL QSWIT,QSHIF,QVSWI,QVSHI,QVDIS
  LOGICAL QGES,QGVS
  LOGICAL QVATOM,QVGROU,QEWALD,QNOEWA
  LOGICAL QBYCU,QBYGR,QFSWT,QFSHI,QVFSWT
  !rjp..02-FEB-99 BYCC
  LOGICAL QBYCC
#if KEY_IMCUBES==1
  LOGICAL QBYCBIM                                 
#endif
#if KEY_LRVDW==1
  LOGICAL QLRVDW,qnolrvdw                         
#endif
  LOGICAL QLRVDW_MS
  LOGICAL QINITL,QSTAND
  LOGICAL QFMA,QNOFMA
#if KEY_OPENMM==1
  logical :: qommrxn, qommswi                               
#endif
#if KEY_ACE==1
  LOGICAL QACE
  real(chm_real)  TEPSI,TEPSS,TACEAL,TSIGHY,TRSOLV
  real(chm_real)  TMXBSO,TTBSOL,TTBSOH,TSCTON,TSCTOF
  LOGICAL TQIDEA,TQCURR
  LOGICAL QACE2,QACE3,QQGAUS,QQPOIN
  real(chm_real)  TFISCL,TFSSCL,TFVSCL
#endif 
  !WXW IPS flag
#if KEY_NBIPS==1
  LOGICAL QIPS                    
#endif
  ! wxw double exponential
  LOGICAL QDBEXP                    
  real(chm_real)  C
  !   RJP:
  INTEGER NXI,NXIMAX,JJ,AT1
  LOGICAL QSMSTRUC
  !
  !!QQQQ
  !!      call PRINTDT_nbond('GTNBCT_entry','BNBND',BNBND)


  DONE=.TRUE.
  QINITL=.FALSE.
  QSTAND=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'INIT') > 0) THEN
     QINITL=.TRUE.
     QSTAND=.TRUE.
     !!      ELSE IF(.NOT.USEDDT_nbond(BNBND)) THEN
     !!         QINITL=.TRUE.
     !!         QSTAND=.TRUE.
  ELSE IF(INDXA(COMLYN,COMLEN,'RESE') > 0) THEN
     !        Do a reset of the nonbond data structure,
     !        but remember all cutoff options
     CALL GETBND(BNBND,.TRUE.)
     QINITL=.TRUE.
  ELSE
     Call GETBND(BNBND,.TRUE.)
     DONE=.FALSE.
  ENDIF

  IF(QINITL) THEN
     ! initialize-the-non-bond-data-structure
     !
     CALL FREEDT_nbond(BNBND)
     !
     allocate(bnbnd%INB14(200))
     allocate(bnbnd%ING14(200))
     !
     ! here initial values are set, but defaults may vary with the nbopt
     ! which is set in gtnbct. therefore cutoffs defaults are set there.
     !
     NNNB=0
     NNNBG=0
     NNB14=0
     NNG14=0
     NBXMOD=-9999
  ENDIF
  !
  IF(QSTAND) THEN
     ! set-standard-defaults
     CUTNB=DFCTNB
     CTONNB=DFCONB
     CTOFNB=DFCFNB
     CGONNB=DFGONB
     CGOFNB=DFGFNB
     WRNMIN=DFWMIN
     WRNMXD=0.5
     E14FAC=DFE14F
     ! JLB 23 OCT 95, SUGGESTED BY RCZ
     CUTIM = MAX(CUTNB, CUTIM)
     ! JLB 23 OCT 95, added CTVTRN, ##IF MMFF
     CTVTRN=10.0
#if KEY_MMFF==1
     IF (FFIELD == MMFF) THEN
        E14FAC=0.75D0
        !           write(6,*) ' Setting E14FAC TO 0.75 FOR MMFF'  ! TAH
        CTVTRN=8.
     ENDIF
#endif 
     EPS=DFEPS
     CTEXNB=999.0
     KAPPA=1.0
#if KEY_LJPME==1
     DKAPPA=0.35
#endif
     KMAX=5
     KSQMAX=27
     ERFMOD=-1
     !         NBSCAL=ONE
     IMSCAL=ONE
#if KEY_FMA==1
     Level= DFLevel
     Terms = DFTerm
#endif 
     !
     LELEC=.TRUE.
     LVDW=.TRUE.
     LGROUP=DFLGRP
     LCONS=DFLCNS
     LSHFT=DFLSHF
     LEGROM=DFLGES
     LVGROM=DFLGVS
     LVSHFT=DFLVSH
     LBYCU=DFLBYC
     !rjp..02-FEB-99 BYCC
     LBYCC=DFLBCC
#if KEY_IMCUBES==1
     LBYCBIM=DFLBYCI                                
#endif
#if KEY_LRVDW==1
     LLRVDW=DFLLRV                                 
#endif
     LVATOM=.NOT.LGROUP
     LFSWT=DFLFSW
     LVFSWT=DFLVFS
     LEWALD=.FALSE.
     LFma = .FALSE.
     QPME = .FALSE.
     QGRF = .FALSE.  ! Wei Chen 2015
#if KEY_OPENMM==1
     ! initialize logicals for OpenMM Reaction field
     qommrxn = .false.
     lommrxn = .false. 
     omm_rxnfld_dielectric = 80.0D0
     ! initialize logical for OpenMM vdW switch
     qommswi = .false.
     lommswi = .false.
#endif 
#if KEY_MMFF==1
     LTRUNC=.FALSE.
     LVTRUNC=.FALSE.
     IF(FFIELD == MMFF) THEN
        LVTRUNC=.TRUE.
        LVSHFT=.FALSE.
        LVFSWT=.FALSE.
        LCONS=.TRUE.
        LSHFT=.FALSE.
        LEGROM=.FALSE.
        LVGROM=.FALSE.
        LFSWT=.FALSE.
        LMSHFT=.TRUE.
     ENDIF
#endif 
     !...##IF SOFTVDW
     !LBIII- M. Vieth
     rgamin=Zero
     qgamin=.false.
     qelexp=.false.
     qvdwexp=.false.
     !...##ENDIF
     QRXNFL=.FALSE.
     RXNORD=10
     EPSEXT=80.0
     RXNSHL=2.0
     RXNMOD='ENERGY'
  ENDIF
  !
  !!QQQQ
  !!      call PRINTDT_nbond('GTNBCT_trime','BNBND',BNBND)


  CALL TRIME(COMLYN,COMLEN)
  IF(COMLEN > 0) DONE=.FALSE.
  IF(DONE) THEN
     CALL SETBND(BNBND)
     RETURN
  ENDIF
  !
  NBXOLD=NBXMOD
  IF (NBXMOD == -9999) NBXMOD=DFNBXM
  !
  QNOEL=(INDXA(COMLYN,COMLEN,'NOEL') > 0)
  QELEC=(INDXA(COMLYN,COMLEN,'ELEC') > 0)
  QNOVD=(INDXA(COMLYN,COMLEN,'NOVD') > 0)
  QVDW =(INDXA(COMLYN,COMLEN,'VDW ') > 0)
#if KEY_LRVDW==1
  QNOLRVDW=(INDXA(COMLYN,COMLEN,'NOLR') > 0)             
#endif
  QLRVDW_MS=(INDXA(COMLYN,COMLEN,'LRC_MS') > 0)
#if KEY_LRVDW==1
  QLRVDW=(INDXA(COMLYN,COMLEN,'LRC') > 0)                
#endif
  QATOM=(INDXA(COMLYN,COMLEN,'ATOM') > 0)
  QGROU=(INDXA(COMLYN,COMLEN,'GROU') > 0)
  QCDIE=(INDXA(COMLYN,COMLEN,'CDIE') > 0)
  QRDIE=(INDXA(COMLYN,COMLEN,'RDIE') > 0)
  QSWIT=(INDXA(COMLYN,COMLEN,'SWIT') > 0)
  QSHIF=(INDXA(COMLYN,COMLEN,'SHIF') > 0)
  QGES=(INDXA(COMLYN,COMLEN,'GSHI').GT.0)
  QGVS=(INDXA(COMLYN,COMLEN,'VGSH').GT.0)
  QVSWI=(INDXA(COMLYN,COMLEN,'VSWI') > 0)
  QVSHI=(INDXA(COMLYN,COMLEN,'VSHI') > 0)
  QVDIS=(INDXA(COMLYN,COMLEN,'VDIS') > 0)
  QVATOM=(INDXA(COMLYN,COMLEN,'VATO') > 0)
  QVGROU=(INDXA(COMLYN,COMLEN,'VGRO') > 0)
  QNOEX=(INDXA(COMLYN,COMLEN,'NOEX') > 0)
  FLGRF=(INDXA(COMLYN,COMLEN,'GRFE') > 0)   ! Wei Chen 2015
  QEXTD=(INDXA(COMLYN,COMLEN,'EXTE') > 0)
  QNOGR=(INDXA(COMLYN,COMLEN,'NOGR') > 0)
  QGRAD=(INDXA(COMLYN,COMLEN,'GRAD') > 0)
  QNOQU=(INDXA(COMLYN,COMLEN,'NOQU') > 0)
  QQUAD=(INDXA(COMLYN,COMLEN,'QUAD') > 0)
  QBYCU=(INDXA(COMLYN,COMLEN,'BYCU') > 0)
  !rjp..02-FEB-99 BYCC
  QBYCC=(INDXA(COMLYN,COMLEN,'BYCC') > 0)
#if KEY_IMCUBES==1
  QBYCBIM=(INDXA(COMLYN,COMLEN,'BYCB') > 0)              
#endif
  QBYGR=(INDXA(COMLYN,COMLEN,'BYGR') > 0)
  QFSWT=(INDXA(COMLYN,COMLEN,'FSWI') > 0)
  QFSHI=(INDXA(COMLYN,COMLEN,'FSHI') > 0)
  QVFSWT=(INDXA(COMLYN,COMLEN,'VFSW') > 0)
  QEWALD=(INDXA(COMLYN,COMLEN,'EWAL') > 0)
  QNOEWA=(INDXA(COMLYN,COMLEN,'NOEW') > 0)
  QFMA=(INDXA(COMLYN,COMLEN,'FMA') > 0)
  QNOFMA=(INDXA(COMLYN,COMLEN,'NOFMA') > 0)
  QDBEXP=(INDXA(COMLYN,COMLEN,'DBEXP') > 0).OR.LDBEXP
  IF(INDXA(COMLYN,COMLEN,'NODBEXP') > 0)THEN
    LDBEXP=.FALSE.
    QDBEXP=.FALSE.
  ENDIF
#if KEY_GRAPE==1
  !      QGRAPE=.FALSE.
  QGRAPE =(INDX(COMLYN,COMLEN,'GRAP',4) > 0)
  IGRAPE=GTRMI(COMLYN,COMLEN,'GRAP',0)
  QNOGRAP=(INDXA(COMLYN,COMLEN,'NOGR') > 0)
  QLIST  =(INDXA(COMLYN,COMLEN,'LIST') > 0)
  QNOLIST=(INDXA(COMLYN,COMLEN,'NOLI') > 0)
  CTOFNBNL=GTRMF(COMLYN,COMLEN,'CNOLI',-ONE)
  QFMM = (INDXA(COMLYN,COMLEN,'FMM') > 0)
  QNOFMM = (INDXA(COMLYN,COMLEN,'NOFMM') > 0)
  QCNOLIST=.FALSE.
  IF(CTOFNBNL >= ZERO)QCNOLIST=.TRUE.
  IF(QNOGRAP)CALL GRAPEFIN
#endif 
#if KEY_MMFF==1
  QMSHI=(INDXA(COMLYN,COMLEN,'MSHI') > 0)
  QTRUNC=(INDXA(COMLYN,COMLEN,'TRUN') > 0)
  QVTRUNC=(INDXA(COMLYN,COMLEN,'VTRU') > 0)
#endif 
#if KEY_ACE==1
  QACE=(INDXA(COMLYN,COMLEN,'ACE ') > 0)
  QACE2=(INDXA(COMLYN,COMLEN,'ACE2') > 0)
  QACE3=(INDXA(COMLYN,COMLEN,'ACE3') > 0)
  TQIDEA=(INDXA(COMLYN,COMLEN,'IDEA') > 0)
  TQCURR=(INDXA(COMLYN,COMLEN,'CURR') > 0)
  QQGAUS=(INDXA(COMLYN,COMLEN,'QGAU') > 0)
  QQPOIN=(INDXA(COMLYN,COMLEN,'QPOI') > 0)
  IF(QACE2) QACE=.TRUE.
  IF(QACE3) QACE=.TRUE.
  IF(QACE3) QACE2=.TRUE.
#endif 
  ! initial double exponential
  IF(QDBEXP)LDBEXP=QDBEXP
#if KEY_NBIPS==1 /*nbips*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  CALL IPSINPUT(COMLYN,COMLEN,QIPS,LVIPS,LEIPS,QGROU,QVGROU, &
       QEWALD,QSHIF,QFSHI,QSWIT,QFSWT,QVSWI,QVFSWT,QVSHI,LDBEXP &
#if KEY_LRVDW==1
       ,QLRVDW                                 & 
#endif
       )
#endif /* (nbips)*/
#if KEY_OPENMM==1
  qommrxn = (indxa(comlyn,comlen,'OMRF') > 0 )     
  qommswi = (indxa(comlyn,comlen,'OMSW') > 0 )     
#endif
  !
  LOOPS=(QELEC.AND.QNOEL).OR.(QVDW.AND.QNOVD).OR.(QGROU.AND.QATOM) &
       .OR.(QEXTD.AND.QNOEX).OR.(QGRAD.AND.QNOGR).OR.(QQUAD.AND.QNOQU) &
       .OR.(QCDIE.AND.QRDIE).OR.(QSWIT.AND.QSHIF).OR.(QVSWI.AND.QVSHI) &
       .OR.(QVATOM.AND.QVGROU).OR.(QBYCU.AND.QBYGR).OR.(QFSWT.AND.QSWIT) &
       .OR.(QFSWT.AND.QSHIF).OR.(QVFSWT.AND.QVSWI).OR.(QVFSWT.AND.QVSHI) &
       .OR.(QFSHI.AND.QSHIF).OR.(QFSHI.AND.QFSWT).OR.(QFSHI.AND.QSWIT) &
       .OR.(QGES.AND.QSHIF).OR.(QGES.AND.QSWIT).OR.(QGES.AND.QFSWT) &
       .OR.(QGES.AND.QFSHI) &
       .OR.(QGVS.AND.QSHIF).OR.(QGVS.AND.QSWIT).OR.(QGVS.AND.QFSWT) &
       .OR.(QGVS.AND.QFSHI) &

                                !rjp..02-FEB-99 BYCC
                                !rjp       .OR.(QEWALD.AND.QNOEWA)
       .OR.(QEWALD.AND.QNOEWA).OR.(QBYCC.AND.QBYGR).OR.(QBYCU.AND.QBYCC)
#if KEY_GRAPE==1
  LOOPS=LOOPS .OR.(QGRAPE.AND.QNOGRAP).OR.(QLIST.AND.QNOLIST)
#endif 
#if KEY_OPENMM==1
loops = loops .or. (qommrxn .and. qewald)          
#endif
  IF(LOOPS) THEN
     IF(WRNLEV >= 2.and.prnlev.ge.2) WRITE(OUTU,32)
32   FORMAT(' ***** ERROR IN GTNBCT ***** ', &
          'CONFLICT IN LOGICAL OPTIONS.')
     CALL DIEWRN(-2)
  ENDIF
  LOOPS=.FALSE.
#if KEY_MMFF==1
  LOOPS=LOOPS .OR.(QSHIF.AND.QTRUNC) &
       .OR.(QFSHI.AND.QTRUNC).OR.(QSWIT.AND.QTRUNC) &
       .OR.(QFSWT.AND.QTRUNC).OR.(QVSWI.AND.QVTRUNC) &
       .OR.(QVSHI.AND.QVTRUNC).OR.(QVFSWT.AND.QVTRUNC)
#endif 
#if KEY_ACE==1
  LOOPS=LOOPS  &
       .OR.(QACE.AND.QGROU).OR.(QACE.AND.QCDIE).OR.(QACE.AND.QRDIE) &
       .OR.(QACE.AND.QEWALD).OR.(QACE.AND.QEXTD).OR.(TQIDEA.AND.TQCURR) &
       .OR.(QQGAUS.AND.QQPOIN)
  ! not implemented: check for allowed switching functions
#endif 
#if KEY_NBIPS==1 /*nbips_loop*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  LOOPS=LOOPS .OR.(QEWALD.AND.LEIPS) &
       .OR.(QSHIF.AND.LEIPS).OR.(QFSHI.AND.LEIPS) &
       .OR.(QSWIT.AND.LEIPS).OR.(QFSWT.AND.LEIPS) &
       .OR.(QVSWI.AND.LVIPS).OR.(QVFSWT.AND.LVIPS) &
       .OR.(QVSHI.AND.LVIPS)
#endif /* (nbips_loop)*/
  IF(LOOPS) THEN
     IF(WRNLEV >= 2.and.prnlev.ge.2) WRITE(OUTU,33)
33   FORMAT(' ***** ERROR IN GTNBCT ***** ', &
          'CONFLICT IN SUPPORTED OPTIONS.')
     CALL DIEWRN(-2)
  ENDIF
  !
  IF (QELEC.OR.QNOEL) LELEC=QELEC
  IF (QVDW.OR.QNOVD) LVDW=QVDW
  IF (QGROU.OR.QATOM) THEN
     LGROUP=QGROU
     IF(.NOT.(QVATOM.OR.QVGROU)) LVATOM=.NOT.LGROUP
  ENDIF
  IF (QCDIE.OR.QRDIE) THEN
     LCONS=QCDIE
#if KEY_ACE==1
     LACE=.FALSE.    
#endif
  ENDIF
  !
#if KEY_GRAPE==1
  IF(QFMM.OR.QNOFMM) LFMM=QFMM    ! this only works if grape is also specified!
  IF(QGRAPE.OR.QNOGRAP) LGRAPE=QGRAPE
  IF(QLIST.OR.QNOLIST)  LNOCUT=QNOLIST
#endif 
  !
#if KEY_MMFF==1
  IF (QMSHI) THEN
     LMSHFT=.TRUE.
     LSHFT=.FALSE.
     LFSWT=.FALSE.
     LTRUNC=.FALSE.
  ENDIF
  IF (QSHIF) THEN
     LSHFT=.TRUE.
     LMSHFT=.FALSE.
     LFSWT=.FALSE.
     LTRUNC=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QGES) THEN
     LSHFT=.FALSE.
     LMSHFT=.FALSE.
     LFSWT=.FALSE.
     LTRUNC=.FALSE.
     LEGROM=.TRUE.
  ENDIF
  IF (QSWIT) THEN
     LSHFT=.FALSE.
     LMSHFT=.FALSE.
     LFSWT=.FALSE.
     LTRUNC=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QFSHI) THEN
     LSHFT=.TRUE.
     LMSHFT=.FALSE.
     LFSWT=.TRUE.
     LTRUNC=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QFSWT) THEN
     LSHFT=.FALSE.
     LMSHFT=.FALSE.
     LFSWT=.TRUE.
     LTRUNC=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QTRUNC) THEN
     LSHFT=.FALSE.
     LMSHFT=.FALSE.
     LFSWT=.FALSE.
     LTRUNC=.TRUE.
     LEGROM=.FALSE.
  ENDIF
  IF (QVSHI.OR.QVSWI) THEN
     LVSHFT=QVSHI
     LVFSWT=.FALSE.
     LVTRUNC=.FALSE.
     LVGROM=.FALSE.
  ENDIF
  IF (QGVS) THEN
     LVSHFT=.FALSE.
     LVFSWT=.FALSE.
     LVTRUNC=.FALSE.
     LVGROM=.TRUE.
  ENDIF
  IF (QVFSWT) THEN
     LVFSWT=.TRUE.
     LVTRUNC=.FALSE.
     LVGROM=.FALSE.
  ENDIF
  IF (QVTRUNC) THEN
     LVSHFT=.FALSE.
     LVFSWT=.FALSE.
     LVTRUNC=.TRUE.
     LVGROM=.FALSE.
  ENDIF
#else /**/
  IF (QSHIF) THEN
     LSHFT=.TRUE.
     LFSWT=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QGES) THEN
     LEGROM=.TRUE.
     LSHFT=.FALSE.
     LFSWT=.FALSE.
  ENDIF
  IF (QSWIT) THEN
     LSHFT=.FALSE.
     LFSWT=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (QFSHI) THEN
     LSHFT=.TRUE.
     LFSWT=.TRUE.
     LEGROM=.FALSE.
  ENDIF
  IF (QFSWT) THEN
     LSHFT=.FALSE.
     LFSWT=.TRUE.
     LEGROM=.FALSE.
  ENDIF
  IF (QVSHI.OR.QVSWI) THEN
     LVSHFT=QVSHI
     LVFSWT=.FALSE.
     LVGROM=.FALSE.
  ENDIF
  IF (QGVS) THEN
     LVGROM=.TRUE.
     LVSHFT=.FALSE.
     LVFSWT=.FALSE.
  ENDIF
#endif 
  IF (QVFSWT) LVFSWT=.TRUE.
  IF (QVATOM.OR.QVGROU) LVATOM=QVATOM
  IF (QEXTD.OR.QNOEX) QEXTND=QEXTD
  IF (QGRAD.OR.QNOGR) QXGRAD=QGRAD
  IF (QQUAD.OR.QNOQU) QXQUAD=QQUAD
  !rjp..02-FEB-99 BYCC
  !rjp      IF (QBYCU.OR.QBYGR) LBYCU=QBYCU
  IF (QBYCU.OR.QBYGR.OR.QBYCC) LBYCU=QBYCU
  IF (QBYCU.OR.QBYGR.OR.QBYCC) LBYCC=QBYCC
#if KEY_IMCUBES==1
  IF (QBYCBIM) LBYCBIM=QBYCBIM
  if(lbycbim)then
     if(qbycbim.and.lbycbim)then
        qbycu=.false.
        qbycc=.false.
        qbygr=.false.
     endif
     if(qbycu.or.qbycc.or.qbygr)LBYCBIM=.false.
  endif
#endif 
#if KEY_LRVDW==1
  IF (QLRVDW.OR.QNOLRVDW) LLRVDW=QLRVDW
  IF (QLRVDW_MS.OR.QNOLRVDW) LLRVDW_MS=QLRVDW_MS

  IF (QLRVDW_MS .AND. (.NOT. QVSWI)) CALL wrndie(-5, '<GTNBCT>', &
         'LRC_MS needs vswitch on!')

#if KEY_PARALLEL==1
  if(mynod /= 0)LLRVDW=.false.                
#endif
#if KEY_PARALLEL==1
  if(mynod /= 0)LLRVDW_MS=.false.                
#endif
#endif
  IF (QEWALD.OR.QNOEWA) LEWALD=QEWALD
  if (QFma .or. QNoFma) LFma = QFma
#if KEY_OPENMM==1
  lommrxn = qommrxn                 
  lommswi = qommswi                 
#endif
#if KEY_FMA==1
  !
  !   FMA
  if (Lfma) then
     Level = GTRMI (Comlyn, Comlen, 'LEVEL', Level)
     Terms = GTRMI (Comlyn, Comlen, 'TERMS', Terms)
     if (Level  <  1) then
        call WRNDIE (-1, '<GTNBCT>', &
             'Invalid LEVEL for Multipole Expansions')
     endif
     if (Terms  <  2) then
        call WRNDIE (0,  '<GTNBCT>', &
             'For valid Multipole expansions, use TERMS > 2')
     endif
  endif
#endif 
#if KEY_NBIPS==1 /*nbips_flag*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  IF (LEIPS) THEN
     LEWALD=.FALSE.
     LSHFT=.FALSE.
     LFSWT=.FALSE.
     LEGROM=.FALSE.
  ENDIF
  IF (LVIPS) THEN
     LVSHFT=.FALSE.
     LVFSWT=.FALSE.
#if KEY_LRVDW==1
     LLRVDW=.FALSE.                 
#endif
     LVGROM=.FALSE.
  ENDIF
  IF(QIPS.AND.LEIPS)THEN
     LEWALD=.FALSE.
     QPME=.FALSE.
  ENDIF
#endif /*  (nbips_flag)*/
#if KEY_OPENMM==1
  if(lommrxn) then
     lewald = .false.
     qpme = .false.
     lshft = .false.
     lfswt = .false.
     legrom = .false.
  endif
#endif 

  IF (FLGRF) QGRF=.TRUE. !reaction field -- Wei Chen 2015

  !
  !-----------------------------------------------------------------------
  ! Check for valid nonbond options.  Reset flags as needed.
  !
  IF(.NOT.LGROUP.AND..NOT.LVATOM) THEN
     CALL WRNDIE(-3,'<GTNBCT>', &
          'Group VDW only available with group electrostatics')
     LVATOM=.TRUE.
  ENDIF
  IF(LGROUP.AND.LVATOM) THEN
     CALL WRNDIE(-3,'<GTNBCT>', &
          'Atom VDW with group electrostatics is no longer supported')
  ENDIF
  IF (LGROUP.AND.LVGROM) THEN
     CALL WRNDIE(-5,'<GTNBCT>', &
             'GROUP GROMACS VDW SHIFT option is NOT valid.')
  ENDIF

  !
  ! Electrostatics
  !
  !CC      IF(LGROUP.AND.LEWALD) THEN
  !CC         CALL WRNDIE(-3,'<GTNBCT>',
  !CC     1        'Ewald with group electrostatics is not implemented yet.')
  !CC         LEWALD=.FALSE.
  !CC      ENDIF
  !
  IF(QEXTND .AND. .NOT.LGROUP) THEN
     CALL WRNDIE(-2,'<GTNBCT>', &
          'Extended Electrostatics valid only with GROUP option')
  ENDIF
  !
  IF (LGROUP.AND.LSHFT) THEN
     CALL WRNDIE(-2,'<GTNBCT>','GROUP SHIFT option is NOT valid.')
     LSHFT=.FALSE.
     LFSWT=.TRUE.
  ENDIF
  !
  ! GRF -- Wei Chen 2015
  IF(QGRF .and. .not. LTRUNC) THEN
     CALL WRNDIE(-2,'<GTNBCT>', &
          'Generalized Reaction Field Electrostatics valid only with TRUNC option')
  ENDIF
  !
  IF (LGROUP.AND.LEGROM) THEN
     CALL WRNDIE(-5,'<GTNBCT>', &
             'GROUP GROMACS Elec SHIFT option is NOT valid.')
  ENDIF
  !
  ! End of nonbond flag checking.
  !-----------------------------------------------------------------------
  !
#if KEY_OPENMM==1
  if(qommrxn) omm_rxnfld_dielectric = gtrmf(comlyn,comlen,'OMRX',omm_rxnfld_dielectric) 
#endif
  IF(QEXTND) THEN
     ! this intializes the extended electrostatics data structure
     IF(INDXA(COMLYN,COMLEN,'NORXN') > 0) QRXNFL=.FALSE.
     IF(INDXA(COMLYN,COMLEN,'RXNFLD') > 0) THEN
        QRXNFL=.TRUE.
        RXNMOD='ENERGY'
     ENDIF
     IF(INDXA(COMLYN,COMLEN,'RXNNB') > 0) THEN
        QRXNFL=.TRUE.
        RXNMOD='NONBOND'
     ENDIF
     IF(QRXNFL) THEN
        EPSEXT=GTRMF(COMLYN,COMLEN,'EPSEXT',EPSEXT)
        RXNORD=GTRMI(COMLYN,COMLEN,'ORDER',RXNORD)
        RXNSHL=GTRMF(COMLYN,COMLEN,'SHELL',RXNSHL)
     ENDIF
     CALL ALLEXT
  ELSE
     QRXNFL=.FALSE.
  ENDIF

  EPS=GTRMF(COMLYN,COMLEN,'EPS',EPS)
  IF(FLGRF)THEN ! reaction field -- Wei Chen 2015
     RFEPSO = GTRMF(COMLYN,COMLEN,'EPSO',80.0D0) ! eps outside cut-off radius
     RFIONI = GTRMF(COMLYN,COMLEN,'IONI',ZERO) ! ionic strength
     RFTEMP = GTRMF(COMLYN,COMLEN,'TEMR',298.0D0) ! temperature
     RFKAP = SQRT((2530.362733d0*RFIONI)/(RFTEMP*RFEPSO)) ! debye screening length (1/angstrom)
     RFCON = ((TWO*RFEPSO - TWO*EPS)*(ONE+RFKAP*CTOFNB)+(RFEPSO*RFKAP*RFKAP*CTOFNB*CTOFNB)) / &
                ((EPS + TWO*RFEPSO)*(ONE+RFKAP*CTOFNB)+(RFEPSO*RFKAP*RFKAP*CTOFNB*CTOFNB))
  ENDIF
  !
  !     AND THE CUTOFFS, RESETTING THE INNER CUTOFFS IF OUTER ONES
  !     ARE SPECIFIED.
  !
  !yw...24-JUL-93, CUTIM is set to the new CUTNB value when CUTIM < CUTNB
  C=CUTNB
  CUTNB=GTRMF(COMLYN,COMLEN,'CUTNB',CUTNB)
  IF (CUTNB /= C) THEN
     CTOFNB=CUTNB-(DFCTNB-DFCFNB)
     CTONNB=CUTNB-(DFCTNB-DFCONB)
     IF (NATIM > NATOM .AND. CUTIM < CUTNB) THEN
        C=CUTIM
        CUTIM=GTRMF(COMLYN,COMLEN,'CUTI',CUTNB)
        IF (PRNLEV >= 2 .AND. CUTIM /= C.and.prnlev.ge.2) THEN
           WRITE(OUTU,'(A)') ' ***** Info from GTNBCT *****'
           WRITE(OUTU,'(2(A,F5.1))') ' CUTIM is reset to ', &
                CUTIM,' from ',C
        ENDIF
     ENDIF
  ENDIF
  C=CTOFNB
  CTOFNB=GTRMF(COMLYN,COMLEN,'CTOFNB',CTOFNB)
  IF(CTOFNB /= C) CTONNB=CTOFNB-(DFCFNB-DFCONB)
  CTONNB=GTRMF(COMLYN,COMLEN,'CTONNB',CTONNB)
  CTEXNB=GTRMF(COMLYN,COMLEN,'CTEXNB',CTEXNB)
  WRNMIN=GTRMF(COMLYN,COMLEN,'WMIN',WRNMIN)
  WRNMXD=GTRMF(COMLYN,COMLEN,'WRNMXD',WRNMXD)
  E14FAC=GTRMF(COMLYN,COMLEN,'E14F',E14FAC)
  CTVTRN=GTRMF(COMLYN,COMLEN,'CTVT',CTVTRN)
  CGONNB=GTRMF(COMLYN,COMLEN,'CGONNB',CGONNB)
  CGOFNB=GTRMF(COMLYN,COMLEN,'CGOFNB',CGOFNB)
  !
  ! PARSE SOFT CORE POTENTIAL
  !
#if KEY_SOFTVDW==1
  Qvdwexp=(INDXA(COMLYN,COMLEN,'VDWE') > 0)
  qelexp=(INDXA(COMLYN,COMLEN,'ELEE') > 0)
  qgamin=(INDXA(COMLYN,COMLEN,'SOFT') > 0)

  if(qgamin) then
     if(lcons)then
        rgamin=30.0/eps
        rgamin=GTrmf(Comlyn,Comlen,'EMAX', rgamin)
        egamina=-300.0/eps
        egamina=GTrmf(Comlyn,Comlen,'MINE', egamina)
        egaminr=-egamina*2.0
        egaminr=GTrmf(Comlyn,Comlen,'MAXE', egaminr)

     else

        rgamin=15.0/eps
        rgamin=GTrmf(Comlyn,Comlen,'EMAX', rgamin)
        egamina=-120.0/eps
        egamina=GTrmf(Comlyn,Comlen,'MINE', egamina)
        egaminr=-egamina*2.0
        egaminr=GTrmf(Comlyn,Comlen,'MAXE', egaminr)

     endif
  endif
  !
  ! separate local parsing variables to avoid confusion
  !
  ! Bugfix from M. Vieth so that softcore potential active on
  ! multiple energy calls w/o respecification. clbiii
  if( LVFSWT .or. &
       (.NOT.LVFSWT .AND. .NOT.LVSHFT )) then

     if( (LCONS .AND. .NOT.LSHFT) &
          .or.(.NOT.LCONS .AND..NOT.LFSWT)) then
        if(rgamin > Zero.and.prnlev >= 2) then
           WRITE(OUTU,'(/a/,a/,a/,a)') ' **** SOFT CORE AVAILABLE ', &
                '       SUGGESTED OPTIONS : RDIE SWIT VSWIT ', &
                ' FOR SPC WATER in CDIE USE : EMAX > 1000/EPS OR MINE=-100/EPS,' &
                ,' FOR SPC WATER in RDIE USE : EMAX > 200/EPS'
        endif
     else
        rgamin=Zero
        IF(WRNLEV >= 2.and.qgamin.and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a/,a)') ' **** WARNING: SOFT CORE', &
             ' POTENTIAL IS AVAILABLE ONLY WITH :', &
             ' VSWIT, VFSHIFT, CFSWIT, CSWIT, RSHIFT, RSWIT '
     endif
  else
     rgamin=Zero
  endif
  !
  if(rgamin > Zero) then
     qgamin=.true.
     if(LCONS) then
        IF(WRNLEV >= 2.and.rgamin < (Thirty/eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             '   - VDW core too soft with ', &
             'respect to electrostatics ', &
             ' **** INCREASE EMAX to at least ', 30.0/eps
        IF(WRNLEV >= 2.and.egamina > (-300/eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             ' - Elec attractive soft core ', &
             'starts at too high Rcut ', &
             ' **** DECREASE MINE to at least ',-300.0/eps
        IF(WRNLEV >= 2.and.egaminr < (600.0/eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             ' - Elec repulsive soft core starts at too high Rcut ', &
             ' **** INCREASE MAXA to at least ', 600/eps

     else

        IF(WRNLEV >= 2.and.rgamin < (15.0/eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             ' - VDW core too soft with respect to electrostatics ', &
             ' **** INCREASE EMAX to at least ', 15.0/eps
        IF(WRNLEV >= 2.and.egamina > (-100./eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             ' - Elec attractive soft core starts at too high Rcut', &
             ' **** DECREASE MINE to at least ',-100.0/eps
        IF(WRNLEV >= 2.and.egaminr < (200.0/eps).and.prnlev.ge.2) &
             WRITE(OUTU,'(/a/,a/,a,f10.2)') &
             ' ****  ERROR :POSSIBLE UNPHYSICAL RESULTS', &
             '  - Elec repulsive soft core starts at too high Rcut', &
             ' **** INCREASE MAXA to at least ', 200.0/eps
     endif

  else
     qgamin=.false.
  endif
#endif 
  !
  !-----------------------------------------------------------------------
  ! Parse ACE parameters
#if KEY_ACE==1
  IF(QACE.OR.LACE) THEN
     TEPSI =GTRMF(COMLYN,COMLEN,'IEPS',EPSI)
     TEPSS =GTRMF(COMLYN,COMLEN,'SEPS',EPSS)
     TACEAL=GTRMF(COMLYN,COMLEN,'ALPH',ACEAL)
     TSIGHY=SIGHYD*1000.0
     TSIGHY=GTRMF(COMLYN,COMLEN,'SIGM',TSIGHY)
     TSIGHY=0.001*TSIGHY
     TRSOLV=GTRMF(COMLYN,COMLEN,'RSOL',RSOLV)
     TRSOLV=GTRMF(COMLYN,COMLEN,'RSOL',RSOLV)
     TMXBSO=GTRMF(COMLYN,COMLEN,'MXBS',MXBSOL)
     TTBSOL=GTRMF(COMLYN,COMLEN,'TBSO',TBSOLV)
     TTBSOH=GTRMF(COMLYN,COMLEN,'TBSH',TBSOLH)
     TSCTON=GTRMF(COMLYN,COMLEN,'SCON',SCTON)
     TSCTOF=GTRMF(COMLYN,COMLEN,'SCOF',SCTOF)
     TFISCL=GTRMF(COMLYN,COMLEN,'FISC',FISCAL)
     TFSSCL=GTRMF(COMLYN,COMLEN,'FSSC',FSSCAL)
     TFVSCL=GTRMF(COMLYN,COMLEN,'FVSC',FVSCAL)
     LQGAUS=((LQGAUS.OR.QQGAUS).AND.(.NOT.QQPOIN))
     IF((QACE.NEQV.LACE).OR.TEPSI /= EPSI.OR.TEPSS.NE.EPSS &
          .OR.(TACEAL /= ACEAL).OR.(QFIACE == 0) &
          .OR.(LIDEAL.AND.TQCURR).OR.(.NOT.(LIDEAL).AND.TQIDEA) &
          .OR.TMXBSO /= MXBSOL.OR.TTBSOL.NE.TBSOLV &
          .OR.TTBSOH /= TBSOLH &
          .OR.TSCTON /= SCTON.OR.TSCTOF.NE.SCTOF &
          .OR.(QACE2.NEQV.LACE2).OR.(QACE3.NEQV.LACE3) &
          .OR.FISCAL /= TFISCL.OR.FSSCAL.NE.TFSSCL &
          .OR.FVSCAL /= TFVSCL &
          ) THEN
        LACE =.TRUE.
        EPSI =TEPSI
        EPSS =TEPSS
        ACEAL=TACEAL
        SIGHYD=TSIGHY
        RSOLV=TRSOLV
        !         initialize ESFIXA array if SCTON or SCTOF have changed:
        IF((SCTON /= TSCTON).OR.(SCTOF.NE.TSCTOF)) QFIACE=0
        MXBSOL=TMXBSO
        TBSOLV=TTBSOL
        TBSOLH=TTBSOH
        SCTON=TSCTON
        SCTOF=TSCTOF
        !         copy "normal" cuton/off to self energy cuton/off if
        !         the latter are not initialized (negative):
        IF ((SCTON <= ZERO).OR.(SCTOF.LE.ZERO)) THEN
           SCTOF=CTOFNB
           SCTON=CTONNB
        ENDIF
        !         leave ACE2 switched on, unless the nonbonded keyword "ACE "
        !         has just been issued without the keyword "ACE2":
        LACE2=((LACE2.OR.QACE2).AND.(QACE.EQV.QACE2))
        !         leave ACE3 switched on, unless the nonbonded keywords "ACE "
        !         or "ACE2" have just been issued without the keyword "ACE3":
        LACE3=((LACE3.OR.QACE3).AND.(QACE.EQV.QACE3).AND. &
             (QACE2.EQV.QACE3))
        !         initialize scaled ionic charges if scaling factors have changed;
        IF ((FISCAL /= TFISCL).OR.(FSSCAL.NE.TFSSCL)) QFIACE=0
        !         initialize ESII, ESFIXA arrays if volumes (FVSCAL) have changed:
        IF(FVSCAL /= TFVSCL) QFIACE=0
        !         initialize ESFIXA array if bonded distance option has changed:
        IF((LIDEAL.AND.TQCURR).OR. &
             (.NOT.(LIDEAL).AND.TQIDEA)) QFIACE=0
        LIDEAL=((LIDEAL.OR.TQIDEA).AND.(.NOT.TQCURR))
        FISCAL=TFISCL
        FSSCAL=TFSSCL
        FVSCAL=TFVSCL
        !
        !         first allocate memory for self energy potential; the
        !         space for ACE atom (psf) arrays and atom pair arrays is
        !         assigned and updated (accoding to changes in the pair list)
        !         in the subroutine ENBOND (.../source/nbonds/enbond.src):
        !
        IF (ATCACE /= MAXATC) THEN
           IF (ATCACE > 0) THEN
              call chmdealloc('nbutil.src','GTNBCT','CES1',ATCACE,ATCACE,crl=CES1)
              call chmdealloc('nbutil.src','GTNBCT','CES2',ATCACE,crl=CES2)
              call chmdealloc('nbutil.src','GTNBCT','SIG2I',ATCACE,ATCACE,crl=SIG2I)
              call chmdealloc('nbutil.src','GTNBCT','MUE4',ATCACE,ATCACE,crl=MUE4)
              call chmdealloc('nbutil.src','GTNBCT','KHYD',ATCACE,crl=KHYD)
              call chmdealloc('nbutil.src','GTNBCT','ESII',ATCACE,crl=ESII)
           ENDIF
           ATCACE = MAXATC
           call chmalloc('nbutil.src','GTNBCT','CES1',ATCACE,ATCACE,crl=CES1)
           call chmalloc('nbutil.src','GTNBCT','CES2',ATCACE,crl=CES2)
           call chmalloc('nbutil.src','GTNBCT','SIG2I',ATCACE,ATCACE,crl=SIG2I)
           call chmalloc('nbutil.src','GTNBCT','MUE4',ATCACE,ATCACE,crl=MUE4)
           call chmalloc('nbutil.src','GTNBCT','KHYD',ATCACE,crl=KHYD)
           call chmalloc('nbutil.src','GTNBCT','ESII',ATCACE,crl=ESII)
        ENDIF
        !
        !         does not harm if ACEINI is called again (e.g., after
        !         changing epsilon parameters), even though it is required
        !         only at the first call to ACE energy or if FVSCAL, SIGHYD,
        !         or RMIN/EFVOL (atom type arrays) have changed -- FVSCAL
        !         scales atom volumes, RMIN is the minimum charge radius
        !         of the atom type:
        CALL ACEINI()
     ENDIF
     !       check whether self energy cut-off limits make sense:
     IF (SCTOF > CTOFNB) THEN
        CALL WRNDIE(5,'<GTNBCT>', &
             'SCTOF>CTOFNB! restting to SCTON/OF=CTON/OFNB')
        SCTOF=CTOFNB
        SCTON=CTONNB
     ENDIF
     !       check whether TBSOL is >0 and <MXBSOL:
     IF (TBSOLV <= ZERO) THEN
        CALL WRNDIE(2,'<GTNBCT>', &
             'TBSOLV<=0! restting to TBSOLV=MXBSOL/2')
        TBSOLV=HALF*MXBSOL
     ENDIF
     IF (TBSOLV >= MXBSOL) THEN
        CALL WRNDIE(2,'<GTNBCT>', &
             'TBSOLV>=MXBSOL! restting to TBSOLV=MXBSOL/2')
        TBSOLV=HALF*MXBSOL
     ENDIF
     !       check whether TBSOLH is >0 and <MXBSOL:
     IF (TBSOLH <= ZERO) THEN
        CALL WRNDIE(2,'<GTNBCT>', &
             'TBSOLH<=0! restting to TBSOLH=MXBSOL/2')
        TBSOLH=HALF*MXBSOL
     ENDIF
     IF (TBSOLH >= MXBSOL) THEN
        CALL WRNDIE(2,'<GTNBCT>', &
             'TBSOLH>=MXBSOL! restting to TBSOLH=MXBSOL/2')
        TBSOLH=HALF*MXBSOL
     ENDIF
  ELSE
     !       ACE is not used, zero the ACE energy properties (this avoids
     !       them being printed "mistakenly" by PRINTE):
     EPROP(SELF)=ZERO
     EPROP(SCREEN)=ZERO
     EPROP(COUL)=ZERO
     EPROP(SOLV)=ZERO
     EPROP(INTER)=ZERO
     !       free reserved for parameters of Eself:
     IF (ATCACE > 0) THEN
        call chmdealloc('nbutil.src','GTNBCT','CES1',ATCACE,ATCACE,crl=CES1)
        call chmdealloc('nbutil.src','GTNBCT','CES2',ATCACE,crl=CES2)
        call chmdealloc('nbutil.src','GTNBCT','SIG2I',ATCACE,ATCACE,crl=SIG2I)
        call chmdealloc('nbutil.src','GTNBCT','MUE4',ATCACE,ATCACE,crl=MUE4)
        call chmdealloc('nbutil.src','GTNBCT','KHYD',ATCACE,crl=KHYD)
        call chmdealloc('nbutil.src','GTNBCT','ESII',ATCACE,crl=ESII)
        ATCACE=0
     ENDIF
     !       free reserved for atom arrays in ACE:
     IF(NATACE > 0) THEN
        call chmdealloc('nbutil.src','GTNBCT','CGIACP',NATACE,crl=CGIACP)
        call chmdealloc('nbutil.src','GTNBCT','CGSACP',NATACE,crl=CGSACP)
        call chmdealloc('nbutil.src','GTNBCT','DESDBP',NATACE,crl=DESDBP)
        call chmdealloc('nbutil.src','GTNBCT','ESFIX',NATACE,crl=ESFIX)
        call chmdealloc('nbutil.src','GTNBCT','ESELF',NATACE,crl=ESELF)
        call chmdealloc('nbutil.src','GTNBCT','BSOLV',NATACE,crl=BSOLV)
        call chmdealloc('nbutil.src','GTNBCT','DISUM',NATACE,crl=DISUM)
        call chmdealloc('nbutil.src','GTNBCT','XFREA',NATACE,crl=XFREA)
        call chmdealloc('nbutil.src','GTNBCT','YFREA',NATACE,crl=YFREA)
        call chmdealloc('nbutil.src','GTNBCT','ZFREA',NATACE,crl=ZFREA)
        call chmdealloc('nbutil.src','GTNBCT','CG2',NATACE,crl=CG2)
        call chmdealloc('nbutil.src','GTNBCT','DBDE',NATACE,crl=DBDE)
        NATACE=0
     ENDIF
     !       free reserved for nonbonded pair arrays in ACE:
     IF(NPAIR > 0) THEN
        call chmdealloc('nbutil.src','GTNBCT','SWIT',NPAIR,crl=SWIT)
        call chmdealloc('nbutil.src','GTNBCT','DSWIT',NPAIR,crl=DSWIT)
        call chmdealloc('nbutil.src','GTNBCT','DISTM',NPAIR,crl=DISTM)
        call chmdealloc('nbutil.src','GTNBCT','XFSDI',2*NPAIR,crl=XFSDI)
        call chmdealloc('nbutil.src','GTNBCT','YFSDI',2*NPAIR,crl=YFSDI)
        call chmdealloc('nbutil.src','GTNBCT','ZFSDI',2*NPAIR,crl=ZFSDI)
        NPAIR=0
     ENDIF
     !       free reserved for bonded pair arrays in ACE:
     IF(NPA14 > 0) THEN
        call chmdealloc('nbutil.src','GTNBCT','SA14P',NPA14,crl=SA14P)
        call chmdealloc('nbutil.src','GTNBCT','SWA14P',NPA14,crl=SWA14P)
        NPA14=0
     ENDIF
     !       reset QFIACE to 0 to indicate ACE arrays are not initialized:
     QFIACE=0
     !       reset SIGHYD to -1 to indicate that it is not initialized
     !       at a later call to ACE:
     SIGHYD=-ONE
  ENDIF
#endif /*  ACE*/
  !br...060710
#if KEY_PBEQ==1
  IF(QGSBP)THEN
     IF(.NOT.QEXTND)THEN
        IF(CUTNB < 2*SRDIST)THEN
           IF(PRNLEV >= 2) WRITE(OUTU,'(6X,2(A,F6.2))')  &
                'CUTNB=',CUTNB,' 2*SRDIST=',2*SRDIST
           CALL WRNDIE(-5,'<GTNBCT>', 'CUTNB smaller than GSBP radius')
        ENDIF
        IF(QRDIE)THEN
           CALL WRNDIE(-5,'<GTNBCT>', 'RDIE incompatible with GSBP')
        ENDIF
        IF(LSHFT .OR. LMSHFT .OR. LFSWT .OR. LEGROM &
             .OR. LTRUNC)THEN
           CALL WRNDIE(-5,'<GTNBCT>', 'only SWITCH is compatible with GSBP')
        ENDIF
     ENDIF
  ENDIF
#endif 
  IF(QSSBP)THEN
     IF(.NOT.QEXTND)THEN
        CALL WRNDIE(0,'<GTNBCT>', 'EXT elec recommended for SSBP')
     ENDIF
     IF(QRDIE)THEN
        CALL WRNDIE(-5,'<GTNBCT>', 'RDIE incompatible with SSBP')
     ENDIF
     IF(LSHFT .OR. LMSHFT .OR. LFSWT .OR. LEGROM &
          .OR. LTRUNC)THEN
        CALL WRNDIE(-5,'<GTNBCT>', 'only SWITCH is compatible with SSBP')
     ENDIF
  ENDIF
  !br...060710
  !
  !-----------------------------------------------------------------------
  ! Parse the ewald options and  flags
  IF(LEWALD &
#if KEY_GRAPE==1
       .OR.LGRAPE & 
#endif
       )then
     call parse_ewald(comlyn, comlen)
  endif
#if KEY_LJPME==1
  ! Try to catch all known incompatibilities (due to not being implemented,
  ! rather than an inherent problem) with the LJPME code.
  if(qljpme) then
     if(QVFSWT) then
        CALL WRNDIE(-5,'<GTNBCT>', 'VFSWitch is not compatible with LJPMe')
     endif
     if(QVGROU) then
        CALL WRNDIE(-5,'<GTNBCT>', 'VGROup is not compatible with LJPMe')
     endif
  endif
#endif
  !-----------------------------------------------------------------------
#if KEY_NBIPS==1 /*nbips_set*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  IF (QIPS.OR.LDBEXP)THEN
     CALL IPSSET(CUTNB,CTOFNB,LCONS,LVIPS,LEIPS,LDBEXP)
  ENDIF
#endif /*  (nbips_set)*/
  !
  IMSCAL=GTRMF(COMLYN,COMLEN,'IMSC',ONE)
  NBSCAL=GTRMF(COMLYN,COMLEN,'NBSC',NBSCAL)
  NBXMD=GTRMI(COMLYN,COMLEN,'NBXM',NBXMOD)
  NBXMOD=NBXMD
  !
  CALL SETBND(BNBND)
  CALL PRNBCT(BNBND)
  !
  IF(CUTNB-CTOFNB < ONE.AND.INBFRQ.LT.0) CALL WRNDIE(1,'<GTNBCT>', &
       'CUTNB and CTOFNB are too close for efficient heuristic update.')
  IF(CUTNB <= CTOFNB .OR. CTOFNB.LE.CTONNB) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,'(A,3F10.1)') &
          ' GTNBCT> CUTNB,CTOFNB,CTONNB=',CUTNB,CTOFNB,CTONNB
     CALL WRNDIE(1,'<GTNBCT>', &
          'CUTNB,CTOFNB,CTONNB are not in correct order.')
  ENDIF
  IF (LEGROM) THEN
     IF(CUTNB-CGOFNB.LT.ONE.AND.INBFRQ.LT.0) THEN
        CALL WRNDIE(1,'<GTNBCT>', &
             'CUTNB and CGOFNB are too close for efficient heuristic update.')
     ENDIF
     IF(CUTNB.LE.CGOFNB .OR. CGOFNB.LE.CGONNB) THEN
        IF(PRNLEV.GE.2) THEN
           WRITE(OUTU,'(A,3F10.1)') &
                ' GTNBCT> CUTNB,CGOFNB,CGONNB=',CUTNB,CGOFNB,CGONNB
        ENDIF
        CALL WRNDIE(1,'<GTNBCT>', &
             'CUTNB,CGOFNB,CGONNB are not in correct order.')
     ENDIF
  ENDIF
  !
  !RCZ  25-OCT-91 modify IF test
  LTMP=INDXA(COMLYN,COMLEN,'EXOF') > 0
  QLBYCC = LBYCC

#if KEY_BLOCK==1
  IF(qblock_excld_upinb .or. &
#else
  IF( &
#endif
     qgopair_upinb .or. LTMP .or. NBXOLD /= NBXMD) CALL UPINB(BNBND)

  ! make clusters if not already done -RJP 10.03
  ! or if change in connectivity detected
  IF(LBYCC) THEN
     QSMSTRUC = .TRUE.
     IF ((NATOM /= NATOMST).OR.(NRES.NE.NRESST).OR.(NSEG.NE.NSEGST) &
          .OR.(NBOND /= NBONDST).OR.(NTHETA.NE.NTHETAST).OR. &
          (NPHI /= NPHIST).OR.(NIMPHI.NE.NIMPHIST)) THEN
        QSMSTRUC = .FALSE.
        IF ((QCLSMD).AND.(PRNLEV >= 2)) WRITE(OUTU,'(A)') &
             'CHANGE DETECTED IN STRUCTURE OR CONNECTIVITY VARIABLES'
     ENDIF
     NATOMST = NATOM
     NRESST = NRES
     NSEGST = NSEG
     NBONDST = NBOND
     NTHETAST = NTHETA
     NPHIST = NPHI
     NIMPHIST = NIMPHI
     ! if either the structural parameters have changed or the clusters have
     ! not been made, make clusters
     IF((.NOT.QCLSMD).OR.(.NOT.QSMSTRUC)) CALL MKCLUSTDEF
  ENDIF !if bycc
  !RCZ
#if KEY_GRAPE==1
  IF(QGRAPE)CALL GRAPEINI(BNBND)     
#endif
  !

  !!QQQQ
  !!      call PRINTDT_nbond('GTNBCT_return','BNBND',BNBND)

  RETURN
END SUBROUTINE GTNBCT

SUBROUTINE GETBND(BNBND,QOPTNS)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE FILLS THE NONBOND DISTANCES AND FLAGS FROM THE
  !     DATA STRUCTURE
  !
  !      By Bernard R. Brooks    1983
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use inbnd
  implicit none
  type(nonbondDataStructure) BNBND
  LOGICAL QOPTNS
  !
  CALL GETBN2(BNBND%LNBOPT,BNBND%NBDIST, &
       BNBND%NBINTS,QOPTNS)
  RETURN
END SUBROUTINE GETBND

SUBROUTINE GETBN2(QNB,RNB,INB,QOPTNS)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use inbnd
  implicit none
  !
  LOGICAL QNB(*)
  real(chm_real)  RNB(*)
  INTEGER INB(*)
  LOGICAL QOPTNS
  !
  ! Get flags and values based on parsed options
  IF(QOPTNS) THEN
     ! set up logical flags
     LELEC = QNB(1)
     LVDW  = QNB(2)
     LGROUP= QNB(3)
     LCONS = QNB(4)
     LSHFT = QNB(5)
     LVATOM= QNB(6)
     LVSHFT= QNB(7)
     LBYCU = QNB(8)
     LFSWT = QNB(9)
     LVFSWT= QNB(10)
#if KEY_IMCUBES==1
     LBYCBIM= QNB(11)                                
#endif
     LFMA  = QNB(12)
     !...##IF MMFF
     !     Jay Banks 23 Oct 95: added LTRUNC,LVTRUNC,LMSHFT,CTVTRN
     LTRUNC= QNB(13)
     LVTRUNC=QNB(14)
     LMSHFT= QNB(15)
     !...##ENDIF
     !...##IF SOFTVDW
     QGAMIN= QNB(16)
     QELEXP= QNB(17)
     QVDWEXP= QNB(18)
     !...##ENDIF
     !rjp..02-FEB-99 BYCC
     LBYCC = QNB(19)
#if KEY_LRVDW==1
     LLRVDW= QNB(20)    
#endif
#if KEY_NBIPS==1 /*nbips*/
     !WXW Long range potential using isotropic periodic sum (IPS)
     LEIPS=QNB(21)
     LVIPS=QNB(22)
#endif /* (nbips)*/
     QETEN=QNB(23)
     LEGROM=QNB(24)
     LVGROM=QNB(25)
     QGRF=QNB(26)  ! Wei Chen 2015
#if KEY_OPENMM==1
     lommrxn = qnb(27)  
     lommswi = qnb(28)
#endif
      QETSR=QNB(29)
      LDBEXP=QNB(30)
!
     ! set up distances needed
     CUTNB = RNB(1)
     CTONNB= RNB(2)
     CTOFNB= RNB(3)
     CGONNB= RNB(19)
     CGOFNB= RNB(20)
     ! add variables for GRF -- Wei Chen 2015
     RFCON= RNB(21)
     RFTEMP = RNB(22)
     RFIONI = RNB(23)
     RFKAP = RNB(24)
     RFEPSO = RNB(25)
     !
     WRNMIN= RNB(4)
     WRNMXD= RNB(5)
     E14FAC= RNB(6)
     EPS   = RNB(7)
     NBSCAL= RNB(9)
     IMSCAL= RNB(10)
     !...##IF MMFF
     CTVTRN= RNB(15)
     !...##ENDIF
     !...##IF SOFTVDW
     RGAMIN= RNB(16)
     EGAMINA = RNB(17)
     EGAMINR = RNB(18)
     !...##ENDIF
#if KEY_OPENMM==1
     omm_rxnfld_dielectric = rnb(26)     
#endif
     !
     NBXMOD= INB(1)
  ENDIF
  !
  ! Get integer values not based on parsed options
  NNNB  = INB(2)
  NNNBG = INB(3)
  NNB14 = INB(4)
  NNG14 = INB(5)
  !
  RETURN
END SUBROUTINE GETBN2

SUBROUTINE SETBND(BNBND)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PUTS THE VALUES IN INBND.FCM ON THE DATA STRUCTURE
  !
  !      By Bernard R. Brooks    1983
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use inbnd
  implicit none
  type(nonbondDataStructure) BNBND
  !
  CALL SETBN2(BNBND%LNBOPT,BNBND%NBDIST, &
       BNBND%NBINTS)
  RETURN
END SUBROUTINE SETBND

SUBROUTINE SETBN2(QNB,RNB,INB)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use inbnd
  implicit none
  !
  LOGICAL QNB(*)
  real(chm_real)  RNB(*)
  INTEGER INB(*)
  !
  ! set up logical flags
  QNB(1) = LELEC
  QNB(2) = LVDW
  QNB(3) = LGROUP
  QNB(4) = LCONS
  QNB(5) = LSHFT
  QNB(6) = LVATOM
  QNB(7) = LVSHFT
  QNB(8) = LBYCU
  QNB(9) = LFSWT
  QNB(10)= LVFSWT
#if KEY_IMCUBES==1
  QNB(11) = LBYCBIM                                 
#endif
  QNB(12)= LFMA
  !...##IF MMFF
  QNB(13)= LTRUNC
  QNB(14)= LVTRUNC
  QNB(15)= LMSHFT
  !...##ENDIF
  !...##IF SOFTVDW
  QNB(16)= QGAMIN
  QNB(17)= QELEXP
  QNB(18)= QVDWEXP
  !...##ENDIF
  !rjp..02-FEB-99 BYCC
  QNB(19)= LBYCC
#if KEY_LRVDW==1
  QNB(20) = LLRVDW                                  
#endif
#if KEY_NBIPS==1 /*nbips*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  QNB(21)= LEIPS
  QNB(22)= LVIPS
#endif /* (nbips)*/
  QNB(23)= QETEN
  QNB(24)= LEGROM
  QNB(25)= LVGROM
  QNB(26)= QGRF  ! GRF, Wei Chen 2015
#if KEY_OPENMM==1
  qnb(27) = lommrxn     
  qnb(28) = lommswi
#endif
  QNB(29) = QETSR
  QNB(30) = LDBEXP
!
  ! set up distances needed
  RNB(1) = CUTNB
  RNB(2) = CTONNB
  RNB(3) = CTOFNB
  RNB(19)= CGONNB
  RNB(20)= CGOFNB
  ! add variables for GRF -- Wei Chen 2015
  RNB(21)= RFCON
  RNB(22)= RFTEMP
  RNB(23)= RFIONI
  RNB(24)= RFKAP
  RNB(25)= RFEPSO
  !
  RNB(4) = WRNMIN
  RNB(5) = WRNMXD
  RNB(6) = E14FAC
  RNB(7) = EPS
  RNB(9) = NBSCAL
  RNB(10)= IMSCAL
  !...##IF MMFF
  RNB(15)= CTVTRN
  !...##ENDIF
  !...##IF SOFTVDW
  RNB(16)= RGAMIN
  RNB(17)= EGAMINA
  RNB(18)= EGAMINR
  !...##ENDIF
#if KEY_OPENMM==1
  rnb(26) = omm_rxnfld_dielectric  
#endif
  !
  INB(1) = NBXMOD
  INB(2) = NNNB
  INB(3) = NNNBG
  INB(4) = NNB14
  INB(5) = NNG14
  !
  RETURN
END SUBROUTINE SETBN2

SUBROUTINE NBONDX(X,Y,Z,BNBND, &
     NREFAT,REFAT,XREF,YREF,ZREF)
  !-----------------------------------------------------------------------
  !     Simulates a call to NBONDS with a subset of atoms
  !     REFAT(NREFAT) for which the non-bonded interactions
  !     should be evaluated. The non-bonded cutoff CUTNB is in this
  !     case defined with respect to the reference
  !     point XREF, YREF, ZREF.
  !
  !
  !     18-FEB-83 Axel Brunger
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  !
  use inbnd
  use stream
  use psf
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)

  type(nonbondDataStructure) BNBND

  INTEGER   NREFAT, REFAT(*)
  real(chm_real) XREF,YREF,ZREF
  !
  INTEGER, PARAMETER :: MGRPC=8, MAXIAT=4
  INTEGER   MAXJNB, MXJNBG, NGRPC, IG, JF
  INTEGER   IGPBSC(MGRPC)
  LOGICAL   CMPLTD
  real(chm_real)    RMXGES
  !
  CALL GETBND(BNBND,.TRUE.)
  RMXGES=(CUTNB**3*MAXIAT +1.0)
  IF(RMXGES > 1.0E7) THEN
     MAXJNB=NATOM*MAXIAT
  ELSE
     MAXJNB=RMXGES
  ENDIF
  MAXJNB=MIN(NATOM*MAXIAT,MAXJNB)
  MXJNBG=2*NST2
  !
  CALL RENBND(BNBND,MAXJNB,NATOM,NGRP,MXJNBG)
  !
  !     now we have to make a group list based on REFAT
  !
  !     WE ASSUME THAT THE REFAT LIST IS ORDERED!
  !
  IG=1
  JF=1
  NGRPC=0
  !
  DO WHILE (IG <= NGRP .AND. IGPBS(IG) < REFAT(JF))
     IG=IG+1
  ENDDO
  IG=IG-1
  !
  DO WHILE (IG <= NGRP)
     IF (IGPBS(IG) < REFAT(JF).AND.REFAT(JF) <= IGPBS(IG+1)) THEN
        NGRPC=NGRPC+1
        IF (NGRPC > MGRPC) THEN
           CALL WRNDIE(-4,'<NBONDX>', &
                'Number of temporary groups exceeded')
        ENDIF
        IGPBSC(NGRPC)=IG
     ENDIF
     IG=IG+1
     DO WHILE (IGPBS(IG) >= REFAT(JF) .AND. JF < NREFAT)
        JF=JF+1
     ENDDO
  ENDDO
  !
  CALL NBONX2(NNNB,BNBND%JNB,MAXJNB, &
       BNBND%INBLO,NNNBG, &
       BNBND%JNBG,MXJNBG,BNBND%INBLOG, &
       X,Y,Z,BNBND%INB14,BNBND%IBLO14, &
       CUTNB,CMPLTD,NGRPC, &
       IGPBSC,NREFAT,REFAT,XREF,YREF,ZREF)
  !
  IF (.NOT.CMPLTD) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,2000) MAXJNB,NNNB,MXJNBG,NNNBG
2000 FORMAT(' ERROR FROM NBONDX: MAXJNB or MXJNBG exceeded:',/, &
          ' MAXJNB=',I9,', NNNB=',I9,' and MXJNBG=',I9, &
          ', NNNBG=',I9)
     CALL DIEWRN(-3)
  ENDIF
  CALL SETBND(BNBND)
  RETURN
END SUBROUTINE NBONDX

SUBROUTINE NBONX2(NNNB,JNB,MAXJNB,INBLO,NNNBG,JNBG,MXJNBG, &
     INBLOG,X,Y,Z,INB14,IBLO14, &
     CUTNB,CMPLTD,NGRPIN,IGRPIN, &
     NATIN,IATIN,XREF,YREF,ZREF)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CONSTRUCTS THE NONBONDED LISTS FOR
  !     A SPECIFIED SUBSET OF ATOMS.
  !
  !  NNNB   - number of interactions found
  !  JNB    - pair list found  (MAXJNB)
  !  MAXJNB - maximum allowed size for JNB list
  !  INBLO  - pointer into JNB for each atom (NATOM)
  !  the next four - same as above, except for groups
  !  X,Y,X  - coordinates (NATOM)
  !  INB14  - exclusion pair list (NATOM)
  !  IBLO14 - pointers into INB14 for each atom (NATOM)
  !  CUTNB  - cutoff distance for interactions
  !  CMPLTD - logical flag indicating sufficient space was allowed
  !  NGRPIN  - number of groups in selected group list
  !  IGRPIN  - selected group list for interactions  (NGRPIN)
  !  NATOM - number of selected atoms
  !  IATIN - list of selected atoms
  !
  !    FOR NOW, ONLY ST2-ST2 INTERACTIONS WILL BE PUT ON THE GROUPS LIST
  !
  !      By Bernard R. Brooks    1983
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use chutil,only:initia
  use machutil,only:die
  !
  implicit none
  !
  INTEGER NNNB,MAXJNB
  INTEGER JNB(*)
  INTEGER INBLO(*),NNNBG
  INTEGER JNBG(*)
  INTEGER MXJNBG,INBLOG(*)
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER INB14(*)
  INTEGER IBLO14(*)
  real(chm_real) CUTNB
  LOGICAL CMPLTD
  INTEGER NGRPIN,IGRPIN(*),NATIN,IATIN(*)
  real(chm_real) XREF,YREF,ZREF
  !
  INTEGER   OXYGEN
  INTEGER I,ILAST,IRS,IGR,IS,IQ,II,NXI,NXIMAX
  INTEGER JGR,JS,JQ,J,NXJ,NXJMAX,INBX
  real(chm_real) CTNBSQ,XI,YI,ZI,XD,YD,ZD,R2
  !
  LOGICAL   LEX14,SELAT
  !
  CMPLTD=.FALSE.
  CTNBSQ=CUTNB*CUTNB
  !
  DO I=1,NATOM
     INBLO(I)=0
  ENDDO
  DO I=1,NGRP
     INBLOG(I)=0
  ENDDO
  !
  !
  !     NOW DECIDE HOW TO TREAT EACH RESIDUE PAIR USING A RECTANGULAR
  !     SEARCH AND STORE THE DISPOSITION IN RSDISP
  !
  NNNB=0
  NNNBG=0
  !     CONSTRUCT THE JNB ARRAY SEARCHING ONLY CLOSE ATOMS
  !
  XI=XREF
  YI=YREF
  ZI=ZREF
  ILAST=0
  DO IRS=1,NGRPIN
     IGR=IGRPIN(IRS)
     IF(IGR <= ILAST) CALL DIE
     ILAST=IGR
     IS=IGPBS(IGR)+1
     IQ=IGPBS(IGR+1)
     !
     DO I=IS,IQ
        II=1
        DO WHILE (II <= NATIN.AND.IATIN(II) /= I)
           II=II+1
        ENDDO
        SELAT=(IATIN(II) == I)
        !
        IF(I > 1) THEN
           NXI=IBLO14(I-1)+1
        ELSE
           NXI=1
        ENDIF
        NXIMAX=IBLO14(I)
        DO JGR=1,NGRP
           !
           !
#if KEY_NOST2==0
           IF(IGPTYP(IGR) == 3 .AND. IGPTYP(JGR).EQ.3) THEN
              !
              !     this is an ST2 interaction
              IF(I /= IS) GOTO 56
              IF(IGR == JGR) GOTO 56
              OXYGEN=IGPBS(JGR)+1
              !
              !     if the hydrogens or lone pairs of the ST2 are not yet initialized
              !     forget this interaction
              IF(.NOT.INITIA(OXYGEN+1,X,Y,Z)) GOTO 56
              IF(.NOT.INITIA(OXYGEN+2,X,Y,Z)) GOTO 56
              IF(.NOT.INITIA(OXYGEN+3,X,Y,Z)) GOTO 56
              IF(.NOT.INITIA(OXYGEN+4,X,Y,Z)) GOTO 56
              XD=XI-X(OXYGEN)
              YD=YI-Y(OXYGEN)
              ZD=ZI-Z(OXYGEN)
              R2=XD*XD+YD*YD+ZD*ZD
              IF(R2 < CTNBSQ) THEN
                 !
                 ! add group JGR to the group non-bonded list for group IGR
                 NNNBG=NNNBG+1
                 IF(NNNBG > MXJNBG) RETURN
                 JNBG(NNNBG)=JGR
              ENDIF
              !
           ELSE
#endif 
              !     PROCESS ORDINARY PAIR INTERACTIONS
              !
              !
              IF (SELAT) THEN
                 JS=IGPBS(JGR)+1
                 JQ=IGPBS(JGR+1)
                 DO J=JS,JQ
                    !
                    !       SEE IF PAIR IS IN EXCLUDED LIST
                    !
                    LEX14=.FALSE.
                    IF (J < I) THEN
                       !
                       IF(J > 1) THEN
                          NXJ=IBLO14(J-1)+1
                       ELSE
                          NXJ=1
                       ENDIF
                       NXJMAX=IBLO14(J)
15                     IF(NXJ > NXJMAX) GOTO 40
                       IF(INB14(NXJ) < 0) THEN
                          INBX=-INB14(NXJ)
                          IF(I == INBX) GOTO 39
                          IF(I < INBX) GOTO 40
                       ELSE
                          IF (I == INB14(NXJ)) GOTO 55
                          IF (I < INB14(NXJ)) GOTO 40
                       ENDIF
                       NXJ=NXJ+1
                       GOTO 15
                    ENDIF
                    !
                    IF(I == J) GOTO 55
35                  IF (NXI > NXIMAX) GOTO 40
                    IF(INB14(NXI) < 0) THEN
                       INBX=-INB14(NXI)
                       IF (J == INBX) GOTO 39
                       IF (J < INBX) GOTO 40
                    ELSE
                       IF (J == INB14(NXI)) GOTO 55
                       IF (J < INB14(NXI)) GOTO 40
                    ENDIF
                    NXI=NXI+1
                    GOTO 35
                    !
39                  LEX14=.TRUE.
40                  CONTINUE
                    IF (INITIA(J,X,Y,Z)) THEN
                       XD=XI-X(J)
                       YD=YI-Y(J)
                       ZD=ZI-Z(J)
                       R2=XD*XD+YD*YD+ZD*ZD
                       !
                       IF (R2 < CTNBSQ) THEN
                          !                             add it to the list
                          NNNB=NNNB+1
                          IF (NNNB > MAXJNB) RETURN
                          JNB(NNNB)=J
                          IF (LEX14) JNB(NNNB)=-J
                       ENDIF
                    ENDIF
55                  CONTINUE
                 ENDDO
              ENDIF
#if KEY_NOST2==0
           ENDIF
#endif 
56         CONTINUE
        ENDDO
        INBLO(I)=NNNB
     ENDDO
     INBLOG(IGR)=NNNBG
  ENDDO
  !
  ILAST=0
  DO I=1,NATOM
     IF(INBLO(I) == 0) INBLO(I)=ILAST
     ILAST=INBLO(I)
  ENDDO
  !
  ILAST=0
  DO I=1,NGRP
     IF(INBLOG(I) == 0) INBLOG(I)=ILAST
     ILAST=INBLOG(I)
  ENDDO
  !
  !
  CMPLTD=.TRUE.
  IF(PRNLEV >= 5) WRITE(OUTU,125) NNNB,NNNBG,CUTNB
125 FORMAT(/' NBONDX: ',I9,' atom and',I7,' group', &
       ' interactions within',F6.2,' A.')
  RETURN
END SUBROUTINE NBONX2

SUBROUTINE PRNBND(IUNIT,BNBNDX,BIMAGX)
  !-----------------------------------------------------------------------
  !  THIS ROUTINE PRINTS THE NONBOND ATOM LISTS
  !
  !      By Bernard R. Brooks    11-APR-1984
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use psf
  use image
  use inbnd
  use stream
  !
  use bases_fcm
  use pert
  implicit none
  INTEGER IUNIT     !!,BNBNDX(*),BIMAGX(*)
  type(nonbondDataStructure) BNBNDX
  type(imageDataStructure) BIMAGX

  !
#if KEY_PERT==1
  logical pertcubes

  pertcubes=.false.
#endif 
  !
  CALL GETBND(BNBND,.TRUE.)
  IF(OUTU == IUNIT .AND. PRNLEV < 2) RETURN
  IF(OUTU /= IUNIT .AND. IOLEV < 0) RETURN
  !
  ! Print the atom list
25 FORMAT(/,'PRNBND:',A,/)
  WRITE(IUNIT,25) ' ATOM NONBOND LIST:'
  CALL PRNBD2(IUNIT,NNNB,NATOM, &
       BNBND%INBLO,BNBND%JNB &
#if KEY_IMCUBES==1
       ,lbycbim                                         & 
#endif
       )
  !
  ! Now print group list
  WRITE(IUNIT,25) ' GROUP NONBOND LIST:'
  CALL PRNBD3(IUNIT,NNNBG,NGRP, &
       BNBND%INBLOG,BNBND%JNBG)
  !
#if KEY_PERT==1 /*pertlist*/
  IF(QPERT) THEN
     CALL GETBND(BNBNDR,.TRUE.)
     ! Print the reactant atom list
     WRITE(IUNIT,25) ' ATOM NONBOND LIST REACTANT:'
     CALL PRNBD2(IUNIT,NNNB,NATOM, &
          BNBNDR%INBLO,BNBNDR%JNB &
#if KEY_IMCUBES==1
          ,pertcubes                                       & 
#endif
          )
     !
     ! print the reactant group list
     WRITE(IUNIT,25) ' GROUP NONBOND LIST REACTANT:'
     CALL PRNBD3(IUNIT,NNNBG,NGRP, &
          BNBNDR%INBLOG,BNBNDR%JNBG)
     ! Print the product atom list
     CALL GETBND(BNBNDP,.TRUE.)
     WRITE(IUNIT,25) ' ATOM NONBOND LIST PRODUCT:'
     CALL PRNBD2(IUNIT,NNNB,NATOM, &
          BNBNDP%INBLO,BNBNDP%JNB &
#if KEY_IMCUBES==1
          ,pertcubes                                       & 
#endif
          )
     !
     ! print the product print group list
     WRITE(IUNIT,25) ' GROUP NONBOND LIST PRODUCT:'
     CALL PRNBD3(IUNIT,NNNBG,NGRP, &
          BNBNDP%INBLOG,BNBNDP%JNBG)
     !
     CALL GETBND(BNBND,.TRUE.)
  ENDIF
#endif /* (pertlist)*/
  !
  IF(NTRANS > 0) THEN
     WRITE(IUNIT,25) ' IMAGE ATOM NONBOND LIST:'
     CALL PRNBD2(IUNIT,bimag%NIMNB,NATIM, &
          bimag%IMBLO,bimag%IMJNB &
#if KEY_IMCUBES==1
          ,lbycbim                                       & 
#endif
          )
     !
     WRITE(IUNIT,25) ' IMAGE SELF TERM ATOM LIST:'
     CALL PRNBD2(IUNIT,bimag%NIMNBS,NATIM, &
          bimag%IMBLOS,bimag%IMJNBS &
#if KEY_IMCUBES==1
          ,lbycbim                                       & 
#endif
          )
     !
     ! Now print group list
     WRITE(IUNIT,25) ' IMAGE ATOM GROUP NONBOND LIST:'
     CALL PRNBD3(IUNIT,bimag%NIMNBG,NGRPT, &
          bimag%IMBLOG,bimag%IMJNBG)
     !
     WRITE(IUNIT,25) ' IMAGE SELF TERM GROUP LIST:'
     CALL PRNBD3(IUNIT,bimag%NIMNBX,NGRPT, &
          bimag%IMBLOX,bimag%IMJNBX)
     !
#if KEY_PERT==1 /*pertimlist*/
     IF(QPERT) THEN
        WRITE(IUNIT,25) ' IMAGE REACTANT ATOM NONBOND LIST:'
        CALL PRNBD2(IUNIT,bimagr%NIMNB,NATIM, &
             bimagr%IMBLO,bimagr%IMJNB &
#if KEY_IMCUBES==1
             ,pertcubes                                       & 
#endif
             )
        !
        WRITE(IUNIT,25) ' IMAGE REACTANT SELF TERM ATOM LIST:'
        CALL PRNBD2(IUNIT,bimagr%NIMNBS,NATIM, &
             bimagr%IMBLOS,bimagr%IMJNBS &
#if KEY_IMCUBES==1
             ,pertcubes                                       & 
#endif
             )
        !
        ! Now print group list
        WRITE(IUNIT,25) ' IMAGE REACTANT ATOM GROUP NONBOND LIST:'
        CALL PRNBD3(IUNIT,bimagr%NIMNBG,NGRPT, &
             bimagr%IMBLOG,bimagr%IMJNBG)
        !
        WRITE(IUNIT,25) ' IMAGE REACTANT SELF TERM GROUP LIST:'
        CALL PRNBD3(IUNIT,bimagr%NIMNBX,NGRPT, &
             bimagr%IMBLOX,bimagr%IMJNBX)
        !
        WRITE(IUNIT,25) ' IMAGE PRODUCT ATOM NONBOND LIST:'
        CALL PRNBD2(IUNIT,bimagp%NIMNB,NATIM, &
             bimagp%IMBLO,bimagp%IMJNB &
#if KEY_IMCUBES==1
             ,pertcubes                                       & 
#endif
             )
        !
        WRITE(IUNIT,25) ' IMAGE PRODUCT SELF TERM ATOM LIST:'
        CALL PRNBD2(IUNIT,bimagp%NIMNBS,NATIM, &
             bimagp%IMBLOS,bimagp%IMJNBS &
#if KEY_IMCUBES==1
             ,pertcubes                                       & 
#endif
             )
        !
        ! Now print group list
        WRITE(IUNIT,25) ' IMAGE PRODUCT ATOM GROUP NONBOND LIST:'
        CALL PRNBD3(IUNIT,bimagp%NIMNBG,NGRPT, &
             bimagp%IMBLOG,bimagp%IMJNBG)
        !
        WRITE(IUNIT,25) ' IMAGE PRODUCT SELF TERM GROUP LIST:'
        CALL PRNBD3(IUNIT,bimagp%NIMNBX,NGRPT, &
             bimagp%IMBLOX,bimagp%IMJNBX)
     ENDIF
#endif /* (pertimlist)*/
  ENDIF
  RETURN
END SUBROUTINE PRNBND

SUBROUTINE PRNBD2(IUNIT,NNNB,NATOM,INBLO,JNB &
#if KEY_IMCUBES==1
     ,lbycbim                                       & 
#endif
     )
  !-----------------------------------------------------------------------
  !  This routine prints an atom based nonbond list.
  !
  use chm_kinds
  use stream
  use chutil,only:atomid
  implicit none
  !
  INTEGER IUNIT,NNNB,NATOM,INBLO(*)
  INTEGER JNB(*)
  !
  INTEGER ITEMP,NB,I,J,NPR,JPR
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  CHARACTER(len=4) L14
#if KEY_IMCUBES==1
  logical lbycbim                                    
#endif
  !
#if KEY_DEBUG==1
  IF(PRNLEV > 7) THEN
     WRITE(OUTU,433) 'Nonbond interactions'
     ITEMP=0
     DO I=1,NATOM
#if KEY_IMCUBES==1
        if(lbycbim)itemp=inblo(i+natom)              
#endif
        NPR= INBLO(I) - ITEMP
        WRITE(OUTU,434) I, NPR
        WRITE(OUTU,435) (JNB(J),J=ITEMP+1,INBLO(I))
        ITEMP = INBLO(I)
     enddo
433  FORMAT(' PRNBND: ',A)
434  FORMAT('   ATOM: ',I6,' has',I6,' interactions with ...')
435  FORMAT(15I6)
  ENDIF
#endif 
  !
  IF(NNNB > 20000) THEN
     WRITE(IUNIT,20) NNNB
20   FORMAT(I10,' NONBOND INTERACTIONS IS TOO MANY TO PRINT')
     CALL WRNDIE(-2,'<PRNBND>','TOO MANY INTERACTIONS FOR PRINTING')
  ENDIF
  !
  WRITE(IUNIT,25) NNNB
25 FORMAT('PRNBD2: Number of elements on this list is:',I10)
  !
  IF(NNNB <= 0) RETURN
  !
  NB=0
  ITEMP=0
  DO I=1,NATOM
#if KEY_IMCUBES==1
     if(lbycbim)itemp=inblo(i+natom)
     nb=itemp
#endif 
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF (NPR /= 0) THEN
        WRITE(IUNIT,35) I,NPR
35      FORMAT('PRNBND: Number of pairs for atom',I6,' is:',I10)
     ENDIF
     IF (NPR > 0 .AND.NNNB.GT.0) THEN
        CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
        !
        DO JPR=1,NPR
           NB=NB+1
           J=JNB(NB)
           L14='    '
           IF (J < 0) THEN
              L14='1-4 '
              J=-J
           ENDIF
           IF(J > 0) THEN
              CALL ATOMID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
              WRITE(IUNIT,45) NB,I,SIDDN(1:idleng), &
                   RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng), &
                   J,SIDDN2(1:idleng),RIDDN2(1:idleng), &
                   RESDN2(1:idleng),ACDN2(1:idleng),L14
45            FORMAT(2I5,4(1X,A),' AND',I5,4(1X,A),3X,A)
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE PRNBD2

SUBROUTINE PRNBD3(IUNIT,NNNB,NGRP,INBLO,JNB)
  !-----------------------------------------------------------------------
  !  This routine prints a group based nonbond list.
  !
  use chm_kinds
  use stream
  use chutil,only:groupid
  implicit none
  !
  INTEGER IUNIT,NNNB,NGRP,INBLO(*)
  INTEGER JNB(*)
  !
  INTEGER ITEMP,NB,I,J,NPR,JPR
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  CHARACTER(len=4) L14
  !
  !
  IF(NNNB > 20000) THEN
     WRITE(IUNIT,20) NNNB
20   FORMAT(I10,' NONBOND INTERACTIONS IS TOO MANY TO PRINT')
     CALL WRNDIE(-2,'<PRNBND>','TOO MANY INTERACTIONS FOR PRINTING')
  ENDIF
  !
  WRITE(IUNIT,25) NNNB
25 FORMAT('PRNBD3: Number of elements on this list is:',I10)
  !
  IF(NNNB <= 0) RETURN
  !
  NB=0
  ITEMP=0
  DO I=1,NGRP
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF (NPR /= 0) THEN
        WRITE(IUNIT,35) I,NPR
35      FORMAT('PRNBD3: Number of pairs for group',I6,' is:',I10)
     ENDIF
     IF (NPR > 0) THEN
        CALL GROUPID(I,SIDDN,RIDDN,RESDN,ACDN)
        !
        DO JPR=1,NPR
           NB=NB+1
           J=JNB(NB)
           L14='    '
           IF (J < 0) THEN
              L14='1-4 '
              J=-J
           ENDIF
           IF(J > 0) THEN
              CALL GROUPID(J,SIDDN2,RIDDN2,RESDN2,ACDN2)
              WRITE(IUNIT,45) NB,I,SIDDN(1:idleng), &
                   RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng), &
                   J,SIDDN2(1:idleng),RIDDN2(1:idleng), &
                   RESDN2(1:idleng),ACDN2(1:idleng),L14
45            FORMAT(5X,2I5,4(1X,A),' AND',I5,4(1X,A),3X,A)
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE PRNBD3

SUBROUTINE ALLEXT
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES SPACE IN THE FOR THE EXTENDED
  !     ELECTROSTATICS
  !
  !sb March 2007 PERT specific stuff moved here from pert.src plus some
  !   supporting logic
  !
  use chm_kinds
  use dimens_fcm
  use exelecm
  use psf
  use stream
  use pert  ! NKB, extelec with pert
  use memory
  implicit none
  !
  LOGICAL QEXTN2,INITIAL,REINIT
  INTEGER  :: IFIRST=0,OLDNATM,OLDNGRP,SPACE,OSPACE  ! NKB, extelec with pert
  SAVE     IFIRST,OLDNATM,OLDNGRP,QEXTN2        ! NKB, extelec with pert
#if KEY_PERT==1
  integer :: ipfirst=0
  save ipfirst
#endif 
  !
  !  TEST CONDITIONS FOR SPACE ALLOCATION , THEN SET FLAGS
  !  If Atoms have been added, then space may have to be
  !  reallocated to accomodate the new system size.  The next few lines
  !  take care of this in a simple way.
  INITIAL=.FALSE.
  IF(IFIRST == 0) INITIAL=.TRUE.
  IFIRST=1
  REINIT=.FALSE.
  IF((OLDNATM /= NATOM).AND.(.NOT. INITIAL)) REINIT=.TRUE.
  IF(QEXTN2.NEQV.QEXTND) REINIT=.TRUE.
#if KEY_PERT==1
  IF(QPERT.AND.(IPFIRST == 0)) REINIT=.TRUE.
#endif 
  !
  IF(REINIT) THEN
     ! free-old--space
     !  Free whatever space may have bee already allocated
     IF(WRNLEV >= 2) WRITE(OUTU,600)
600  FORMAT(2X,'WARNING: NBOND CONFIGURATION HAS BEEN ', &
          'CHANGED SINCE LAST UPDATE',/)
     !
     IF(QEXTN2) THEN
        call chmdealloc('nbutil.src','ALLEXT','ATPOT',OLDNATM,crl=ATPOT)
        call chmdealloc('nbutil.src','ALLEXT','ATFX',OLDNATM,crl=ATFX)
        call chmdealloc('nbutil.src','ALLEXT','ATFY',OLDNATM,crl=ATFY)
        call chmdealloc('nbutil.src','ALLEXT','ATFZ',OLDNATM,crl=ATFZ)
        call chmdealloc('nbutil.src','ALLEXT','ATGXX',OLDNATM,crl=ATGXX)
        call chmdealloc('nbutil.src','ALLEXT','ATGYY',OLDNATM,crl=ATGYY)
        call chmdealloc('nbutil.src','ALLEXT','ATGZZ',OLDNATM,crl=ATGZZ)
        call chmdealloc('nbutil.src','ALLEXT','ATGXY',OLDNATM,crl=ATGXY)
        call chmdealloc('nbutil.src','ALLEXT','ATGYZ',OLDNATM,crl=ATGYZ)
        call chmdealloc('nbutil.src','ALLEXT','ATGZX',OLDNATM,crl=ATGZX)
#if KEY_PERT==1
        ! --------------------------------------------------------------------
        ! added by NKB and Wonpil Im for extended electrostatics implementation
        ! in PERT, April 2001
        !sb070507 fix IF(QPERT) THEN
        IF((QPERT).and.(ipfirst > 0)) THEN
           call chmdealloc('nbutil.src','ALLEXT','ATPOT0',OLDNATM,crl=ATPOT0)
           call chmdealloc('nbutil.src','ALLEXT','ATFX0',OLDNATM,crl=ATFX0)
           call chmdealloc('nbutil.src','ALLEXT','ATFY0',OLDNATM,crl=ATFY0)
           call chmdealloc('nbutil.src','ALLEXT','ATFZ0',OLDNATM,crl=ATFZ0)
           call chmdealloc('nbutil.src','ALLEXT','ATGXX0',OLDNATM,crl=ATGXX0)
           call chmdealloc('nbutil.src','ALLEXT','ATGYY0',OLDNATM,crl=ATGYY0)
           call chmdealloc('nbutil.src','ALLEXT','ATGZZ0',OLDNATM,crl=ATGZZ0)
           call chmdealloc('nbutil.src','ALLEXT','ATGXY0',OLDNATM,crl=ATGXY0)
           call chmdealloc('nbutil.src','ALLEXT','ATGYZ0',OLDNATM,crl=ATGYZ0)
           call chmdealloc('nbutil.src','ALLEXT','ATGZX0',OLDNATM,crl=ATGZX0)
           call chmdealloc('nbutil.src','ALLEXT','RSPOT0',OLDNGRP,crl=RSPOT0)
           call chmdealloc('nbutil.src','ALLEXT','RSFX0',OLDNGRP,crl=RSFX0)
           call chmdealloc('nbutil.src','ALLEXT','RSFY0',OLDNGRP,crl=RSFY0)
           call chmdealloc('nbutil.src','ALLEXT','RSFZ0',OLDNGRP,crl=RSFZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSGXX0',OLDNGRP,crl=RSGXX0)
           call chmdealloc('nbutil.src','ALLEXT','RSGYY0',OLDNGRP,crl=RSGYY0)
           call chmdealloc('nbutil.src','ALLEXT','RSGZZ0',OLDNGRP,crl=RSGZZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSGXY0',OLDNGRP,crl=RSGXY0)
           call chmdealloc('nbutil.src','ALLEXT','RSGYZ0',OLDNGRP,crl=RSGYZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSGZX0',OLDNGRP,crl=RSGZX0)

           call chmdealloc('nbutil.src','ALLEXT','RSQ0',OLDNGRP,crl=RSQ0)
           call chmdealloc('nbutil.src','ALLEXT','RSDX0',OLDNGRP,crl=RSDX0)
           call chmdealloc('nbutil.src','ALLEXT','RSDY0',OLDNGRP,crl=RSDY0)
           call chmdealloc('nbutil.src','ALLEXT','RSDZ0',OLDNGRP,crl=RSDZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSQXX0',OLDNGRP,crl=RSQXX0)
           call chmdealloc('nbutil.src','ALLEXT','RSQYY0',OLDNGRP,crl=RSQYY0)
           call chmdealloc('nbutil.src','ALLEXT','RSQZZ0',OLDNGRP,crl=RSQZZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSQXY0',OLDNGRP,crl=RSQXY0)
           call chmdealloc('nbutil.src','ALLEXT','RSQYZ0',OLDNGRP,crl=RSQYZ0)
           call chmdealloc('nbutil.src','ALLEXT','RSQZX0',OLDNGRP,crl=RSQZX0)
        ENDIF
        ! --- end of addition by NKB, Wonpil Im -------------------------------
#endif 
     ENDIF
  ENDIF
  !
  IF(INITIAL.OR.REINIT) THEN
     ! allocate-space-for-extended
     !  Now, allocate new space on the
     !
     IF(QEXTND) THEN
        call chmalloc('nbutil.src','ALLEXT','ATPOT',NATOM,crl=ATPOT)
        call chmalloc('nbutil.src','ALLEXT','ATFX',NATOM,crl=ATFX)
        call chmalloc('nbutil.src','ALLEXT','ATFY',NATOM,crl=ATFY)
        call chmalloc('nbutil.src','ALLEXT','ATFZ',NATOM,crl=ATFZ)
        call chmalloc('nbutil.src','ALLEXT','ATGXX',NATOM,crl=ATGXX)
        call chmalloc('nbutil.src','ALLEXT','ATGYY',NATOM,crl=ATGYY)
        call chmalloc('nbutil.src','ALLEXT','ATGZZ',NATOM,crl=ATGZZ)
        call chmalloc('nbutil.src','ALLEXT','ATGXY',NATOM,crl=ATGXY)
        call chmalloc('nbutil.src','ALLEXT','ATGYZ',NATOM,crl=ATGYZ)
        call chmalloc('nbutil.src','ALLEXT','ATGZX',NATOM,crl=ATGZX)
#if KEY_PERT==1
        !sb this is essentially the stuff moved from pert.src
        IF(QPERT) THEN
           call chmalloc('nbutil.src','ALLEXT','ATPOT0',NATOM,crl=ATPOT0)
           call chmalloc('nbutil.src','ALLEXT','ATFX0',NATOM,crl=ATFX0)
           call chmalloc('nbutil.src','ALLEXT','ATFY0',NATOM,crl=ATFY0)
           call chmalloc('nbutil.src','ALLEXT','ATFZ0',NATOM,crl=ATFZ0)
           call chmalloc('nbutil.src','ALLEXT','RSPOT0',NGRP,crl=RSPOT0)
           call chmalloc('nbutil.src','ALLEXT','RSFX0',NGRP,crl=RSFX0)
           call chmalloc('nbutil.src','ALLEXT','RSFY0',NGRP,crl=RSFY0)
           call chmalloc('nbutil.src','ALLEXT','RSFZ0',NGRP,crl=RSFZ0)
           call chmalloc('nbutil.src','ALLEXT','RSQ0',NGRP,crl=RSQ0)
           call chmalloc('nbutil.src','ALLEXT','RSDX0',NGRP,crl=RSDX0)
           call chmalloc('nbutil.src','ALLEXT','RSDY0',NGRP,crl=RSDY0)
           call chmalloc('nbutil.src','ALLEXT','RSDZ0',NGRP,crl=RSDZ0)
           call chmalloc('nbutil.src','ALLEXT','ATGXX0',NATOM,crl=ATGXX0)
           call chmalloc('nbutil.src','ALLEXT','ATGYY0',NATOM,crl=ATGYY0)
           call chmalloc('nbutil.src','ALLEXT','ATGZZ0',NATOM,crl=ATGZZ0)
           call chmalloc('nbutil.src','ALLEXT','ATGXY0',NATOM,crl=ATGXY0)
           call chmalloc('nbutil.src','ALLEXT','ATGYZ0',NATOM,crl=ATGYZ0)
           call chmalloc('nbutil.src','ALLEXT','ATGZX0',NATOM,crl=ATGZX0)
           call chmalloc('nbutil.src','ALLEXT','RSQXX0',NGRP,crl=RSQXX0)
           call chmalloc('nbutil.src','ALLEXT','RSQYY0',NGRP,crl=RSQYY0)
           call chmalloc('nbutil.src','ALLEXT','RSQZZ0',NGRP,crl=RSQZZ0)
           call chmalloc('nbutil.src','ALLEXT','RSQXY0',NGRP,crl=RSQXY0)
           call chmalloc('nbutil.src','ALLEXT','RSQYZ0',NGRP,crl=RSQYZ0)
           call chmalloc('nbutil.src','ALLEXT','RSQZX0',NGRP,crl=RSQZX0)
           call chmalloc('nbutil.src','ALLEXT','RSGXX0',NGRP,crl=RSGXX0)
           call chmalloc('nbutil.src','ALLEXT','RSGYY0',NGRP,crl=RSGYY0)
           call chmalloc('nbutil.src','ALLEXT','RSGZZ0',NGRP,crl=RSGZZ0)
           call chmalloc('nbutil.src','ALLEXT','RSGXY0',NGRP,crl=RSGXY0)
           call chmalloc('nbutil.src','ALLEXT','RSGYZ0',NGRP,crl=RSGYZ0)
           call chmalloc('nbutil.src','ALLEXT','RSGZX0',NGRP,crl=RSGZX0)
           IF(IPFIRST == 0) IPFIRST=1
        ENDIF
#endif 
     ENDIF
     ! -------- NKB -------------------------
     OLDNGRP=NGRP
     ! --------------------------------------
     OLDNATM=NATOM
     QEXTN2=QEXTND
  ENDIF
  !
  ! ALLOCATE SPACE FOR REACTION FIELD STUFF
  IF(INITIAL) siz_LEGPN=0
  IF(QRXNFL) THEN
     SPACE=RXNORD*RXNORD+2*RXNORD+1
     OSPACE=siz_LEGPN
     IF(SPACE /= OSPACE) THEN
        IF(OSPACE > 0) THEN
           call chmdealloc('nbutil.src','ALLEXT','LEGPN',OSPACE,crl=LEGPN)
           call chmdealloc('nbutil.src','ALLEXT','LEGPND',OSPACE,crl=LEGPND)
           call chmdealloc('nbutil.src','ALLEXT','ERXMNT',OSPACE,crl=ERXMNT)
           call chmdealloc('nbutil.src','ALLEXT','MPMMNT',OSPACE,cmpx=MPMMNT)
           call chmdealloc('nbutil.src','ALLEXT','RXMMNT',OSPACE,cmpx=RXMMNT)
        ENDIF
        siz_LEGPN =SPACE
        call chmalloc('nbutil.src','ALLEXT','LEGPN',SPACE,crl=LEGPN)
        call chmalloc('nbutil.src','ALLEXT','LEGPND',SPACE,crl=LEGPND)
        call chmalloc('nbutil.src','ALLEXT','ERXMNT',SPACE,crl=ERXMNT)
        call chmalloc('nbutil.src','ALLEXT','MPMMNT',SPACE,cmpx=MPMMNT)
        call chmalloc('nbutil.src','ALLEXT','RXMMNT',SPACE,cmpx=RXMMNT)
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE ALLEXT

LOGICAL FUNCTION QINLIST(NAME,LIST,N)
  !-----------------------------------------------------------------------
  ! RCZ 91/06/26 : TRUE IF NAME IS IN LIST, FALSE OTHERWISE
  !
  ! This routine is used to support the extra exclusions method (excl.f90)
  !
  !
  use chm_kinds
  implicit none
  INTEGER     N,I
  CHARACTER(len=*) NAME,LIST(N)
  !
  DO I=1,N
     IF(NAME == LIST(I)) THEN
        QINLIST=.TRUE.
        RETURN
     ENDIF
  enddo
  QINLIST=.FALSE.
  RETURN
END FUNCTION QINLIST

!end module nbutil_module

