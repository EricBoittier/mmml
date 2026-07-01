module travelmain
  ! TRAVEL file-names facilitate compilation with previous versions of CHARMM.

  !******************************************************************************
  !                                                                             *
  !           TReK: a program for Trajectory REfinement and Kinematics.         *
  !                                                                             *
  !                        Version 2.10 , July  5-2003.                         *
  !                                                                             *
  !******************************************************************************
  !         Please report problems or send suggestions to the author:           *
  !                                                                             *
  !                              Stefan Fischer                                 *
  !                         Tel. (49)6221-548879                                *
  !                e-mail: stefan.fischer@iwr.uni-heidelberg.de                 *
  !                                                                             *
  !            Check for application examples and bug-fixes under :             *
  !                                                                             *
  !            http://www.iwr.uni-heidelberg.de/iwr/biocomp/fischer             *
  !                                                                             *
  !******************************************************************************

  use travelsub
  use travelsub2

contains

  SUBROUTINE TREK(COMLYN,COMLEN)


    use chm_kinds
    use chm_types
    use dimens_fcm
    use psf
    use image
    use number
    use select
    use stream
    use string
    use coord
    use coordc
    use travel
    use memory
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    real(chm_real),allocatable,dimension(:) :: XREF
    real(chm_real),allocatable,dimension(:) :: YREF
    real(chm_real),allocatable,dimension(:) :: ZREF
    real(chm_real),allocatable,dimension(:) :: PTENE
    real(chm_real),allocatable,dimension(:) :: LENSEG
    real(chm_real),allocatable,dimension(:) :: SEGSTP
    real(chm_real),allocatable,dimension(:) :: SRTENE
    real(chm_real),allocatable,dimension(:) :: ENEMAX
    real(chm_real),allocatable,dimension(:) :: PTGRAD
    real(chm_real),allocatable,dimension(:) :: DYNBRA
    integer,allocatable,dimension(:) :: OLDMOV
    integer,allocatable,dimension(:) :: SERIAL
    integer,allocatable,dimension(:) :: NLINMN
    integer,allocatable,dimension(:) :: NEIBR1
    integer,allocatable,dimension(:) :: NEIBR2
    integer,allocatable,dimension(:) :: IDPREC
    integer,allocatable,dimension(:) :: IDNEXT
    integer,allocatable,dimension(:) :: STPMAX
    integer,allocatable,dimension(:) :: NEXMIN
    integer,allocatable,dimension(:) :: PREMIN
    integer,allocatable,dimension(:) :: SRTIDX
    integer,allocatable,dimension(:) :: IFREEP
    type(chm_array),allocatable,dimension(:) :: XBAS,YBAS,ZBAS
    logical,allocatable,dimension(:) :: QUP
    logical,allocatable,dimension(:) :: QDOWN
    real(chm_real),allocatable,dimension(:) :: PRNORM
    integer,allocatable,dimension(:) :: FPSLCT
    real(chm_real),allocatable,dimension(:) :: MWGHT
    real(chm_real),allocatable,dimension(:) :: IDUM1
    integer,allocatable,dimension(:) :: ITEMP
    real(chm_real),allocatable,dimension(:) :: DTEMP
    real(chm_real),allocatable,dimension(:) :: IDUMX
    real(chm_real),allocatable,dimension(:) :: IDUMY
    real(chm_real),allocatable,dimension(:) :: IDUMZ
    real(chm_real),allocatable,dimension(:) :: XS0
    real(chm_real),allocatable,dimension(:) :: YS0
    real(chm_real),allocatable,dimension(:) :: ZS0
    real(chm_real),allocatable,dimension(:) :: XKEEP
    real(chm_real),allocatable,dimension(:) :: YKEEP
    real(chm_real),allocatable,dimension(:) :: ZKEEP
    integer,allocatable,dimension(:) :: ISLCT
    integer,allocatable,dimension(:) :: FREEAT
    real(chm_real4),allocatable,dimension(:) :: RDTMP
    integer,allocatable,dimension(:) :: SLCT
    CHARACTER(len=*) COMLYN
    INTEGER       COMLEN

#if KEY_TRAVEL==0 /*travel_main*/
    CALL WRNDIE(-1,'<TREK>','TReK code is not compiled.')
    RETURN
  end SUBROUTINE TREK
#else /* (travel_main)*/

  INTEGER      I,OLDLST,OLDLEV


  LOGICAL      IMGROT, LSCM,LCPR, LREFER
#if KEY_HFB==1
  !ivk
  LOGICAL      LLIP,LFP,LMWGHT
  !ivk  LLIP - line integral path
  !ivk  LFP - Fourier path
#endif 
  INTEGER      IMGAXI,NFIXED
  INTEGER      NMAXP,NPOINT,NFREEP,IDXPRO,IDXSAD,NEIBOR,TOTUSD,NMAXI

  SAVE         IMGROT, LSCM,LCPR, LREFER,IMGAXI,NFIXED
  SAVE         NMAXP,NPOINT,NFREEP,IDXPRO,IDXSAD,NEIBOR,TOTUSD,NMAXI

  SAVE         NEIBR2,QUP,QDOWN,NLINMN,IFREEP
  SAVE         IDNEXT, PTENE, PTGRAD,LENSEG, STPMAX,ENEMAX
  SAVE         NEXMIN,PREMIN, SRTENE,SRTIDX,SERIAL
  SAVE         PRNORM,DYNBRA,XREF,YREF,ZREF,SEGSTP

#if KEY_HFB==1
  INTEGER      NSELR     /*ivk*/
#endif
  ! first trip into trek, allocate space in travel_ltm. cb3
  if(.not.allocated(xscal)) call allocate_travel_ltm(natom)

  if(nmaxp > 0 ) then
     call chmalloc('adiab.src','TREK','ITEMP',NMAXP,intg=ITEMP)
     call chmalloc('adiab.src','TREK','DTEMP',NMAXP,crl=DTEMP)
  else
     call chmalloc('adiab.src','TREK','ITEMP',1,intg=ITEMP)
     call chmalloc('adiab.src','TREK','DTEMP',1,crl=DTEMP)
  endif
  call chmalloc('adiab.src','TREK','IDUMX',NATOM,crl=IDUMX)
  call chmalloc('adiab.src','TREK','IDUMY',NATOM,crl=IDUMY)
  call chmalloc('adiab.src','TREK','IDUMZ',NATOM,crl=IDUMZ)
  call chmalloc('adiab.src','TREK','XS0',NATOM,crl=XS0)
  call chmalloc('adiab.src','TREK','YS0',NATOM,crl=YS0)
  call chmalloc('adiab.src','TREK','ZS0',NATOM,crl=ZS0)
  call chmalloc('adiab.src','TREK','XKEEP',NATOM,crl=XKEEP)
  call chmalloc('adiab.src','TREK','YKEEP',NATOM,crl=YKEEP)
  call chmalloc('adiab.src','TREK','ZKEEP',NATOM,crl=ZKEEP)
  call chmalloc('adiab.src','TREK','ISLCT',NATOM,intg=ISLCT)

  CALL INII2(ISLCT,NATOM, 1 )

  call chmalloc('adiab.src','TREK','FREEAT',NATOM,intg=FREEAT)
  call chmalloc('adiab.src','TREK','RDTMP',NATOM,cr4=RDTMP)

  OLDLEV = PRNLEV
#if KEY_PARALLEL==1
  IF (MYNOD == 0) PRNLEV = 3
#else /**/
  PRNLEV = 3
#endif 

  IF (.NOT. QTRAV) THEN

     CALL W0(' ')
     CALL W0( '**************************************'// &
          '****************************************')
     CALL W0( '*          TReK (Trajectory REfinement &'// &
          ' Kinematics) version 2.10            *' )
     CALL W0( '* Reference : S.Fischer & M.Karplus, '// &
          'Chem. Phys. Letters vol.194, p.252, 1992*')
     CALL W0( '**************************************'// &
          '****************************************')

     NMAXP  = GTRMI(COMLYN,COMLEN,'MAXP',100)
     CALL FIXED(NFIXED)
     LSCM   = .FALSE.
     LCPR   = .FALSE.
#if KEY_HFB==1
     !ivk
     LLIP   = .FALSE.
     LFP    = .FALSE.
     !ivk
#endif 
     LREFER = .FALSE.
     NPOINT = 0
     NFREEP = 0
     TOTUSD = 0

     LSCAL = (INDXA(COMLYN,COMLEN,'SCAL')  >  0)
     IF (LSCAL) THEN
        IF (INDXA(COMLYN,COMLEN,'COMP')  >  0) THEN
           IF (INDXA(COMLYN,COMLEN,'WEIG')  >  0) THEN
              DO  I = 1, NATOM
                 XSCAL(I) = WCOMP(I)
                 YSCAL(I) = WCOMP(I)
                 ZSCAL(I) = WCOMP(I)
              ENDDO
           ELSE
              DO  I = 1, NATOM
                 XSCAL(I) = XCOMP(I)
                 YSCAL(I) = YCOMP(I)
                 ZSCAL(I) = ZCOMP(I)
              ENDDO
           ENDIF
        ELSE
           IF (INDXA(COMLYN,COMLEN,'WEIG')  >  0) THEN
              DO I = 1, NATOM
                 XSCAL(I) = WMAIN(I)
                 YSCAL(I) = WMAIN(I)
                 ZSCAL(I) = WMAIN(I)
              ENDDO
           ELSE
              DO I = 1, NATOM
                 XSCAL(I) = X(I)
                 YSCAL(I) = Y(I)
                 ZSCAL(I) = Z(I)
              ENDDO
           ENDIF
        ENDIF

        DO I = 1, NATOM
           IF (XSCAL(I) < ONE)  LSCAL = .FALSE.
           IF (YSCAL(I) < ONE)  LSCAL = .FALSE.
           IF (ZSCAL(I) < ONE)  LSCAL = .FALSE.
        ENDDO
        IF (.NOT.LSCAL) THEN
           CALL W0(' ')
           CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
           CALL W0('  Coord.-scaling constants must be >= 1 .')
           CALL WRNDIE(-4,'<TREK>','Coordinate-scaling ignored.')
        ENDIF
        DO I = 1, NATOM
           XSCAL(I) = ONE/XSCAL(I)
           YSCAL(I) = ONE/YSCAL(I)
           ZSCAL(I) = ONE/ZSCAL(I)
        ENDDO
     ENDIF

     IMGAXI = 0
     IF     (INDXA(COMLYN,COMLEN,'XAXI')  >  0) THEN
        IMGAXI = 1
     ELSEIF (INDXA(COMLYN,COMLEN,'YAXI')  >  0) THEN
        IMGAXI = 2
     ELSEIF (INDXA(COMLYN,COMLEN,'ZAXI')  >  0) THEN
        IMGAXI = 3
     ENDIF

     IMGROT = .FALSE.
     IF (INDXA(COMLYN,COMLEN,'ROTA')  >  0) THEN
        IF (IMGAXI  >  0) THEN
           IMGROT = .TRUE.
        ELSE
           CALL WRNDIE(-2,'<TREK>', &
                'ROTA keyword requires an axis keyword. Ignored !')
        ENDIF
     ENDIF

     IF (NTRANS == 0 .AND. IMGAXI > 0) THEN
        IMGAXI = 0
        CALL WRNDIE(1,'<TREK>', &
             'There are no images: axis keywords ignored !')
     ENDIF
     IF (NTRANS == 1.AND.IMGAXI == 0)  CALL WRNDIE(-1,'<TREK>', &
          'Only 1 transformation, but no axis has been declared !')
     IF (NTRANS > 1.AND.IMGAXI > 0)  CALL WRNDIE(-2,'<TREK>', &
          'More than 1 transformation, but reorienting around axis !')

     call chmalloc('adiab.src','TREK','XREF',NATOM,crl=XREF)
     call chmalloc('adiab.src','TREK','YREF',NATOM,crl=YREF)
     call chmalloc('adiab.src','TREK','ZREF',NATOM,crl=ZREF)
     call chmalloc('adiab.src','TREK','PTENE',NMAXP,crl=PTENE)
     call chmalloc('adiab.src','TREK','LENSEG',NMAXP,crl=LENSEG)
     call chmalloc('adiab.src','TREK','SEGSTP',NMAXP,crl=SEGSTP)
     call chmalloc('adiab.src','TREK','SRTENE',NMAXP,crl=SRTENE)
     call chmalloc('adiab.src','TREK','ENEMAX',NMAXP,crl=ENEMAX)
     call chmalloc('adiab.src','TREK','PTGRAD',NMAXP,crl=PTGRAD)
     call chmalloc('adiab.src','TREK','DYNBRA',NMAXP,crl=DYNBRA)
     call chmalloc('adiab.src','TREK','OLDMOV',NATOM,intg=OLDMOV)
     call chmalloc('adiab.src','TREK','SERIAL',NMAXP,intg=SERIAL)
     call chmalloc('adiab.src','TREK','NLINMN',NMAXP,intg=NLINMN)
     call chmalloc('adiab.src','TREK','NEIBR1',2*NMAXP,intg=NEIBR1)
     call chmalloc('adiab.src','TREK','NEIBR2',2*NMAXP,intg=NEIBR2)
     call chmalloc('adiab.src','TREK','IDPREC',NMAXP,intg=IDPREC)
     call chmalloc('adiab.src','TREK','IDNEXT',NMAXP,intg=IDNEXT)
     call chmalloc('adiab.src','TREK','STPMAX',NMAXP,intg=STPMAX)
     call chmalloc('adiab.src','TREK','NEXMIN',NMAXP,intg=NEXMIN)
     call chmalloc('adiab.src','TREK','PREMIN',NMAXP,intg=PREMIN)
     call chmalloc('adiab.src','TREK','SRTIDX',NMAXP,intg=SRTIDX)
     call chmalloc('adiab.src','TREK','IFREEP',NMAXP,intg=IFREEP)
     call chmalloc_chm_array('adiab.src','TREK','XBAS',NMAXP,XBAS)
     call chmalloc_chm_array('adiab.src','TREK','YBAS',NMAXP,YBAS)
     call chmalloc_chm_array('adiab.src','TREK','ZBAS',NMAXP,ZBAS)
     call chmalloc('adiab.src','TREK','QUP',NMAXP,log=QUP)
     call chmalloc('adiab.src','TREK','QDOWN',NMAXP,log=QDOWN)
     call chmalloc('adiab.src','TREK','PRNORM',NMAXP,crl=PRNORM)
#if KEY_HFB==1
     call chmalloc('adiab.src','TREK','SLCT',NATOM,intg=SLCT)
     CALL SELCTA(COMLYN,COMLEN,SLCT,X,Y,Z,WMAIN,.TRUE.)
     CALL cntslct(SLCT,NATOM,NSELR)
     call chmalloc('adiab.src','TREK','FPSLCT',NSELR,intg=FPSLCT)

     CALL pntslct(SLCT,FPSLCT,NATOM,NSELR)
     LMWGHT = (INDXA(COMLYN,COMLEN,'MASS')  >  0)
     if (LMWGHT) then
        call chmalloc('adiab.src','TREK','MWGHT',NSELR,crl=MWGHT)
     endif
#endif 
     SEGSTP = ZERO
     ENEMAX = ZERO
     NLINMN = 0
     CALL W0(' ')
     CALL WI(' TReK initialized with MAXPoints = ', NMAXP)
     CALL W0(' If there is a heap failure, reduce MAXPoints.')

     QTRAV  = .TRUE.
  ELSE
     IF (INDXA(COMLYN,COMLEN,'MAXP')  >  0)              CALL WI( &
          ' MAXPoints already set, will be kept unchanged =', NMAXP)
     IF ( (INDXA(COMLYN,COMLEN,'XAXI')  >  0) .OR. &
          (INDXA(COMLYN,COMLEN,'YAXI')  >  0) .OR. &
          (INDXA(COMLYN,COMLEN,'ZAXI')  >  0) ) THEN
        IF (NTRANS == 0) THEN
           CALL WRNDIE(1,'<TREK>', &
                'There are no images: axis keywords ignored !')
        ELSE
           CALL W0( &
                ' Image axis already set, will be kept unchanged.')
        ENDIF
     ENDIF
  ENDIF

  CALL XTRANE(COMLYN,COMLEN,'TReK')

  CALL TREKCOM(NMAXP, NPOINT, XBAS,YBAS,ZBAS, &
       IDPREC, IDNEXT,IDXPRO,NMAXI, &
       PTENE, LENSEG,NEXMIN,PREMIN, &
       STPMAX,SRTENE,SRTIDX,ENEMAX, &
       QUP, QDOWN, SEGSTP, SERIAL, &
       IFREEP, NFREEP, PTGRAD, IDXSAD, &
       NEIBR1,NEIBR2, QTRAV,LSCAL,LREFER, &
       NLINMN,OLDMOV,IMGAXI,IMGROT, LSCM,LCPR, &
#if KEY_HFB==1
       LLIP,LFP,SLCT,NSELR,FPSLCT,                  & 
#endif
#if KEY_HFB==1
       LMWGHT,MWGHT,                                       & 
#endif
       PRNORM,DYNBRA,XREF,YREF, &
       ZREF,TOTUSD, ITEMP,DTEMP, &
       IDUMX,IDUMY,IDUMZ,XS0, &
       YS0,ZS0,XKEEP,YKEEP, &
       ZKEEP,ISLCT,RDTMP,FREEAT)

  IF (.NOT. QTRAV) THEN

     CALL FREEHEAP(NPOINT+NFREEP,XBAS,YBAS,ZBAS, NATOM)

     call chmdealloc('adiab.src','TREK','XREF',NATOM,crl=XREF)
     call chmdealloc('adiab.src','TREK','YREF',NATOM,crl=YREF)
     call chmdealloc('adiab.src','TREK','ZREF',NATOM,crl=ZREF)
     call chmdealloc('adiab.src','TREK','PTENE',NMAXP,crl=PTENE)
     call chmdealloc('adiab.src','TREK','LENSEG',NMAXP,crl=LENSEG)
     call chmdealloc('adiab.src','TREK','SEGSTP',NMAXP,crl=SEGSTP)
     call chmdealloc('adiab.src','TREK','SRTENE',NMAXP,crl=SRTENE)
     call chmdealloc('adiab.src','TREK','ENEMAX',NMAXP,crl=ENEMAX)
     call chmdealloc('adiab.src','TREK','PTGRAD',NMAXP,crl=PTGRAD)
     call chmdealloc('adiab.src','TREK','PRNORM',NMAXP,crl=PRNORM)
     call chmdealloc('adiab.src','TREK','DYNBRA',NMAXP,crl=DYNBRA)
     call chmdealloc('adiab.src','TREK','OLDMOV',NATOM,intg=OLDMOV)
     call chmdealloc('adiab.src','TREK','SERIAL',NMAXP,intg=SERIAL)
     call chmdealloc('adiab.src','TREK','NLINMN',NMAXP,intg=NLINMN)
     call chmdealloc('adiab.src','TREK','NEIBR1',2*NMAXP,intg=NEIBR1)
     call chmdealloc('adiab.src','TREK','NEIBR2',2*NMAXP,intg=NEIBR2)
     call chmdealloc('adiab.src','TREK','IDPREC',NMAXP,intg=IDPREC)
     call chmdealloc('adiab.src','TREK','IDNEXT',NMAXP,intg=IDNEXT)
     call chmdealloc('adiab.src','TREK','STPMAX',NMAXP,intg=STPMAX)
     call chmdealloc('adiab.src','TREK','NEXMIN',NMAXP,intg=NEXMIN)
     call chmdealloc('adiab.src','TREK','PREMIN',NMAXP,intg=PREMIN)
     call chmdealloc('adiab.src','TREK','SRTIDX',NMAXP,intg=SRTIDX)
     call chmdealloc('adiab.src','TREK','IFREEP',NMAXP,intg=IFREEP)
     call chmdealloc_chm_array('adiab.src','TREK','XBAS',NMAXP,XBAS)
     call chmdealloc_chm_array('adiab.src','TREK','YBAS',NMAXP,YBAS)
     call chmdealloc_chm_array('adiab.src','TREK','ZBAS',NMAXP,ZBAS)
     call chmdealloc('adiab.src','TREK','QUP',NMAXP,log=QUP)
     call chmdealloc('adiab.src','TREK','QDOWN',NMAXP,log=QDOWN)

#if KEY_HFB==1

     call chmdealloc('adiab.src','TREK','FPSLCT',NSELR,intg=FPSLCT)
     if (LMWGHT) then
        call chmdealloc('adiab.src','TREK','MWGHT',NSELR,crl=MWGHT)
     endif
#endif 
  ENDIF

  PRNLEV = OLDLEV

  call chmdealloc('adiab.src','TREK','ITEMP',NMAXP,intg=ITEMP)
  call chmdealloc('adiab.src','TREK','DTEMP',NMAXP,crl=DTEMP)
  call chmdealloc('adiab.src','TREK','IDUMX',NATOM,crl=IDUMX)
  call chmdealloc('adiab.src','TREK','IDUMY',NATOM,crl=IDUMY)
  call chmdealloc('adiab.src','TREK','IDUMZ',NATOM,crl=IDUMZ)
  call chmdealloc('adiab.src','TREK','XS0',NATOM,crl=XS0)
  call chmdealloc('adiab.src','TREK','YS0',NATOM,crl=YS0)
  call chmdealloc('adiab.src','TREK','ZS0',NATOM,crl=ZS0)
  call chmdealloc('adiab.src','TREK','XKEEP',NATOM,crl=XKEEP)
  call chmdealloc('adiab.src','TREK','YKEEP',NATOM,crl=YKEEP)
  call chmdealloc('adiab.src','TREK','ZKEEP',NATOM,crl=ZKEEP)
  call chmdealloc('adiab.src','TREK','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('adiab.src','TREK','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('adiab.src','TREK','RDTMP',NATOM,cr4=RDTMP)
#if KEY_HFB==1
  call chmdealloc('adiab.src','TREK','SLCT',NATOM,intg=SLCT)            
#endif
  
  ! leaving trek, dealloate travel_ltm space. cb3
  if(allocated(xscal)) call deallocate_travel_ltm(natom)

  RETURN
END subroutine trek


SUBROUTINE FREEHEAP(N, XBAS,YBAS,ZBAS, NATOM )


  use chm_kinds
  use chm_types
  use exfunc
  use memory
  implicit none

  INTEGER      N, NATOM
  type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
  INTEGER      I

  DO  I = 1, N
     call chmdealloc('adiab.src','FREEHEAP','XBAS(I)',NATOM,crl=XBAS(I)%a)
     call chmdealloc('adiab.src','FREEHEAP','YBAS(I)',NATOM,crl=YBAS(I)%a)
     call chmdealloc('adiab.src','FREEHEAP','ZBAS(I)',NATOM,crl=ZBAS(I)%a)
  end DO

  RETURN
END subroutine freeheap

SUBROUTINE TREKCOM( NMAXP, NPOINT, XBAS, YBAS, ZBAS, &
     IDPREC, IDNEXT , IDXPRO, NMAXI, PTENE, &
     LENSEG, NEXMIN, PREMIN, STPMAX, SRTENE, SRTIDX, &
     ENEMAX, QUP, QDOWN, SEGSTP, SERIAL, IFREEP, NFREEP, &
     PTGRAD, IDXSAD, NEIBR1,NEIBR2, QTRAV,LSCAL,LREFER, &
     NLINMN, OLDMOV, IMGAXI,IMGROT,LSCM,LCPR, &
#if KEY_HFB==1
     LLIP,LFP,SLCT,NSELR,FPSLCT,                  & 
#endif
#if KEY_HFB==1
     LMWGHT,MWGHT,                                & 
#endif
     PRNORM,DYNBRA, XREF,YREF,ZREF, TOTUSD, &
     ITEMP,DTEMP,XDUM,YDUM,ZDUM,XS0,YS0,ZS0,XKEEP,YKEEP, &
     ZKEEP,ISLCT,RDTMP,FREEAT )

  use chm_kinds
  use chm_types
#if KEY_CHEQ==1
  use cheq,only:qcg               
#endif

  use dimens_fcm
  use exfunc
  use number
  use comand
  use consta
  use contrl
  use coord
  use coordc
  use ctitla
  use deriv
  use cvio
  use energym
  use psf
  use stream
  use string
  use trek1
  use trek2
  use memory
#if KEY_PARALLEL==1
  use parallel      
#endif
  use coorio_mod,only:coorio
  use machutil,only:timrb,timre

  implicit none

#if KEY_CHEQ==1
  ! WARNING: CHARGE TRAVEL is dangerous! FLUCQ is not compatible.
  ! Holds QCG :
#endif 


  INTEGER      NMAXP,NPOINT
  type(chm_array)  XBAS(:),YBAS(:),ZBAS(:)
  INTEGER      TOTUSD,SERIAL(*),IDXPRO,IDPREC(*),IDNEXT(*)
  real(chm_real)       LENSEG(*),SEGSTP(*)

  real(chm_real)       PTENE(*),PTGRAD(*)
  INTEGER      STPMAX(*)
  real(chm_real)       ENEMAX(*)
  LOGICAL      QUP(*),QDOWN(*)
  INTEGER      NEXMIN(*),PREMIN(*),NLINMN(*)
  real(chm_real)       DYNBRA(*)

  INTEGER      NFREEP,IFREEP(*)

  INTEGER      NMAXI,SRTIDX(*)
  real(chm_real)       SRTENE(*)

  INTEGER      NEIBOR,NEIBR1(*),NEIBR2(*)

  SAVE         NEIBOR


  LOGICAL      LMVCPR

  SAVE         LMVCPR

  INTEGER      ITEMP(*)
  real(chm_real)       DTEMP(*)

  INTEGER      NDISPL

  LOGICAL      LCPR, LSCANP, LSCANS
#if KEY_HFB==1
  LOGICAL      LLIP,LFP,LMWGHT
  INTEGER      SLCT(*),NSELR,FPSLCT(*)
  real(chm_real)       MWGHT(*)
#endif 

  INTEGER      ININPP

  INTEGER      OLDCYC
  real(chm_real)       OLDSTP,OLDPR1,OLDPR2,OLDGRA
  INTEGER, PARAMETER :: BIGINT=999999
  real(chm_real), PARAMETER :: PT03=PT01*THREE,PT09=PT01*NINE

  LOGICAL      QSTART

  SAVE         QSTART


  LOGICAL      QTRAV,LSCAL
  INTEGER      VERBEU

  INTEGER      OLDMOV(*),NFIXED,IDIM
  real(chm_real)       DDIM

  LOGICAL      EOF
  LOGICAL      LUSED
  CHARACTER(len=4) WRD

  INTEGER, PARAMETER :: MXFLNM=240
  INTEGER      FILLEN,STARLN,NUNIT
  CHARACTER(len=MXFLNM) FILENM
  CHARACTER(len=MXFLNM) STARNM
  CHARACTER(len=MXFLNM) CDUMY1
  CHARACTER(len=6) CUNIT

  INTEGER      I20DUM(20)

  real(chm_real4)       RDTMP(:)
  INTEGER      FREEAT(*),NFREAT,NFILE,ISTEP,ISTATS,NDEGF,BEGIN
  INTEGER      STOP,SKIP,NSAVV
  real(chm_real)       IODUM

  LOGICAL      ORIREF,LREFER,QP1REF
  INTEGER      ISLCT(*)
  real(chm_real)       XREF(:),YREF(:),ZREF(:)
  INTEGER      IMGAXI
  LOGICAL      IMGROT

  INTEGER, PARAMETER :: MAXSAD=1000
  INTEGER      NPTSAD
  INTEGER      INPSAD(MAXSAD)
  INTEGER      IDXSAD

  LOGICAL      ANASCA,NOPEAK,NOMINI,NOSADL,QMINI,LFOUND
  INTEGER      ILARGS,ISMALS,NTERPL
  real(chm_real) ANALST,LAMDA,LARGST,SMALST,STPSIZ,SNORM,JRN,REAPRO

  SAVE         REAPRO

  LOGICAL      LCOMP

  INTEGER      I7010,I7020,I7030,I7040,I7050,I7060,I7070,I7080
  INTEGER      I7090,I7100

  LOGICAL      LDUM1,QERROR
  INTEGER      I,J,K,L
  INTEGER      IDUM1,IDUM2,IDX,IDXNEW,IDXPRE,IDXFOL,TOTENE
  real(chm_real)       DDUM1,DDUM2,DDUM3,DDUM4,DDUM5,DDUM6,DDUM7

  real(chm_real)       XDUM(*),YDUM(*),ZDUM(*)
  real(chm_real)       XS0(*),YS0(*),ZS0(*)

  real(chm_real)       XKEEP(*),YKEEP(*),ZKEEP(*)

  LOGICAL      LSCM
  real(chm_real)       PRNORM(*)
  LOGICAL      LMVSCM

  SAVE         LMVSCM

#if KEY_HFB==1
  !ivk Local Variables
  LOGICAL      LINITFP,LDENS,LFEND,LPRJCT,LPSTRUCT,LNOROTATE
  INTEGER      IDSTART,IDSTOP,ITRUNC,IRTYPE
  INTEGER      IPROFU,NBEADS,NQGRID
  real(chm_real)       RRFRC,RSTEP
#endif 


  NDISPL = 80
  EOF    = .FALSE.
  VERBEU = 2
#if KEY_PARALLEL==1
  IF (MYNOD /= 0) VERBEU = 0
#endif 
  NPTSAD = 0
  CALL FIXED(NFIXED)

  CALL COP3D( X,Y,Z, XKEEP,YKEEP,ZKEEP, NATOM)

  NFREAT = NATOM

  IF (NPOINT == 0) THEN
     DO I = 1, NATOM
        OLDMOV(I) = IMOVE(I)
     ENDDO
  ELSE
     LDUM1 = .FALSE.
     DO I = 1, NATOM
        IF (IMOVE(I) /= OLDMOV(I))  LDUM1 = .TRUE.
     ENDDO
     IF (LDUM1) THEN
        CALL W0(' ')
        CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        CALL W0(' Warning : do not proceed with REFInements,')
        CALL WRNDIE(-4,'<TREKCOM>', &
             'because some atoms had their "FIXED" status changed.')
     ENDIF
  ENDIF

  IDIM  = 3*(NATOM - NFIXED)
  DDIM  = IDIM

  IF (NFIXED < 0 .OR. (NFIXED > 0.AND.NFIXED < 3)) THEN
     CALL W0(' ')
     CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
     CALL WI(' Warning : there are fixed atoms NFIXED =', NFIXED)
     CALL WRNDIE(-4,'<TREKCOM>', &
          'Recommended NFIXED: either 0 or much larger than 3')
  ENDIF

1 CALL XTRANE(COMLYN,COMLEN,'TREKCOM')
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
       'TReK> ')
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  IF (LUSED) GOTO  1
  WRD=NEXTA4(COMLYN,COMLEN)


  IF          (WRD ==  '    ')     THEN
     GOTO  1

  ELSEIF   (WRD == 'REFI'.OR.WRD == 'CPR') THEN

     IF ( NPOINT > 1 ) THEN

        NCYCLE = 1
        NCYCLE = GTRMI(COMLYN,COMLEN,'NCYC', NCYCLE )

        IF (.NOT.LCPR) THEN

           LMVCPR = .TRUE.

           LSCM = .FALSE.
           call sub7050

        ENDIF

        LSCANP = .FALSE.

        NECALL = GTRMI(COMLYN,COMLEN,'NECA', NECALL )
        IF (NECALL < BIGINT.AND.NCYCLE == 1)  NCYCLE = BIGINT
        MAXCAL = MAXCAL + NECALL

        IF (.NOT.LCPR .OR. COMLEN > 3) THEN

           LMVCPR = .TRUE.

           call sub7080

        ENDIF

        CALL XTRANE(COMLYN,COMLEN,'TREKCOM')

        IF (.NOT.LCPR.OR.LSCANS) THEN
           LMVCPR = .TRUE.

           call sub7090
        ENDIF

        CALL SETIHI( IDXSAD,NPOINT,IDNEXT,PTGRAD,SADGRD,QUP,QDOWN, &
             NLINMN,SADMIN,PTENE )

        LCPR   = .TRUE.
        LSCANS = .FALSE.

        IF (LMVCPR .OR. (LMVSCM.AND.LSCM)) THEN

           LMVCPR = .FALSE.

           CALL CPRRUN( LMVCPR,IDXSAD, NMAXP,NPOINT,XBAS,YBAS,ZBAS, &
                TOTUSD,SERIAL,IDXPRO,IDPREC,IDNEXT,LENSEG, &
                SEGSTP, PTENE,PTGRAD,STPMAX,ENEMAX,QUP,QDOWN, &
                NEXMIN,PREMIN,NLINMN,DYNBRA, NFREEP,IFREEP, &
                NMAXI,SRTIDX,SRTENE, NEIBOR,NEIBR1,NEIBR2, &
                ISLCT,XREF,YREF,ZREF,IMGAXI,IMGROT, &
                ITEMP,DTEMP, NATOM,NFIXED,VERBEU,NDISPL, &
                XDUM,YDUM,ZDUM,XS0,YS0,ZS0, X,Y,Z)
        ELSE
           CALL W0(' ')
           CALL W0(' Nothing new. CPR command is ignored !')
        ENDIF

     ELSE

        CALL W0(' ')
        IF (NPOINT < 2) &
             CALL W0('Need 2 or more path-points to run CPR !')

        CALL W0('CPR command is ignored !')
        COMLYN = ' '
        COMLEN = LEN(COMLYN)

     ENDIF

  ELSEIF   (WRD ==  'DISP')     THEN
     NDISPL = NEXTI(COMLYN,COMLEN)

  ELSEIF   (WRD ==  'SADD')     THEN
     IF (.NOT.LCPR .AND. .NOT.LSCM) THEN
8       IDUM1 = NEXTI(COMLYN,COMLEN)
        IF (IDUM1 > 0) THEN
           IF (NPTSAD < MAXSAD) THEN
              NPTSAD = NPTSAD + 1
              INPSAD(NPTSAD) = IDUM1
              GOTO  8
           ELSE
              CALL W0(' ')
              CALL WI(' Already maximum allowed of '// &
                   'flagged saddles MAXSAD =', MAXSAD)
              CALL WRNDIE(-1,'<TREKCOM>','Point refused !')
           ENDIF
        ENDIF
     ELSE
        CALL W0(' ')
        CALL W0( &
             ' Cannot use this command after refinement started !')
        CALL WRNDIE(-1,'<TREKCOM>','Command out of context !')
     ENDIF

  ELSEIF   (WRD ==  'SCM')     THEN

     IF (LMVSCM.OR.LSCM .OR. (LMVCPR.AND.LCPR) ) THEN

        IF (.NOT.LSCM) THEN

           DO L=1, NPTSAD
              NLINMN(INPSAD(L)) = -3*IDIM
           ENDDO
           NPTSAD = 0

        ENDIF

        CALL SCM( NMAXP, NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
             IDPREC,IDNEXT,IDXPRO,NLINMN, COMLYN,COMLEN, &
             VERBEU,LSCM, LMVSCM, XS0,YS0,ZS0, &
             XDUM,YDUM,ZDUM,PRNORM,DYNBRA, PTENE,PTGRAD, &
             NFIXED,IMGAXI,IMGROT, ISLCT, XREF,YREF,ZREF)

        IF (LMVSCM) THEN

           LCPR = .FALSE.

           IDX = 1
           DO J=1,NPOINT
              SEGSTP(IDX) = -ONE
              LENSEG(IDX) =  ZERO
              STPMAX(IDX) =  BIGINT
              ENEMAX(IDX) = -RBIG
              QUP(IDX) = .FALSE.
              QDOWN(IDX) = .FALSE.
              NEXMIN(IDX) = 0
              PREMIN(IDX) = 0

              IDX = IDNEXT(IDX)
           ENDDO
        ENDIF

     ELSE
        CALL W0(' ')
        CALL W0( &
             'Path has not changed since last SCM re-initialization.')
        CALL W0('SCM command is ignored !')
        COMLYN = ' '
        COMLEN = LEN(COMLYN)
     ENDIF

  ELSEIF   (WRD == 'SDP') THEN
     CALL SDPATH( NMAXP, NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
          IDPREC,IDNEXT,IDXPRO, COMLYN,COMLEN, &
          XS0,YS0,ZS0, XDUM,YDUM,ZDUM, VERBEU, &
          NFREEP,IFREEP,TOTUSD,SERIAL,NLINMN)

  ELSEIF   (WRD == 'CROS') THEN
     IF (NPOINT > 3.AND.IDXSAD <= 0) THEN
        CALL WRNDIE(1,'<TREKCOM>','Points > 3, but no saddle-point.')
     ELSEIF (NPOINT < 3) THEN
        CALL WRNDIE(1,'<TREKCOM>','CROSAD needs 3 path-points.')
     ELSE
        IF (NPOINT == 3) THEN
           CALL W0(' ')
           CALL W0('Of the 3 points provided, the 2nd will be used'// &
                ' as the saddle-point.')
        ELSEIF (NPOINT > 3) THEN
           CALL W0(' ')
           CALL WI('The highest saddle-point, index =', IDXSAD)
           CALL W0('will be used to get the crossing mode.')
           CALL W0('Except for its two neighbors, all other'// &
                ' points are now deleted.')
           IDX = IDPREC(IDXSAD)
           CALL COP3D( XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
                XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM)
           IDX = IDNEXT(IDX)
           CALL COP3D( XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
                XBAS(2)%a,YBAS(2)%a,ZBAS(2)%a, NATOM)
           IDX = IDNEXT(IDX)
           CALL COP3D( XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
                XBAS(3)%a,YBAS(3)%a,ZBAS(3)%a, NATOM)
           DO I=4, NPOINT+NFREEP
              call chmdealloc('adiab.src','TREKCOM','XBAS(I)', &
                   NATOM,crl=XBAS(I)%a)
              call chmdealloc('adiab.src','TREKCOM','YBAS(I)', &
                   NATOM,crl=YBAS(I)%a)
              call chmdealloc('adiab.src','TREKCOM','ZBAS(I)', &
                   NATOM,crl=ZBAS(I)%a)
           ENDDO
           NPOINT = 3
           NFREEP = 0
           IDXPRO = 3
           IDNEXT(1) = 2
           IDNEXT(2) = 3
           IDNEXT(3) = 0
           IDPREC(1) = 0
           IDPREC(2) = 1
           IDPREC(3) = 2

           NLINMN(2) = NLINMN(IDXSAD)
           NLINMN(IDXSAD) = 0
           IDXSAD = 2
        ENDIF
        LCPR = .FALSE.
        LSCM = .FALSE.
        CALL CROSAD( NMAXP,NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
             IDPREC,IDNEXT,IDXPRO, COMLYN,COMLEN, &
             XS0,YS0,ZS0, XDUM,YDUM,ZDUM, VERBEU, &
             NFREEP,IFREEP, TOTUSD,SERIAL,NLINMN)

     ENDIF

  ELSEIF   (WRD ==  'TRAJ')     THEN


     call sub7030

  ELSEIF   (WRD ==  'COPY')     THEN

     call sub7010

  ELSEIF   (WRD ==  'CHRO')     THEN
     WRD = NEXTA4(COMLYN,COMLEN)
     IF (WRD ==  'RESE') THEN
        CALL TIMRB
        CALL W0(' Resetting the CPU chronometer .')
     ELSEIF (WRD ==  'PRIN') THEN
        CALL W0(' Since last CHRONo RESEt : ')
        CALL TIMRE
     ELSE
        CALL WRNDIE(1,'<TREKCOM>', &
             ' Required options with CHRONo : RESEt or PRINt ')
     ENDIF

  ELSEIF   (WRD ==  'VERB')     THEN
     VERBEU = NEXTI(COMLYN,COMLEN)
#if KEY_PARALLEL==1
     IF  (MYNOD /= 0) VERBEU = 0
#endif 

#if KEY_HFB==1
     !ivk
  ELSEIF   (WRD ==  'LIP') THEN
     CALL WRNDIE(1,'<TREKCOM>','Line Integral Implementation')
  ELSEIF   (WRD ==  'FP') THEN
     CALL WRNDIE(1,'<TREKCOM>','Fourier Path Implementation')
     if (.not. LFP) then
        !ivk initialize what you would need
        !ivk in case of CPR at this point the restart file is read
        !ivk I need to provide a selection for the reaction coordinate
        !ivk and that is basically it for now 'noro notra mass weigh frcp sele space end'
        !ivk     Select atoms for the reaction space
        !ivk +clb3          LPRJCT=(INDXA(COMLYN,COMLEN,'PRJC') > 0)
        !ivk +clb3          LFEND=(INDXA(COMLYN,COMLEN,'FEND') > 0)
        LDENS=(INDXA(COMLYN,COMLEN,'DENS') > 0)
! added keyword norotate to eliminate rotation/translation of replicas
        LNOROTATE = (INDXA(COMLYN,COMLEN,'NORO') > 0)
        if (lnorotate) then
           write (outu,'(a)') &
                " No CM translation or rotation done to beads"
        endif
        LINITFP=(INDXA(COMLYN,COMLEN,'INIT') > 0)
        IF (LDENS .and. LINITFP) THEN
           CALL WRNDIE(-1,'<TREKCOM>','keywords DENS and INIT are mutually exclusive')
        ENDIF
        IF (LINITFP) THEN
           IDSTART = GTRMI(COMLYN,COMLEN,'STRT',1)
           IDSTOP  = GTRMI(COMLYN,COMLEN,'STOP',2)
        ENDIF
        ITRUNC = GTRMI(COMLYN,COMLEN,'TRNC',NMAXP)
        NQGRID = GTRMI(COMLYN,COMLEN,'GRID',NMAXP*4)
        ! Fill the weight array
        DO I=1,NSELR
           J=FPSLCT(I)
           MWGHT(I)=ONE
           IF(LMWGHT) MWGHT(I)=AMASS(J)
        ENDDO
        LFP = .TRUE.
     else
        CALL W0('FP already initialized')
        LINITFP=.FALSE.
     endif
     !ivk do the work right here - no need for any specail place
     CALL FRCPATH(XBAS,YBAS,ZBAS, &
          FPSLCT,NSELR,NATOM,NPOINT,NMAXP, &
          MWGHT,LINITFP,IDSTART,IDSTOP,ITRUNC, &
          LDENS,LNOROTATE, &
          XS0,YS0,ZS0, &
          XKEEP,YKEEP,ZKEEP, &
          XDUM,YDUM,ZDUM, &
          QERROR,NFREEP,IFREEP, &
          TOTUSD,SERIAL,NLINMN, &
          IDPREC,IDNEXT, &
          NQGRID)

  ELSEIF   (WRD ==  'WRKP') THEN
     CALL WRNDIE(1,'<TREKCOM>','Fourier Path Implementation')
     !ivk On the evening of Aug 22 I had an idea how to improve the FP method
     !ivk in particular 1) how to speed up path optimization
     !ivk and more importantly 2) how to get free energy profile along the alpha 
     !ivk so I am very exciting, especially because my way of computing the 
     !ivk free energy profile is very general and does not need any WHAM at all!
     !ivk so WHAM will soon be a history, unless of cause you want to have multiple
     !ivk dimensions to the Free Energy surface
     !ivk After spending all the weekends Aug 26 and 27 trying to prove the connection
     !ivk I finally succeeded after making one important approximation (that I called 
     !ivk Khavrutskii-Brooks approximation). During the derivation I realized that 
     !ivk my original implementation as of Aug22 was slightly incorrect. Ayori Mitsutake
     !ivk verified my final proof, and I am now correcting the implementation. 
     if (.not. LFP) then
        !ivk initialize what you would need
        !ivk in case of CPR at this point the restart file is read
        !ivk I need to provide a selection for the reaction coordinate
        !ivk and that is basically it for now 'noro notra mass weigh frcp sele space end'
        !ivk     Select atoms for the reaction space
        LPRJCT=(INDXA(COMLYN,COMLEN,'PRJC') > 0)
        LFEND=(INDXA(COMLYN,COMLEN,'FEND') > 0)
        LDENS=(INDXA(COMLYN,COMLEN,'DENS') > 0)
        LINITFP=(INDXA(COMLYN,COMLEN,'INIT') > 0)
        LPSTRUCT =(INDXA(COMLYN,COMLEN,'PTRJ') > 0)
        IRTYPE = GTRMI(COMLYN,COMLEN,'RTYP',1)
        IPROFU = GTRMI(COMLYN,COMLEN,'IPRF',-1)
        NBEADS = GTRMI(COMLYN,COMLEN,'BEAD',-1)
        NQGRID = GTRMI(COMLYN,COMLEN,'GRID',-1)
! added keyword norotate to eliminate rotation/translation of replicas
        LNOROTATE = (INDXA(COMLYN,COMLEN,'NORO') > 0)
        if (lnorotate) then
           write (outu,'(a)') &
                " No CM translation or rotation done to beads"
        endif
        !ivk Introducing the type of the harmonic restraint RTYP
        !ivk 1 - absolute positional harmonic restraint (default)
        RSTEP = GTRMF(COMLYN,COMLEN,'STEP',ZERO)
        RRFRC = GTRMF(COMLYN,COMLEN,'RFRC',ZERO)
        !ivk RFRC - restraint force constant
        IF (LINITFP) THEN
           IDSTART = GTRMI(COMLYN,COMLEN,'STRT',1)
           IDSTOP  = GTRMI(COMLYN,COMLEN,'STOP',2)
        ENDIF
        ITRUNC = GTRMI(COMLYN,COMLEN,'TRNC',-1)
        ! Fill the weight array
        DO I=1,NSELR
           J=FPSLCT(I)
           MWGHT(I)=ONE
           IF(LMWGHT) MWGHT(I)=AMASS(J)
        ENDDO
        LFP = .TRUE.
     else
        CALL W0('FP already initialized')
        LINITFP=.FALSE.
     endif
     !ivk Take care of some default settings here
     IF (NBEADS  <  0) THEN
        IF (MOD (NPOINT,2)  ==  0) THEN
           NBEADS = NPOINT/2
        ELSE
           CALL WI('An odd total number of evolved and reference points provided ', &
                NPOINT)
           CALL WRNDIE(-1,'<TREKOM>','Exiting for now')
        ENDIF
     ENDIF

     IF (ITRUNC  <  0) THEN
        ITRUNC = (NBEADS)/4*3
     ENDIF

     IF (NQGRID  <  0) THEN
        NQGRID = NBEADS * 4
        !        CALL WI('The number of qudrature grid-points is ',
        !     $          NQGRID )
     ELSEIF (NQGRID  <  NBEADS * 4) THEN
        NQGRID = NBEADS * 4
        CALL WI('The number of grid-points too small, increasing to ', &
             NQGRID )
     ELSEIF (NQGRID  >  NMAXP .and. LPSTRUCT) THEN
        CALL WI('GRID cannot exceed MAXP with PTRJ option, either reduce GRID or increase MAXP to ', &
             NQGRID )
        CALL WRNDIE(-1,'<TREKOM>','Exiting for now')
        RETURN
     ENDIF

     CALL WI('The number of qudrature grid-points is ', &
          NQGRID)


     !ivk do the work right here - no need for any specail place
     CALL WRKFRCPATH(XBAS,YBAS,ZBAS, &
          FPSLCT,NSELR,NATOM,NPOINT,NMAXP, &
          MWGHT,LINITFP,IDSTART,IDSTOP,ITRUNC, &
          LDENS,LFEND,LPRJCT,LPSTRUCT,LNOROTATE, &
          XS0,YS0,ZS0, &
          XKEEP,YKEEP,ZKEEP, &
          XDUM,YDUM,ZDUM, &
          QERROR,NFREEP,IFREEP, &
          TOTUSD,SERIAL,NLINMN, &
          IDPREC,IDNEXT, &
          IRTYPE,RRFRC,RSTEP, &
          IPROFU,NBEADS,NQGRID)

#endif 

  ELSEIF   (WRD ==  'QUIT' .OR. WRD ==  'END ') THEN
     IF (WRD ==  'QUIT')  THEN
        QTRAV = .FALSE.
     ENDIF

     CALL COP3D(XKEEP,YKEEP,ZKEEP,X,Y,Z,NATOM)
     CALL XTRANE(COMLYN,COMLEN,'TREKCOM')
     RETURN

  ELSE
     CALL WRNDIE(1,'<TREKCOM>','non-existent command !')
  ENDIF

  GOTO  1

  !--------------------------- recursive subroutines ----------------------------
contains

  subroutine sub7030

    NUNIT = GTRMI(COMLYN,COMLEN,'UNIT',40)
    CALL GTRMWA(COMLYN,COMLEN,'NAME',4,FILENM,MXFLNM,FILLEN)
    CALL GTRMWA(COMLYN,COMLEN,'REST',4,STARNM,MXFLNM,STARLN)
    WRD = NEXTA4(COMLYN,COMLEN)

    IF (WRD ==  'READ') THEN
       QP1REF = .FALSE.
       IF (.NOT.LREFER)  QP1REF = (INDXA(COMLYN,COMLEN,'REFE') > 0)
       IF (.NOT.LCPR .AND. .NOT.LSCM) THEN
          LMVCPR = .TRUE.
          LMVSCM = .TRUE.
          ORIREF = .FALSE.
          LORIEN = .FALSE.
          IF (INDXA(COMLYN,COMLEN,'NOOR') > 0) LORIEN = .FALSE.
          IF (INDXA(COMLYN,COMLEN,'ORIE') > 0) THEN
             LORIEN = .TRUE.
             ORIREF = .TRUE.
          ENDIF

          IF (FILLEN == 0) THEN
             call sub7100

          ELSE
             call sub7020

          ENDIF
          CALL W0(' ')
          CALL WI('Number of starting Path-Points: NPOINT =',NPOINT)
       ELSE
          CALL WRNDIE(-1,'<TREKCOM>', &
               'Cannot "READ"-in more structures after refinement started.')
       ENDIF

    ELSEIF (WRD ==  'WRIT') THEN
       IF (FILLEN < 1) THEN
          CALL WRNDIE(1,'<TREKCOM>','Need to provide  "NAME file".')
          CALL W0('Will be writing path to "trek.trj".')
          FILENM = ' trek.trj '
          FILLEN = 10
       ENDIF
       CALL ENCODI(NUNIT,CUNIT,6,IDUM1)
       COMLYN = 'WRITE UNFO UNIT '//CUNIT//' NAME '//FILENM(1:FILLEN+1)
       COMLEN = FILLEN + LEN(CUNIT) + 25
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          CALL OPNLGU(COMLYN,COMLEN,NUNIT)
#if KEY_PARALLEL==1
       ENDIF
#endif 

       NDEGF = 0
       IODUM = ZERO
       I = 1
       DO J=1,NPOINT
          CALL COP3D( XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, &
               XDUM,YDUM,ZDUM, NATOM)
          CALL UNSTRE(XDUM,YDUM,ZDUM, NATOM)


#if KEY_PARALLEL==1
          IF (MYNOD == 0) THEN
#endif 

             CALL WRITCV(XDUM,YDUM,ZDUM, &
#if KEY_CHEQ==1
                  CG,QCG,                 & 
#endif
                  NATOM,FREEAT,NFREAT,1,J,NDEGF,IODUM,1,NPOINT, &
                  TITLEA,NTITLA,NUNIT,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))

#if KEY_PARALLEL==1
          ENDIF
#endif 


          I = IDNEXT(I)
       ENDDO

       IF (STARLN < 1) THEN
          STARNM = FILENM(1:FILLEN)//'.trk'
          STARLN = FILLEN + 4
          CALL WRNDIE(1,'<TREKCOM>','Should provide  "RESTART file".')
          CALL W0('Will use file-name = '//STARNM(1:STARLN))
       ENDIF

       CALL ENCODI(NUNIT,CUNIT,6,IDUM1)
       COMLYN = 'WRITE FORM UNIT '//CUNIT//' NAME '//STARNM(1:STARLN+1)
       COMLEN = STARLN + LEN(CUNIT) + 25
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          CALL OPNLGU(COMLYN,COMLEN,NUNIT)

          WRITE(NUNIT,'(A)') '* CPR RESTART-FILE (FORMAT 1.1),'// &
               '  CORRESPONDING TO THE PATH IN FILE :'
          WRITE(NUNIT,'(A)') '*      '//FILENM(1:FILLEN+1)
          CALL WRTITL(TITLEA,NTITLA,NUNIT,0)
          WRITE(NUNIT,'(A)') '                  THE CPR PARAMETERS WERE :'

          WRITE(NUNIT,'(A)') ' SADGRD    SADMIN   STEPSZ  REDLUP'// &
               '   DELTAS  LORIEN  PRTOL1   PRTOL2   INTERP'
          WRITE(NUNIT,1114) SADGRD, SADMIN, STEPSZ/SQRT(DDIM), REDLUP, &
               DELTAS/SQRT(DDIM), LORIEN, PRTOL1*(DDIM-ONE), &
               PRTOL2*(DDIM-ONE), INTERP
          WRITE(NUNIT,'(A)') '========================================'// &
               '==============================================='
1114      FORMAT( 1PG9.4E1,1X,I6,2X,1PG9.4E1,1X,I4,3X,1PG9.2E2,3X,L2, &
               2X,2(1X,1PG9.4E1),2x,I4 )

          WRITE(NUNIT,'(A)') '  N    IDX   Lambda  X(N)-Xref     Energy'// &
               '               Grad    LinMin  Curvat    StepSiz         '// &
               'LenSeg      StpMax         MaxEnergy       NxMn PvMn Up Dwn'
          WRITE(NUNIT,'(A)') '----- ----- --------- -------- ----------'// &
               '------------- ------- ------- -------- ----------- ------'// &
               '----------- ------ ----------------------- ---- ---- -- ---'
#if KEY_PARALLEL==1
       ENDIF
#endif 

       LAMDA = ZERO
       I = 1
       DO J=1,NPOINT
          CALL DSUM2( XDUM,YDUM,ZDUM, &
               XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, &
               -ONE, XREF,YREF,ZREF, NATOM)
          DDUM1 = SQRT( DSDOT2( XDUM,YDUM,ZDUM,NATOM) / DDIM )

          DDUM2 = ZERO
          IF (I == 1 ) THEN
             CALL DSUM2( XS0,YS0,ZS0, &
                  XBAS(IDNEXT(I))%a,YBAS(IDNEXT(I))%a, &
                  ZBAS(IDNEXT(I))%a, -ONE, &
                  XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, NATOM)
             IF ( LENSEG(I) <= ZERO) &
                  LENSEG(I) = SQRT( DSDOT1( XS0,YS0,ZS0, NATOM) )

          ELSEIF ( I /= IDXPRO ) THEN
             CALL DSUM2(XDUM,YDUM,ZDUM, &
                  XBAS(IDNEXT(I))%a,YBAS(IDNEXT(I))%a, &
                  ZBAS(IDNEXT(I))%a, -ONE, &
                  XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, NATOM)
             IF ( LENSEG(I) <= ZERO) &
                  LENSEG(I) = SQRT( DSDOT1( XDUM,YDUM,ZDUM, NATOM) )

             DDUM2 = VANGLD( XS0,YS0,ZS0, LENSEG(IDPREC(I)), &
                  XDUM,YDUM,ZDUM, LENSEG(I), NATOM)
             DDUM2=DDUM2*TWO*SQRT(DDIM)/(LENSEG(IDPREC(I))+LENSEG(I))

             CALL COP3D( XDUM,YDUM,ZDUM, XS0,YS0,ZS0, NATOM)
          ENDIF

          IDUM1 = I
          IF ( SERIAL(I) == I .AND. I <= ININPP )  IDUM1 = -I
          IDUM2 = PRTLNM( NLINMN(I) )
          IF (ABS(DDUM1) < TENM5)  DDUM1 = ZERO
          IF (ABS(DDUM2) < TENM8)  DDUM2 = ZERO

          K = IDPREC(I)
          IF (I == 1)  K = IDXPRO
          DDUM3 = SEGSTP(K)/SQRT(DDIM)
          IF (SEGSTP(K) <= ZERO)  DDUM3 = -ONE

#if KEY_PARALLEL==1
          IF (MYNOD == 0) THEN
#endif 
             WRITE(NUNIT,1131)  J,IDUM1,LAMDA/SQRT(DDIM),DDUM1, PTENE(I), &
                  PTGRAD(I), IDUM2, DDUM2,DDUM3,LENSEG(K)/SQRT(DDIM), &
                  STPMAX(K),ENEMAX(K), NEXMIN(K), PREMIN(I), &
                  QUP(I), QDOWN(K)
#if KEY_PARALLEL==1
          ENDIF
#endif 

          LAMDA = LAMDA + LENSEG(I)

          I = IDNEXT(I)
       ENDDO

1131   FORMAT(I5,1X,I5,1X,1PE9.4E1,1X,1PE8.3E1,1X,1PG23.16E2,1X, &
            1PE7.2E1,1X,I7,1X,1PE8.3E1,1X,1PG11.6E1,1X, &
            1PG17.12E1,1X,I6,1X,1PG23.16E2,1X,I4,1X,I4,1X,L2,1X,L2)

#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          CALL VCLOSE(NUNIT,'KEEP',LDUM1)
#if KEY_PARALLEL==1
       ENDIF
#endif 


    ELSEIF (WRD ==  'ANAL') THEN

       call sub7060

    ELSEIF (WRD ==  'INCR') THEN

       CALL DSUM2( XDUM,YDUM,ZDUM, &
            XBAS(IDXPRO)%a,YBAS(IDXPRO)%a,ZBAS(IDXPRO)%a, &
            -ONE ,XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM )
       REAPRO = SQRT( DSDOT2(XDUM,YDUM,ZDUM,NATOM) / DDIM )
       IDUM1 = GTRMI(COMLYN,COMLEN,'NGRI',   5    )
       ANALST = ( REAPRO*SQRT(DDIM) ) / (IDUM1+1)

       IF (LSCAL) THEN
          LORIEN = .FALSE.
       ELSE
          IF (INDXA(COMLYN,COMLEN,'ORIE') > 0) LORIEN = .TRUE.
          IF (INDXA(COMLYN,COMLEN,'NOOR') > 0) LORIEN = .FALSE.
       ENDIF
       ANALST = GTRMF(COMLYN,COMLEN,'STEP', ANALST/SQRT(DDIM) )
       ANALST = ANALST*SQRT(DDIM)

       CALL W0(' ')
       CALL W0('Increasing the numb. of points by interpolating')
       CALL WR( 'between present points, interpolat. STEPSize =', &
            ANALST/SQRT(DDIM))
       CALL W0('-----------------------------------------------')

       IDX = 1
       IDUM1 = NPOINT
       DO J=2,IDUM1

          IDXPRE = IDX
          IDX    = IDNEXT(IDX)
          CALL DSUM2( XS0,YS0,ZS0, &
               XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
               -ONE, XBAS(IDXPRE)%a,YBAS(IDXPRE)%a, &
               ZBAS(IDXPRE)%a, NATOM)
          DDUM4 =SQRT( DSDOT1(XS0,YS0,ZS0,NATOM) )

          NTERPL = MAX( NINT(DDUM4/ANALST) , 1 )

          DO K = 1, NTERPL-1
             IDXNEW = GETIDX( QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
                  XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
             IF (QERROR)  GOTO  205

             DDUM1 = K
             DDUM1 = DDUM1/NTERPL
             CALL DSUM2( &
                  XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
                  XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
                  DDUM1, XS0,YS0,ZS0, NATOM )

             IF (LORIEN.AND..NOT.LSCAL) THEN
                CALL ORIENT(IDXNEW, XBAS(IDXNEW)%a, &
                     YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a,XREF,YREF,ZREF, &
                     NATOM,NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )
             ENDIF

             IDPREC(IDXNEW)      = IDPREC(IDX)
             IDNEXT(IDPREC(IDX)) = IDXNEW
             IDPREC(IDX)         = IDXNEW
             IDNEXT(IDXNEW)      = IDX

             LENSEG(IDPREC(IDXNEW)) = DDUM4/NTERPL
             SEGSTP(IDPREC(IDXNEW)) = -ONE

             STPMAX(IDXNEW) =  BIGINT
             ENEMAX(IDXNEW) = -RBIG
             QUP(IDXNEW) = .FALSE.
             QDOWN(IDXNEW) = .FALSE.
             NEXMIN(IDXNEW) = 0
             PREMIN(IDXNEW) = 0
             PTENE(IDXNEW) = RBIG
             PTGRAD(IDXNEW) = MEGA

          ENDDO
          IF (NTERPL > 1)  THEN
             CALL W2I( 'Segment-IDX & Number of points added =', &
                  IDXPRE, NTERPL-1 )

             LENSEG(IDPREC(IDXNEW)) = DDUM4/NTERPL
             SEGSTP(IDXNEW) = -ONE
          ENDIF

       ENDDO

205    CALL W0(' ')
       CALL WI('The new total number of path-points NPOINT =', NPOINT)
       CALL LISTSD( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )

       LMVCPR = .TRUE.
       LMVSCM = .TRUE.
       LSCM   = .FALSE.
       LCPR   = .FALSE.

    ELSEIF (WRD ==  'DECR') THEN
       TOTENE = ECALLS

       DO L=1, NPTSAD
          NLINMN(INPSAD(L)) = -3*IDIM
       ENDDO
       NPTSAD = 0

       CALL DSUM2( XDUM, YDUM, ZDUM, &
            XBAS(IDXPRO)%a,YBAS(IDXPRO)%a,ZBAS(IDXPRO)%a, &
            -ONE ,XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM )
       REAPRO = SQRT( DSDOT2(XDUM,YDUM,ZDUM,NATOM) / DDIM )
       NOPEAK = .TRUE.
       NOMINI = .TRUE.
       NOSADL = .TRUE.

       IDUM1 = GTRMI(COMLYN,COMLEN,'NGRI',   10   )
       ANALST = ( REAPRO*SQRT(DDIM) ) / (IDUM1+1)
       ANALST = GTRMF(COMLYN,COMLEN,'STEP', ANALST/SQRT(DDIM) )
       ANALST = ANALST*SQRT(DDIM)
       IF (INDXA(COMLYN,COMLEN,'PEAK') > 0)  NOPEAK = .FALSE.
       IF (INDXA(COMLYN,COMLEN,'MINI') > 0)  NOMINI = .FALSE.
       IF (INDXA(COMLYN,COMLEN,'SADD') > 0)  NOSADL = .FALSE.

       CALL W0(' ')
       CALL W0('Decreasing the numb. of points, by removing')
       CALL WR('successive points closer than  STEPSize     =', &
            ANALST/SQRT(DDIM))

       IF (NOSADL) &
            CALL W0('Preserves declared saddle.')

       IF (NOMINI) &
            CALL W0('Preserves local energy-minima along the path.')

       IF (NOPEAK) &
            CALL W0('Allows no energy-peaks within new path-segments.')
       CALL W0('--------------------------------------------------')

       IDX    = IDNEXT(1)
       IF ( .NOT.LCPR .AND. (NOMINI.OR.NOPEAK) ) THEN
          CALL COP3D( XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, X,Y,Z,NATOM)
          PTENE(1) = ENERG( X,Y,Z, DX,DY,DZ, .FALSE.,.FALSE., ZERO )
          CALL COP3D( XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
               X,Y,Z, NATOM )
          PTENE(IDX) = ENERG( X,Y,Z, DX,DY,DZ, .FALSE.,.FALSE., ZERO )
       ENDIF
       QMINI  = .FALSE.
       LFOUND = .FALSE.

208    IF (IDX == IDXPRO)  GOTO  209

       IDXPRE = IDPREC(IDX)
       IDXFOL = IDNEXT(IDX)
       IF ( .NOT.LCPR .AND. (NOMINI.OR.NOPEAK) ) THEN
          CALL COP3D( XBAS(IDXFOL)%a,YBAS(IDXFOL)%a, &
               ZBAS(IDXFOL)%a, X,Y,Z, NATOM)
          PTENE(IDXFOL) = ENERG( X,Y,Z,DX,DY,DZ,.FALSE.,.FALSE.,ZERO)
       ENDIF

       IF (NOMINI) THEN
          QMINI = .FALSE.
          IF ( PTENE(IDX) < PTENE(IDXPRE) .AND. &
               PTENE(IDX) < PTENE(IDXFOL)       )  QMINI = .TRUE.
       ENDIF

       IF (NOPEAK) THEN
          CALL DSUM2( XS0,YS0,ZS0, &
               XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
               -ONE, XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
               NATOM )
          SNORM = SQRT(DSDOT1(XS0,YS0,ZS0,NATOM))
          DDUM1 = ZERO
          DDUM2 = HALF
          DDUM3 = ONE
          DDUM4 = PTENE(IDXPRE)
          DDUM6 = PTENE(IDXFOL)

          CALL SERCH1(DDUM1,DDUM2,DDUM3,DDUM4,DDUM5,DDUM6, E1DIM, &
               XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
               XS0,YS0,ZS0, SNORM, NATOM, &
               3, .FALSE., .FALSE.,.TRUE.,.FALSE., LFOUND )
       ENDIF

       CALL DSUM2( XS0,YS0,ZS0, &
            XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
            -ONE, XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
            NATOM)
       SNORM = SQRT(DSDOT1(XS0,YS0,ZS0,NATOM))

       IF ( SNORM < ANALST.AND. (NLINMN(IDX) >= 0 .OR. .NOT.NOSADL) &
            .AND. .NOT.QMINI .AND. .NOT.LFOUND ) THEN
          CALL WI( 'Removing point IDX =', IDX)
          NPOINT = NPOINT - 1
          NFREEP = NFREEP + 1
          IFREEP(NFREEP) = IDX
          IDNEXT(IDXPRE) = IDXFOL
          IDPREC(IDXFOL) = IDXPRE
          SEGSTP(IDXPRE) = -ONE
          LENSEG(IDXPRE) = ZERO
       ENDIF

       IDX = IDXFOL
       GOTO  208

209    CALL W0(' ')
       CALL WI('The remaining number of path-points NPOINT =',NPOINT)
       CALL WI('Nb. of energy calls during DECReasing=',ECALLS-TOTENE)
       CALL LISTSD( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )
       CALL LISTMN( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )
       LMVCPR = .TRUE.
       LMVSCM = .TRUE.
       LSCANS = .TRUE.
       LSCM   = .FALSE.

       ECALLS = TOTENE

    ELSE
       CALL WRNDIE(1,'<TREKCOM>', &
            'Required options: READ,WRITe,INCRease,DECRease or ANALyse.')
    ENDIF

    CALL XTRANE(COMLYN,COMLEN,'TREKCOM')
    return
  end subroutine sub7030

  subroutine sub7050
    CALL W0(' ')
    CALL W0(' Initializing CPR. ')
    CALL W0(' ***************** ')

    INTERP = 3

    CALL DSUM2( XDUM,YDUM,ZDUM, &
         XBAS(IDXPRO)%a,YBAS(IDXPRO)%a,ZBAS(IDXPRO)%a, &
         -ONE ,XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM )
    REAPRO = SQRT( DSDOT2(XDUM,YDUM,ZDUM,NATOM) / DDIM )

    STEPSZ = REAPRO / 6

    LORIEN = .FALSE.
    DELTAS = -PT001*PT0001
    STPTOL = TENM5*TENM5
    BRASTP = PTONE
    FIRSTP = PT05
    MAXTOL = PT0001
    GRATOL = PT05
    ENETOL = PT001*PT0001
    BRAMAG = FIVE
    ATOMAX = PT75
    BRASCA = TWO
    NECALL = BIGINT
    LINMIN = MIN(IDIM-1, 2000)
    MODREM = 0
    LXEVAL = 8
    MODXIT = 3
    LHISAD = .FALSE.
    FRAME  = 10
    OSCTOL = ONEPT5*PTONE
    NTANGT = 3
    DDUM1 = NPOINT
    REDLUP = MAX( 4, NINT(SQRT(DDUM1)) )
    OSCMAX = 4
    PROJIN = TWO
    SADMIN = NINT( SQRT(DDIM/THREE) )
    SADGRD = PT05
    DISPLP = 0
    PRTOL1 = ONE
    PRTOL2 = THREE
    ININPP = NPOINT
    IDXSAD = 0
    LASTMV = 0
    TOTLUP = 0
    TOTOSC = 0
    CPLUP1 = 0
    CPLUP2 = 0
    OSSKIP = 2*FRAME
    LTMPRJ = .FALSE.
    ECALLS = 0
    MAXCAL = 0
    TOTUPD = 0
    TOTMIN = 0
    TOTCYC = 0
    LSTOT1 = 0
    LSTOT2 = 0
    INSMIN = 0
    RMVMIN = 0

    QSTART = .FALSE.
    CALL GTRMWA(COMLYN,COMLEN,'REST',4,STARNM,MXFLNM,STARLN)
    IF (STARLN > 0) THEN
       CALL W0(' ')
       NUNIT  = GTRMI(COMLYN,COMLEN,'UNIT',40)

       CALL ENCODI(NUNIT,CUNIT,6,IDUM1)
       CDUMY1 = 'READ  FORM UNIT '//CUNIT//' NAME '//STARNM(1:STARLN+1)
       IDUM1 = STARLN + LEN(CUNIT) + 25

#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          CALL OPNLGU( CDUMY1,IDUM1, NUNIT)
#if KEY_PARALLEL==1
       ENDIF
#endif 

       IF (NUNIT > 0)  QSTART = .TRUE.
    ENDIF


#if KEY_PARALLEL==1
    IF (QSTART.AND.(MYNOD == 0)) THEN
#else /**/
    IF (QSTART) THEN
#endif 
33     READ(NUNIT,'(A)') CDUMY1
       CALL W0( CDUMY1(1:79) )
       IF ( CDUMY1(1:1) == '*' )  GOTO  33

       READ(NUNIT,'(A)') CDUMY1
       CALL W0( CDUMY1(1:79) )

       READ(NUNIT,'(A)') CDUMY1
       CALL W0( CDUMY1(1:79) )

       READ(NUNIT,'(A)') CDUMY1
       CALL W0( CDUMY1(1:79) )
       READ(NUNIT,'(A)') CDUMY1
       READ(NUNIT,'(A)') CDUMY1
    ENDIF

#if KEY_PARALLEL==1
    CALL PSND8(SADGRD, 1)
    CALL PSND4(SADMIN, 1)
    CALL PSND8(STEPSZ, 1)
    CALL PSND4(REDLUP, 1)
    CALL PSND8(DELTAS, 1)
    CALL PSND4(LORIEN, 1)
    CALL PSND8(PRTOL1, 1)
    CALL PSND8(PRTOL2, 1)
#endif 

    STEPSZ = STEPSZ*SQRT(DDIM)
    DELTAS = DELTAS*SQRT(DDIM)
    STPTOL = STPTOL*SQRT(DDIM)
    BRASTP = BRASTP*SQRT(DDIM)
    PRTOL1 = PRTOL1/(DDIM-ONE)
    PRTOL2 = PRTOL2/(DDIM-ONE)


#if KEY_PARALLEL==1
    IF (QSTART.AND.(MYNOD == 0)) THEN
#else /**/
    IF (QSTART) THEN
#endif 
       
       READ(NUNIT,'(A)') CDUMY1

       IDXFOL = 1
       LDUM1 = .TRUE.
       DO L=1, NPOINT-1

          IDXFOL = IDNEXT(IDXFOL)
          IDX    = IDPREC(IDXFOL)
          IDUM1  = -1

          READ(NUNIT,*,END=177) IDUM1,IDUM2, DDUM1,DDUM2, &
               PTENE(IDXFOL),PTGRAD(IDXFOL),NLINMN(IDXFOL), &
               DDUM5, SEGSTP(IDX),LENSEG(IDX),STPMAX(IDX), &
               ENEMAX(IDX),NEXMIN(IDX),PREMIN(IDXFOL), &
               QUP(IDXFOL), QDOWN(IDX)


          NLINMN(IDXFOL) = REDLNM( NLINMN(IDXFOL), SADMIN, NCYCLE )

          SEGSTP(IDX) = SEGSTP(IDX)*SQRT(DDIM)
          LENSEG(IDX) = LENSEG(IDX)*SQRT(DDIM)

177       IF (IDUM1 < 0) THEN
             IF (LDUM1) THEN
                CALL W0(' ')
                CALL WRNDIE(0,'<TREKCOM>', &
                     'Prematurely reached EOF while reading restart-file!')
                CALL W0('The extra segments will be scanned.')
                CALL W0(' ')
                LDUM1 = .FALSE.
             ENDIF
             SEGSTP(IDX) = -ONE
          ENDIF

       ENDDO

       READ(NUNIT, '(A)', END=176)  CDUMY1
       CALL W0(' ')
       CALL WRNDIE(1,'<TREKCOM>', 'CPR restart-file is longer'// &
            ' than expected. Ignoring extra lines.' )
       CALL W0(' ')
176    CONTINUE

       CALL VCLOSE(NUNIT,'KEEP',LDUM1)

    ENDIF


#if KEY_PARALLEL==1
    CALL PSND4(NLINMN, NPOINT)
    CALL PSND4(STPMAX, NPOINT)
    CALL PSND4(NEXMIN, NPOINT)
    CALL PSND4(PREMIN, NPOINT)
    CALL PSND4(QUP,    NPOINT)
    CALL PSND4(QDOWN,  NPOINT)
    CALL PSND8(PTENE,  NPOINT)
    CALL PSND8(PTGRAD, NPOINT)
    CALL PSND8(SEGSTP, NPOINT)
    CALL PSND8(LENSEG, NPOINT)
    CALL PSND8(ENEMAX, NPOINT)
#endif 

    IF (REAPRO < RSMALL.AND.NPOINT < 3) THEN
       CALL W0(' ')
       CALL W0(' Identical reactant and product,')
       CALL W0(' but no intermediate path-points between them.')
       CALL W0(' Should have at least one intermediate point.')
       CALL W0(' ')
       CALL WRNDIE(-5,'<TREKCOM>','Cannot define initial path.')
    ENDIF

    return
  end subroutine sub7050

  subroutine sub7080

    LSCANP = (INDXA(COMLYN,COMLEN,'SCAN')  >  0)

    IDUM1  = INTERP
    INTERP = GTRMI(COMLYN,COMLEN,'INTE', INTERP )
    IF (INTERP >= 2)  THEN
       IF (INTERP /= IDUM1)  LSCANP = .TRUE.
       REDLUP = 0
    ENDIF

    OLDSTP = STEPSZ

    IDUM1 = GTRMI(COMLYN,COMLEN,'NGRI',  -1    )
    IF (IDUM1 > 0.AND.REAPRO > RSMALL) &
         STEPSZ = (REAPRO*SQRT(DDIM)) / (IDUM1+1)

    STEPSZ = GTRMF(COMLYN,COMLEN,'STEP', STEPSZ/SQRT(DDIM) )
    STEPSZ = STEPSZ*SQRT(DDIM)

    IF ( ABS(OLDSTP-STEPSZ)/STEPSZ  > RSMALL)  LSCANP = .TRUE.

    IF (STEPSZ < RSMALL) THEN
       CALL W0(' ')
       CALL WR( ' !!! STEPSZ is too small: STEPSZ = ', STEPSZ)
       IF (REAPRO < RSMALL) THEN
          CALL W0(' Identical reactant and product structures.')
          CALL W0(' Therefore, give value of STEPSZ explicitly.')
       ENDIF

       IF (INTERP >= 2)  THEN
          STEPSZ = RBIG
          CALL WRNDIE(1,'<TREKCOM>','STEPSZ will be ignored.')
       ELSE
          CALL WRNDIE(-5,'<TREKCOM>','STEPSZ cannot be near zero.')
       ENDIF

    ENDIF

    STEPLW = ZERO
    STEPLW = GTRMF(COMLYN,COMLEN,'STPL', STEPLW )
    STEPLW = STEPLW*SQRT(DDIM)

    DELTAS = GTRMF(COMLYN,COMLEN,'DELT', DELTAS/SQRT(DDIM) )
    STPTOL = GTRMF(COMLYN,COMLEN,'TOLS', STPTOL/SQRT(DDIM) )
    BRASTP = GTRMF(COMLYN,COMLEN,'BRAK', BRASTP/SQRT(DDIM) )
    DELTAS = DELTAS*SQRT(DDIM)
    STPTOL = STPTOL*SQRT(DDIM)
    BRASTP = BRASTP*SQRT(DDIM)

    MAXTOL = GTRMF(COMLYN,COMLEN,'TOLM', MAXTOL )
    GRATOL = GTRMF(COMLYN,COMLEN,'TOLG', GRATOL )
    ENETOL = GTRMF(COMLYN,COMLEN,'TOLE', ENETOL )
    BRAMAG = GTRMF(COMLYN,COMLEN,'BRKM', BRAMAG )
    BRASCA = GTRMF(COMLYN,COMLEN,'BRKS', BRASCA )
    LINMIN = GTRMI(COMLYN,COMLEN,'LINM', LINMIN )
    MODREM = GTRMI(COMLYN,COMLEN,'REMO', MODREM )
    LXEVAL = GTRMI(COMLYN,COMLEN,'LXEV', LXEVAL )
    MODXIT = GTRMI(COMLYN,COMLEN,'EXIT', MODXIT )
    LHISAD = (INDXA(COMLYN,COMLEN,'HIGH')  >  0)
    FRAME  = GTRMI(COMLYN,COMLEN,'FRAM', FRAME  )
    FRAME  = MIN(FRAME,INT(HISLEN/2))
    OSCTOL = GTRMF(COMLYN,COMLEN,'TOLO', OSCTOL )
    DISPLP = GTRMI(COMLYN,COMLEN,'DISP', DISPLP )

    OLDCYC = SADMIN
    OLDGRA = SADGRD
    OLDPR1 = PRTOL1
    OLDPR2 = PRTOL2
    PRTOL1 = PRTOL1*(DDIM-ONE)
    PRTOL2 = PRTOL2*(DDIM-ONE)
    IF (INDXA(COMLYN,COMLEN,'SADD')  >  0) THEN
       SADMIN =  MIN( IDIM - 1, 1000 )
       SADGRD =  PT001
       NTANGT =  6
       REDLUP =  0
       LORIEN = .FALSE.
       ATOMAX = ZERO
       DELTAS = ABS(DELTAS)
       IF ( ABS(PRTOL1-ONE) < RSMALL .OR. &
            ABS(PRTOL2-THREE) < RSMALL ) THEN
          PRTOL1 =  HALF
          PRTOL2 =  HALF
       ENDIF
    ENDIF

    IF (INDXA(COMLYN,COMLEN,'ORIE') > 0) LORIEN = .TRUE.
    IF (INDXA(COMLYN,COMLEN,'NOOR') > 0) LORIEN = .FALSE.
    IF (LSCAL.AND.LORIEN) THEN
       LORIEN = .FALSE.
       CALL WRNDIE(1,'<TREKCOM>', &
            'Cannot use coord.-scaling and coord.-orienting together.')
       CALL W0(' ')
       CALL W0('Coord.-orienting will remain disabled.')
    ENDIF

    SADMIN = GTRMI(COMLYN,COMLEN,'SADM', SADMIN )
    SADMIN = GTRMI(COMLYN,COMLEN,'SADC', SADMIN )

    SADGRD = GTRMF(COMLYN,COMLEN,'SADG', SADGRD )
    PRTOL1 = GTRMF(COMLYN,COMLEN,'TOL1', PRTOL1 )
    PRTOL2 = GTRMF(COMLYN,COMLEN,'TOL2', PRTOL2 )
    PRTOL1 = MIN( PRTOL1/(DDIM-ONE) , PT03 )
    PRTOL2 = MIN( PRTOL2/(DDIM-ONE) , PT09 )

    NTANGT = GTRMI(COMLYN,COMLEN,'NTAN', NTANGT )
    REDLUP = GTRMI(COMLYN,COMLEN,'LOOP', REDLUP )
    OSCMAX = GTRMI(COMLYN,COMLEN,'MAXO', OSCMAX )
    PROJIN = GTRMF(COMLYN,COMLEN,'PROJ', PROJIN )
    ATOMAX = GTRMF(COMLYN,COMLEN,'ATOM', ATOMAX )

    CALL XTRANE(COMLYN,COMLEN,'TREKCOM')

    CALL W0(' ')
    CALL W0(' Conjugate Peak Refinement parameters :')
    CALL W0(' ======================================')
    CALL W0(' Exit/Convergence criteria :')
    CALL W0(' ---------------------------')
    CALL WI(' Max. number of CPR-cycles        NCYCLE =',NCYCLE)
    CALL WI(' Max. number of new energy-calls  NECALL =',NECALL)
    CALL WL(' Stop refinem. when highest saddle found =',LHISAD)
    CALL W0(' ')
    CALL W0(' Saddle-point criteria :')
    CALL W0(' -----------------------')
    CALL WR(' Desired RMS Gradient at saddles  SADGRD =',SADGRD)
    CALL WI(' SADGRD satisfied for line-min.   SADMIN =',SADMIN)
    CALL WI('          Reminder :  3*(NATOM-NFIXED)-1 =',IDIM-1)
    CALL W0(' ')
    CALL W0(' Path-scanning (interpolation along path-segments) :')
    CALL W0(' ---------------------------------------------------')
    CALL WI(' Number of steps/segment (if >0), INTERP =',INTERP)
    CALL WR(' Largest allowed scanning-step,   STEPSZ =', &
         STEPSZ/SQRT(DDIM))
    CALL WR(' Lowest allowed scanning-step,    STEPLW =', &
         STEPLW/SQRT(DDIM))
    CALL W0(' ')
    CALL W0(' Path-relaxation :')
    CALL W0(' -----------------')
    CALL WR(' Finite diff. step for 2nd deriv. DELTAS =', &
         DELTAS/SQRT(DDIM))
    CALL WI(' Max. number of line-minim./cycle LINMIN =',LINMIN)
    CALL WR(' Grad. PRojection tol. when refin.,PRTOL1=', &
         PRTOL1*(DDIM-ONE))
    CALL WR(' Grad. PRojection tol. when adding,PRTOL2=', &
         PRTOL2*(DDIM-ONE))
    CALL WI(' Number of probes on path tangent NTANGT =',NTANGT)
    CALL WI(' Remove Mode (numb. of line min.) MODREM =',MODREM)
    CALL WL(' Re-orienting of modified path-points is =',LORIEN)
    CALL WI(' # of path-points displayed each cycle   =',DISPLP)
    CALL W0(' ')
    CALL W0(' Algorithm-looping prevention :')
    CALL W0(' ------------------------------')
    CALL WI(' Looping modulus of step REDuction REDLUP=',REDLUP)
    CALL WI(' FRAME-length for oscillation-detection  =',FRAME)
    CALL WR(' Energy/Gradient oscillation tol. OSCTOL =',OSCTOL)
    CALL WR(' PROJection-INcrease when oscill. PROJIN =',PROJIN)
    CALL WI(' Max. allowed nb. of oscillations OSCMAX =',OSCMAX)
    CALL W0(' ')
    CALL W0(' Line-extremization parameters :')
    CALL W0(' -------------------------------')
    CALL WR(' Maximal 1-D Braketing Step       BRASTP =', &
         BRASTP/SQRT(DDIM))
    CALL WR(' First   1-D Braketing Step       FIRSTP =',FIRSTP)
    CALL WR(' Dynamic braketing Scaling factor BRASCA =',BRASCA)
    CALL WR(' Braket Magnification factor      BRAMAG =',BRAMAG)
    CALL WR(' Toler. grad. cosin. 1-D maximiz. MAXTOL =',MAXTOL)
    CALL WR(' Toler. grad. cosin. 1-D minimiz. GRATOL =',GRATOL)
    CALL WR(' Smallest 1 dim. extremiz. Step   STPTOL =', &
         STPTOL/SQRT(DDIM))
    CALL WR(' Smallest fractional Ener. change ENETOL =',ENETOL)
    CALL WI(' Exit-Mode with respect to GRATOL MODXIT =',MODXIT)
    CALL WI(' Line eXtrem. energy-Evaluations  LXEVAL =',LXEVAL)
    CALL WR(' Max. allowed atomic displacement ATOMAX =',ATOMAX)

    IF (LCPR.AND..NOT.LSCANS) THEN

       IF (LSCANP) THEN
          CALL W0(' ')
          CALL W0(' Changed INTERP or STEPSZ, or asked to rescan.')

          LSCAN1 = .TRUE.
          LSCAN2 = .FALSE.
          LSCAN3 = .FALSE.
          CALL SCANPATH( NPOINT,IDXPRO,IDXSAD,SADMIN,SADGRD,NLINMN, &
               IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
               LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN, &
               PREMIN,SRTENE,SRTIDX,SEGSTP,PTGRAD, &
               XBAS,YBAS,ZBAS, NATOM,VERBEU)

          LSCAN1 = .FALSE.
          LSCAN2 = .FALSE.
          LSCAN3 = .TRUE.
       ENDIF

       IF (PRTOL1 > (1.5*OLDPR1).OR.PRTOL1 > (1.5*OLDPR2)) &
            OSSKIP = 2*FRAME

       IF (OLDCYC < SADMIN.OR.OLDGRA > SADGRD) THEN
          call sub7040
       ENDIF

    ENDIF

    return
  end subroutine sub7080

  subroutine sub7090

    CALL W0(' ')
    CALL W0(' Scanning path-segments.')
    CALL W0(' -----------------------')

    ORISTP = STEPSZ
    CALL COP3D(XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, X,Y,Z, NATOM)
    PTENE(1) = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
    PTGRAD(1) = SQRT( DSDOT2(DX,DY,DZ,NATOM)/DDIM )
    QUP(1) = .FALSE.
    PREMIN(1) = 0
    QDOWN(IDXPRO) = .FALSE.
    LENSEG(IDXPRO) = ZERO
    STPMAX(IDXPRO) = BIGINT
    NEXMIN(IDXPRO) = 0

    NMAXI  = 0

    IDXFOL = 1
    DO L=1, NPOINT-1

       IDXFOL = IDNEXT(IDXFOL)
       IDX    = IDPREC(IDXFOL)

       IF ( QSTART .AND. ABS(PTENE(IDXFOL)-PTENE(IDX))/ABS(PTENE(IDX)) &
            <  PT001*PT001 ) THEN
          SEGSTP(IDX) = -ONE
          CALL COP3D( XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, X,Y,Z, &
               NATOM)
          PTENE(IDX) = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
          PTGRAD(IDX) = SQRT( DSDOT2(DX,DY,DZ,NATOM)/DDIM )
       ENDIF

       IF (SEGSTP(IDX) <= ZERO.OR.LSCANP) THEN

          LSCAN1 = .TRUE.
          LSCAN2 = .TRUE.
          LSCAN3 = .TRUE.
          CALL SGSCAN( L,IDX,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
               LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
               SRTENE,SRTIDX,SEGSTP,PTGRAD, &
               XBAS,YBAS,ZBAS, NATOM,VERBEU)

       ELSE

          IF (QSTART.AND.VERBEU >= 1) THEN
             WRITE(OUTU,1040) ' Segment', L, &
                  'Scanning-data read from restart-file:  E=', PTENE(IDX)
1040         FORMAT(/, A, I5, 4X, A, F12.4, / )

          ELSEIF (VERBEU >= 1) THEN
             WRITE(OUTU,1040) ' Segment', L, &
                  'Scanning-data already available :      E=', PTENE(IDX)
          ENDIF

          IF ( STPMAX(IDX) < BIGINT ) THEN
             NMAXI = NMAXI + 1
             SRTENE(NMAXI) = ENEMAX(IDX)
             SRTIDX(NMAXI) = IDX
             IF (STPMAX(IDX) <= 0)  SRTENE(NMAXI) = PTENE(IDX)
          ENDIF

       ENDIF

       IF ( NLINMN(IDXFOL) < 0 ) THEN
          NPTSAD = NPTSAD + 1
          INPSAD(NPTSAD) = IDXFOL
       ENDIF

       IDX = IDNEXT(IDX)

    ENDDO


    IF (VERBEU >= 1)  WRITE(OUTU,1030)  IDXPRO, PTENE(IDXPRO)
1030 FORMAT(/,'  Path end-point Index',I5,3X,'(no interpolations),', &
         6X,'E=',F12.4)

    LSCAN1 = .FALSE.
    LSCAN2 = .FALSE.
    LSCAN3 = .TRUE.

    CALL DSORT1(SRTENE, SRTIDX, NMAXI, .FALSE.)

    DO L=1, NPTSAD
       IDX = INPSAD(L)

       IF (PTGRAD(IDX) <= SADGRD.AND.QUP(IDX).AND.QDOWN(IDX)) THEN

          IF (STPMAX(IDX) <= 0) THEN
             CALL GETREM(PTENE(IDX), SRTENE,SRTIDX, NMAXI,.FALSE.)

             IF (NLINMN(IDX) == 0)  NLINMN(IDX) = -3*IDIM

             IF (STPMAX(IDX) < 0) THEN
                NMAXI = NMAXI + 1
                SRTENE(NMAXI) = ENEMAX(IDX)
                SRTIDX(NMAXI) = IDX
                CALL ADDBOT( SRTENE, SRTIDX, NMAXI, .FALSE.)
             ENDIF

          ELSE

             IF (NLINMN(IDX) == 0)  NLINMN(IDX) = +3*IDIM

             NLINMN(IDX) = ABS( NLINMN(IDX) )
          ENDIF

          CALL W0(' ')
          CALL WI( ' Flagging as a saddle-point : IDX =', IDX)
       ELSE

          CALL W0(' ')
          CALL WI( ' Doesn`t qualify as saddle-point: IDX =', IDX)
          NLINMN(IDX) = ABS( NLINMN(IDX) )
          IF ( PTGRAD(IDX) > SADGRD )  NLINMN(IDX) = 0
       ENDIF

    ENDDO

    NPTSAD = 0

    DO I = 1,NMAXP
       DYNBRA(I) = FIRSTP
    ENDDO

    IDX = 1
    NEIBOR = 0
    DO I=2,NPOINT
       IDXPRE =     IDX
       IDX = IDNEXT(IDX)
       NEIBOR = NEIBOR + 1
       NEIBR1(NEIBOR) = SERIAL(IDXPRE)
       NEIBR2(NEIBOR) = SERIAL(   IDX)
    ENDDO

    CALL W0(' ')
    QSTART = .FALSE.

    return
  end subroutine sub7090

  subroutine sub7060

    TOTENE = ECALLS

    IF (NPOINT >= 2) THEN

       CALL W0(' ')
       CALL W0('Analysing the trajectory :')
       CALL W0('==========================')
       CALL W0( '  N    IDX   Lambda  X(N)-Xref    Energy'// &
            '    rms(Grad)  LINMN    Curvat  Grd/Prj')
       CALL W0( '----- ----- -------- -------- '// &
            '-------------- -------- -------  -------- -------')

       ANASCA = (INDXA(COMLYN,COMLEN,'SCAN')  >  0)
       ANALST = GTRMF(COMLYN,COMLEN,'STEP', STEPSZ/SQRT(DDIM) )
       ANALST = ANALST*SQRT(DDIM)
       IF (ANALST  <=  ZERO)  ANASCA = .FALSE.
       LARGST = -RBIG
       SMALST =  RBIG

       LAMDA = ZERO
       I = 1
       DO J=1,NPOINT
          IDXPRE = IDPREC(I)
          IDX    = I
          IDXFOL = IDNEXT(I)
          CALL COP3D(XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, X,Y,Z, NATOM)
          I = IDXFOL

          CALL DSUM2( XDUM,YDUM,ZDUM,X,Y,Z, -ONE,XREF,YREF,ZREF,NATOM)
          DDUM1 = SQRT( DSDOT2(XDUM,YDUM,ZDUM,NATOM) / DDIM )

          DDUM2 = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
          DDUM3 = SQRT( DSDOT1(DX,DY,DZ,NATOM) / DDIM )

          DDUM6 = ZERO
          DDUM7 = ZERO
          IF (IDX == 1 ) THEN
             CALL DSUM2( XS0,YS0,ZS0,XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, &
                  -ONE, X,Y,Z, NATOM)
             DDUM4 =SQRT( DSDOT1(XS0,YS0,ZS0,NATOM) )

          ELSEIF ( IDX /= IDXPRO ) THEN
             CALL DSUM2( XDUM,YDUM,ZDUM, &
                  XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, &
                  -ONE, X,Y,Z, NATOM)
             DDUM5 = SQRT( DSDOT1(XDUM,YDUM,ZDUM,NATOM) )

             DDUM6 = VANGLD( XS0,YS0,ZS0, DDUM4, XDUM,YDUM,ZDUM, &
                  DDUM5, NATOM)
             DDUM6 = DDUM6*TWO*SQRT(DDIM)/( DDUM4 + DDUM5 )

             CALL DSUM2( XS0,YS0,ZS0, &
                  XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
                  -ONE, &
                  XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
                  NATOM )
             DDUM4 =SQRT( DSDOT1(XS0,YS0,ZS0,NATOM) )

             DDUM7 = DDUM3*SQRT(DDIM)
             CALL PROJKT( DX,DY,DZ, DDUM7, XS0,YS0,ZS0,DDUM4, NATOM)
             DDUM7 = DDUM7/SQRT(DDIM)
             IF ( DDUM7 > ZERO )  DDUM7 = DDUM3/DDUM7

             CALL COP3D( XDUM,YDUM,ZDUM, XS0,YS0,ZS0, NATOM)
             DDUM4 = DDUM5
          ENDIF

          IDUM1 = IDX
          IF ( SERIAL(IDX) == IDX .AND. IDX <= ININPP )  IDUM1 = -IDX
          IDUM2 = PRTLNM( NLINMN(IDX) )
          IF (ABS(DDUM1) < TENM5)  DDUM1 = ZERO
          IF (ABS(DDUM2) < TENM8)  DDUM2 = ZERO
          IF (ABS(DDUM3) < TENM8)  DDUM3 = ZERO
          IF (ABS(DDUM6) < TENM8)  DDUM6 = ZERO
          IF (ABS(DDUM7) < TENM8)  DDUM7 = ZERO

#if KEY_PARALLEL==1
          IF (MYNOD == 0) THEN
#endif 
             WRITE(OUTU,1190) J,IDUM1,LAMDA/SQRT(DDIM),DDUM1,DDUM2, &
                  DDUM3,IDUM2,DDUM6, DDUM7
#if KEY_PARALLEL==1
          ENDIF
#endif 

          IF ( IDX /= IDXPRO ) THEN
             IF (ANASCA) THEN
                NTERPL = MAX( NINT(DDUM4/ANALST) , 1 )
                STPSIZ = DDUM4/NTERPL

                DO K = 1, NTERPL-1
                   JRN = ONE/NTERPL
                   CALL DSUM2( X,Y,Z,  X,Y,Z, JRN, XS0,YS0,ZS0, NATOM )
                   DDUM2 = ENERG( X,Y,Z, DX,DY,DZ, .TRUE.,.FALSE., ZERO)
                   DDUM3 = SQRT( DSDOT1(DX,DY,DZ,NATOM) / DDIM )

                   DDUM7 = DDUM3*SQRT(DDIM)
                   CALL PROJKT( DX,DY,DZ, DDUM7, XS0,YS0,ZS0,DDUM4, NATOM)
                   DDUM7 = DDUM7/SQRT(DDIM)

                   CALL DSUM2( XDUM,YDUM,ZDUM, &
                        X,Y,Z, -ONE, XREF,YREF,ZREF, NATOM)
                   DDUM1 = SQRT( DSDOT2(XDUM, YDUM,ZDUM,NATOM) / DDIM )

                   LAMDA = LAMDA + STPSIZ
#if KEY_PARALLEL==1
                   IF (MYNOD == 0) THEN
#endif 
                      WRITE(OUTU,1190)  K,IDX,LAMDA/SQRT(DDIM),DDUM1,DDUM2, &
                           DDUM3, 0, ZERO, DDUM3/DDUM7
#if KEY_PARALLEL==1
                   ENDIF
#endif 
                ENDDO

                LAMDA = LAMDA + STPSIZ

             ELSE
                LAMDA = LAMDA + DDUM4
             ENDIF

             IF (DDUM4 > LARGST) THEN
                LARGST = DDUM4
                ILARGS = IDX
             ENDIF
             IF (DDUM4 < SMALST) THEN
                SMALST = DDUM4
                ISMALS = IDX
             ENDIF

          ENDIF
       ENDDO

       CALL W0(' ')
       CALL W0( 'RMS-distance between adjacent path-points :')
       CALL W0( '-------------------------------------------')
       CALL WIR('Smallest: IDX & distance =', ISMALS,SMALST/SQRT(DDIM) )
       CALL WIR('Largest:  IDX & distance =', ILARGS,LARGST/SQRT(DDIM) )
       CALL WR( 'Average  distance =', LAMDA/(SQRT(DDIM)*(NPOINT-1)) )

1190   FORMAT(I5,1X,I5,1X,2(1PE8.3E1,1X),1PE14.8E1,1X,1PE8.3E1,1X,I7, &
            2X,1PE8.3E1,1X,1PE7.2E1)

    ENDIF

    ECALLS = TOTENE

    return
  end subroutine sub7060

  subroutine sub7100

    CALL XTRANE(COMLYN,COMLEN,'TREKCOM')
10  CALL RDCMND(FILENM,MXFLNM,FILLEN,ISTRM,EOF,.TRUE.,.TRUE.,'  ')
    IF (FILLEN  <=  1)                           GOTO  10
    IF ( INDX(FILENM,FILLEN,'DONE',4)  >  0 )   GOTO  11

    CALL ENCODI(NUNIT,CUNIT,6,IDUM1)
    COMLYN = 'READ  FORM UNIT '//CUNIT//' NAME '//FILENM(1:FILLEN+1)
    COMLEN = FILLEN + LEN(CUNIT) + 25
#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       CALL OPNLGU(COMLYN,COMLEN,NUNIT)
#if KEY_PARALLEL==1
    ENDIF
#endif 

    COMLYN = 'INIT CARD'
    COMLEN = 10
    CALL COORIO(-1,NUNIT,COMLYN(1:COMLEN+1),COMLEN,TITLEB,NTITLB, &
         I20DUM,NATOM, X,Y,Z,WMAIN,ATYPE, &
         RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.TRUE. )

    call sub7070

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       CALL VCLOSE(NUNIT,'KEEP',LDUM1)
#if KEY_PARALLEL==1
    ENDIF
#endif 

    GOTO  10
11  CONTINUE

    IDXPRO = NPOINT
    COMLYN = ' '
    COMLEN = 0

    return
  end subroutine sub7100

  subroutine sub7020
!!    CALL TRJSPC should be used in case more than one file is allowed
    BEGIN = GTRMI(COMLYN,COMLEN,'BEGI', 0 )
    STOP  = GTRMI(COMLYN,COMLEN,'STOP', 0 )
    SKIP  = GTRMI(COMLYN,COMLEN,'SKIP', 1 )

    CALL ENCODI(NUNIT,CUNIT,6,IDUM1)
    COMLYN = 'READ  UNFO UNIT '//CUNIT//' NAME '//FILENM(1:FILLEN+1)
    COMLEN = FILLEN + LEN(CUNIT) + 25
#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       CALL OPNLGU(COMLYN,COMLEN,NUNIT)
#if KEY_PARALLEL==1
    ENDIF
#endif 

    IODUM = ZERO
    NDEGF = 0
    NFILE = 0
    NSAVV = 0
    IDUM1 = NUNIT
    ISTEP  = 1

    ISTATS = 1
111 IF (ISTATS == -1)   GOTO  101

    CALL READCV( X,Y,Z, &
#if KEY_CHEQ==1
         CG,QCG,                                             & 
#endif
         RDTMP,NATOM,FREEAT,NFREAT,NUNIT,1,IDUM1, &
         NFILE,ISTEP,ISTATS,NDEGF,IODUM,BEGIN,STOP, &
         SKIP,NSAVV,'CORD','CORD',TITLEB,NTITLB,.FALSE., (/ ZERO /), .TRUE.)

    IF (ISTATS == 1) THEN
       CALL WRNDIE(-5,'<TREKCOM>','Reading more coord. than in file.')
       GOTO  101
    ENDIF

    call sub7070

    GOTO  111
101 CONTINUE

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       CALL VCLOSE(NUNIT,'KEEP',LDUM1)
#if KEY_PARALLEL==1
    ENDIF
#endif 

    IDXPRO = NPOINT
    COMLYN = ' '
    COMLEN = 0

    return
  end subroutine sub7020

  subroutine sub7070

    IF (.NOT.LREFER) THEN
       CALL W0(' ')
       CALL W0(' The first coordinate-set will be used as')
       CALL W0(' reference-set for coordinate orientation.')
       CALL COP3D( X,Y,Z, XREF,YREF,ZREF, NATOM)

       IF (ORIREF) THEN
          CALL W0(' - It will be oriented, as would COOR ORIENT.')
          CALL ORIENT( 0, XREF,YREF,ZREF,  XREF,YREF,ZREF, &
               NATOM,NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )
       ELSE
          CALL W0(' - Its orientation will NOT be changed.')
       ENDIF

       IF (QP1REF) THEN
          CALL W0(' - It will NOT be used as the 1st path-point.')
       ELSE
          CALL W0(' - It will ALSO be used as the 1st path-point.')
          LREFER = .TRUE.
       ENDIF
    ENDIF

    IF (LREFER) THEN
       IDXNEW = GETIDX( QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
            XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )

       IDXPRE = NPOINT - 1
       IDPREC(IDXNEW) = IDXPRE
       IDNEXT(IDXNEW) = 0
       IF (NPOINT  >  1)  THEN
          IDNEXT(IDXPRE) = IDXNEW
          SEGSTP(IDXPRE) = -ONE
          LENSEG(IDXPRE) =  ZERO
       ENDIF

       CALL COP3D( X,Y,Z, &
            XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
            NATOM)

       IF (LORIEN)  CALL ORIENT( IDXNEW,XBAS(IDXNEW)%a, &
            YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, XREF,YREF,ZREF, &
            NATOM,NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

       CALL STRECH( XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a,NATOM )

       IF (NFIXED > 0) THEN
          CALL CHKFIX( XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
               XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
               IDUM2 )
          IF (IDUM2 > 0) THEN
             CALL W0(' ')
             CALL WI(' !!! Warning : in path-point number =', NPOINT)
             CALL WI(' there were a number of fixed atoms =', IDUM2 )
             CALL W0(' whose coordinates had to be changed to those'// &
                  ' of the reactant !!!')
          ENDIF
       ENDIF

    ENDIF

    LREFER = .TRUE.

    return
  end subroutine sub7070

  subroutine sub7010

    WRD = NEXTA4(COMLYN,COMLEN)
    LCOMP = (INDXA(COMLYN,COMLEN,'COMP')  >  0)

    IF       (WRD ==  'INDE') THEN
       IDUM1 = NEXTI(COMLYN,COMLEN)
       IF (IDUM1 > 0 .AND. IDUM1 <= NPOINT+NFREEP) THEN
          CALL WI( ' The coordinates of path-point index =',IDUM1)
          IF (LCOMP) THEN
             CALL COP3D( XBAS(IDUM1)%a,YBAS(IDUM1)%a,ZBAS(IDUM1)%a, &
                  XCOMP,YCOMP,ZCOMP, NATOM )
             CALL UNSTRE( XCOMP,YCOMP,ZCOMP, NATOM )
             CALL W0(' have been copied to the comparison-set.')
          ELSE
             CALL COP3D( XBAS(IDUM1)%a,YBAS(IDUM1)%a,ZBAS(IDUM1)%a, &
                  XKEEP,YKEEP,ZKEEP, NATOM )
             CALL UNSTRE( XKEEP,YKEEP,ZKEEP, NATOM )
             CALL W0(' have been copied to the main-set.')
          ENDIF
       ELSE
          CALL W0(' This index number has not yet been used !')
       ENDIF

    ELSEIF (WRD ==  'ORDE') THEN
       IDUM1 = NEXTI(COMLYN,COMLEN)
       IDUM2 = PINDEX( IDUM1, NPOINT,IDNEXT)
       IF (IDUM2 > 0) THEN
          CALL WI( ' The coordinates of path-point index =',IDUM2)
          CALL WI( ' whose path-order position =',IDUM1)
          IF (LCOMP) THEN
             CALL COP3D( XBAS(IDUM2)%a,YBAS(IDUM2)%a,ZBAS(IDUM2)%a, &
                  XCOMP,YCOMP,ZCOMP, NATOM )
             CALL UNSTRE( XCOMP,YCOMP,ZCOMP, NATOM )
             CALL W0(' have been copied to the comparison-set.')
          ELSE
             CALL COP3D( XBAS(IDUM2)%a,YBAS(IDUM2)%a,ZBAS(IDUM2)%a, &
                  XKEEP,YKEEP,ZKEEP, NATOM )
             CALL UNSTRE( XKEEP,YKEEP,ZKEEP, NATOM )
             CALL W0(' have been copied to the main-set.')
          ENDIF
       ENDIF

    ELSEIF (WRD ==  'SADD') THEN
       IF (IDXSAD > 0) THEN
          CALL WI( ' The coordinates of path-point index =',IDXSAD)
          CALL W0(' which is the highest tentative saddle-point')
          IF (LCOMP) THEN
             CALL COP3D( &
                  XBAS(IDXSAD)%a,YBAS(IDXSAD)%a,ZBAS(IDXSAD)%a, &
                  XCOMP,YCOMP,ZCOMP, NATOM )
             CALL UNSTRE( XCOMP,YCOMP,ZCOMP, NATOM )
             CALL W0(' have been copied to the comparison-set.')
          ELSE
             CALL COP3D( &
                  XBAS(IDXSAD)%a,YBAS(IDXSAD)%a,ZBAS(IDXSAD)%a, &
                  XKEEP,YKEEP,ZKEEP, NATOM )
             CALL UNSTRE( XKEEP,YKEEP,ZKEEP, NATOM )
             CALL W0(' have been copied to the main-set.')
          ENDIF
       ELSE
          CALL W0(' No saddle-point has yet been found !')
       ENDIF

    ELSE
       CALL WRNDIE(1,'<TREKCOM>', &
            'Unknown keyword, expected is : INDEx, ORDered or HIGHest.')
    ENDIF

    CALL XTRANE(COMLYN,COMLEN,'TREKCOM')

    return
  end subroutine sub7010

  subroutine sub7040

    I = 1
    DO J=2, NPOINT-1

       I = IDNEXT(I)
       IF ( NLINMN(I) < 0 .AND. &
            (PTGRAD(I) > SADGRD.OR.ABS(NLINMN(I)) < SADMIN) ) THEN
          NLINMN(I) = ABS(NLINMN(I))
          IF (STPMAX(I) <= 0) THEN
             IF (STPMAX(I) < 0)  CALL GETREM( ENEMAX(I), SRTENE, &
                  SRTIDX, NMAXI,.FALSE.)
             NMAXI = NMAXI + 1
             SRTENE(NMAXI) = PTENE(I)
             SRTIDX(NMAXI) = I
             CALL ADDBOT( SRTENE, SRTIDX, NMAXI, .FALSE.)
          ENDIF
       ELSEIF (NLINMN(I) >= SADMIN .AND. &
            (QUP(I).AND.QDOWN(I).AND.PTGRAD(I) <= SADGRD) ) THEN
          NLINMN(I) = -ABS(NLINMN(I))
          IF (STPMAX(I) <= 0) THEN
             CALL GETREM( PTENE(I), SRTENE,SRTIDX, NMAXI,.FALSE.)
             IF (STPMAX(I) < 0) THEN
                NMAXI = NMAXI + 1
                SRTENE(NMAXI) = ENEMAX(I)
                SRTIDX(NMAXI) = I
                CALL ADDBOT( SRTENE, SRTIDX, NMAXI, .FALSE.)
             ENDIF
          ENDIF
       ENDIF

    ENDDO
    CALL SETIHI( IDXSAD,NPOINT,IDNEXT,PTGRAD,SADGRD,QUP,QDOWN, &
         NLINMN,SADMIN,PTENE )
    return
  end subroutine sub7040

end subroutine TREKCOM

#endif /* (travel_main)*/

end module travelmain


