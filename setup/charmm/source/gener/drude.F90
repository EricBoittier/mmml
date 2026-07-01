module drude
  use chm_kinds
  implicit none

contains

  SUBROUTINE DRUDE0(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     Process DRUDE command
    !-----------------------------------------------------------------------
    use dimens_fcm
    use bases_fcm
    use psf
    use stream
    use memory

    !     Passed variables
    CHARACTER(len=*) :: COMLYN
    INTEGER   COMLEN

    !     Local variables
    integer,allocatable,dimension(:) :: ISLCT
    integer,allocatable,dimension(:) :: SEGLST
    integer,allocatable,dimension(:) :: RESLST
    integer,allocatable,dimension(:) :: GRPLST
    integer,allocatable,dimension(:) :: ATMLST
    integer,allocatable,dimension(:) :: MAP
    integer,allocatable,dimension(:) :: INVMAP
    integer,allocatable,dimension(:) :: LINB
    integer,allocatable,dimension(:) :: LIBLO

    call chmalloc('drude.src','DRUDE0','ISLCT',MAXA,intg=ISLCT)
    call chmalloc('drude.src','DRUDE0','SEGLST',MAXA,intg=SEGLST)
    call chmalloc('drude.src','DRUDE0','RESLST',MAXA,intg=RESLST)
    call chmalloc('drude.src','DRUDE0','GRPLST',MAXA,intg=GRPLST)
    call chmalloc('drude.src','DRUDE0','ATMLST',MAXA,intg=ATMLST)
    call chmalloc('drude.src','DRUDE0','MAP',MAXA,intg=MAP)
    call chmalloc('drude.src','DRUDE0','INVMAP',MAXA,intg=INVMAP)
    call chmalloc('drude.src','DRUDE0','LINB',NNB,intg=LINB)
    call chmalloc('drude.src','DRUDE0','LIBLO',MAXA,intg=LIBLO)

    CALL DRUDE1(ISLCT, &
         SEGLST,RESLST,GRPLST, &
         ATMLST,MAP,INVMAP, &
         LINB,LIBLO)

    call chmdealloc('drude.src','DRUDE0','ISLCT',MAXA,intg=ISLCT)
    call chmdealloc('drude.src','DRUDE0','SEGLST',MAXA,intg=SEGLST)
    call chmdealloc('drude.src','DRUDE0','RESLST',MAXA,intg=RESLST)
    call chmdealloc('drude.src','DRUDE0','GRPLST',MAXA,intg=GRPLST)
    call chmdealloc('drude.src','DRUDE0','ATMLST',MAXA,intg=ATMLST)
    call chmdealloc('drude.src','DRUDE0','MAP',MAXA,intg=MAP)
    call chmdealloc('drude.src','DRUDE0','INVMAP',MAXA,intg=INVMAP)
    call chmdealloc('drude.src','DRUDE0','LINB',NNB,intg=LINB)
    call chmdealloc('drude.src','DRUDE0','LIBLO',MAXA,intg=LIBLO)

    !     Print out the structure file counters
    CALL PSFSUM(OUTU)

    RETURN
  END SUBROUTINE DRUDE0


  SUBROUTINE DRUDE1(ISLCT,SEGLST,RESLST,GRPLST,ATMLST, &
       MAP,INVMAP,LINB,LIBLO)
    !-----------------------------------------------------------------------
    !     DRUDE command
    !-----------------------------------------------------------------------
    use cnst_fcm, only: cnst_is_allocated, allocate_cnst
    use dimens_fcm
    use exfunc
    use number
    use consta
    use bases_fcm
    use psf
    use rtf,only:armass
    use coord
    use coordc
    use modpsf
    use select
    use stream
    use string
    use mmffm
    use comand
    use param
    use aniso_fcm
    use surface
    use chutil,only:getseg,atomid
#ifdef KEY_RESIZE
    use resize
#endif
    !     Passed variables
    INTEGER :: ISLCT(:)
    INTEGER :: SEGLST(:), RESLST(:), GRPLST(:), ATMLST(:)
    INTEGER :: MAP(:), INVMAP(:)
    INTEGER :: LINB(:), LIBLO(:)

    !     Local variables
    CHARACTER(len=8) SEGX,RESX,RESNX,TYPX
    CHARACTER(len=4) PTYP
    CHARACTER(len=4) WRD
    INTEGER   I1,I2,I3,I4, J1, INCB
    INTEGER   I, II
    INTEGER   J, JJ, K
    INTEGER   IUNIT
    INTEGER   NBONDX
    real(chm_real)    ALPHA, DMASS, KDRUDE, KDRUDE1
    real(chm_real)    RTEMP
    INTEGER   NNOTDRUDE
    LOGICAL   FOUND
    real(chm_real)    THOLEA

    !     Temporary mark for deleted atoms, bonds, ... in PSF:
    INTEGER, parameter :: MARK = -99999999
#ifdef KEY_RESIZE
    integer na0
#endif    

    if (natomt .gt. natom) then
        call wrndie(-5, '<DRUDE>', &
            'DRUDe command not supported with images. ' &
            // 'Change order of commands.')
    end if
    
    IF (INDXA(COMLYN,COMLEN,'RESE').GT.0) THEN
       IF (QDRUDE) THEN
          DO I = 1,NATOM
             IF (ISDRUDE(I)) THEN
                ISDRUDE(I) = .FALSE.
                !                 But back the mass and the charge on the heavy atom
                AMASS(I-1) = AMASS(I-1)+AMASS(I)
                AMASS(I) = ZERO
                CG(I-1) = CG(I-1)+CG(I)
                CG(I) = ZERO
             ENDIF
          ENDDO
          NDRUDE = 0
          NBDRUDE = 0
          !           Reset ANISO arrays (E. Harder 2005)
          NANISO = 0
          QDRUDE = .FALSE.
          !            QTHOLE = .FALSE.
          IF (PRNLEV.GT.2) THEN
             WRITE(OUTU,'(A,/)') &
                  ' DRUDE> Drude particles ready to be deleted'
             WRITE(OUTU,'(A,/)') &
                  '        Call: DELETE ATOMS SELECT TYPE D* END'
          ENDIF
       ELSE
          CALL WRNDIE(0,'<DRUDE>','No Drude particles to reset.')
       ENDIF
       RETURN
    ENDIF

    !
    ! Drude command needs ansio_fcm data arrays, make sure they are allocated. cb3
    if(.not.allocated(lstani1)) call allocate_aniso
    !
    !

    !     DUPLICATE flag
    !     Places the Drude particles on top of their corresponding
    !     heavy atoms
    IF (INDXA(COMLYN,COMLEN,'DUPL').GT.0) THEN
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       DO I = 1,NATOM
          IF (ISLCT(I).EQ.1 .AND. ISDRUDE(I)) THEN
             X(I) = X(I-1)
             Y(I) = Y(I-1)
             Z(I) = Z(I-1)
          ENDIF
       ENDDO
       RETURN
    ENDIF


    !     NOBUILD flag
    !     Labels Drude particles without building them
    !     (Provided for backward compatibility only)
    IF (INDXA(COMLYN,COMLEN,'NOBU').GT.0) THEN
       CALL WRNDIE(0,'<DRUDE>','NOBUILD FLAG: DO NOT USE')
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       NDRUDE = 0
       DO I = 1,NATOM
          IF (ISLCT(I).EQ.1) THEN
             IF (.NOT.ISDRUDE(I)) THEN
                NDRUDE = NDRUDE + 1
                ISDRUDE(I) = .TRUE.
             ENDIF
          ENDIF
       ENDDO
       RETURN
    ENDIF


    IF (PRNLEV.GT.2) THEN
       WRITE(OUTU,'(A,/)') ' DRUDE> CONSTRUCTION OF DRUDE PARTICLES'
    ENDIF

    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    NDRUDE = 0
    DO I = 1,NATOM
       IF (ISLCT(I).EQ.1) THEN
          NDRUDE = NDRUDE + 1
       ENDIF
    ENDDO
    IF (NDRUDE.EQ.0) THEN
       QDRUDE = .FALSE.
       !         QTHOLE = .FALSE.
       NBDRUDE = 0
       CALL WRNDIE(0,'<DRUDE>','ZERO ATOMS SELECTED')
       RETURN
    ENDIF

    QDRUDE = .TRUE.


    DMASS  = GTRMF(COMLYN,COMLEN,'MASS',ZERO)
    IF (DMASS.GT.ZERO) THEN
       IF (PRNLEV.GT.6) THEN
          WRITE(OUTU,'(2A)') ' DRUDE> The MASS parameter ', &
               'will override the topology file.'
       ENDIF
    ENDIF
    KDRUDE = GTRMF(COMLYN,COMLEN,'KDRU',ZERO)
    IF (KDRUDE.GT.ZERO) THEN
       IF (PRNLEV.GT.6) THEN
          WRITE(OUTU,'(2A)') ' DRUDE> The KDRUDE parameter ', &
               'will override the parameter file.'
       ENDIF
    ENDIF
    IUNIT  = GTRMI(COMLYN,COMLEN,'IPAR',-1)
    RTEMP=2.6D0
    THOLEA = GTRMF(COMLYN,COMLEN,'THOL',RTEMP)
    THOLEA = ABS(THOLEA)
    !      QTHOLE = .FALSE.
    !      IF (THOLEA.GT.ZERO) QTHOLE = .TRUE.

    THOLES = GTRMI(COMLYN,COMLEN,'SHAP',1)
    !     THOLES = 1: Slater-Delta
    !     THOLES = 2: Gaussian
    IF (THOLES.GT.2 .OR. THOLES.LT.1) THEN
       CALL WRNDIE(-3,'<DRUDE>','INVALID SHAPE')
    ENDIF
    IF (PRNLEV.GT.2) THEN
       !      IF (QTHOLE .AND. PRNLEV.GT.5) THEN
       WRITE(OUTU,'(A)') ' DRUDE> THOLE-TYPE DIPOLE SCREENING'
       WRITE(OUTU,'(8X,A,F10.6)') 'THOLE PARAMETER:',THOLEA
       IF (THOLES.EQ.1) WRITE(OUTU,'(8X,A,/)') &
            'SLATER-DELTA SHAPE:  S(u) = 1 - (1+u/2)*exp(-u)'
       IF (THOLES.EQ.2) WRITE(OUTU,'(8X,A,/)') &
            'GAUSSIAN SHAPE:  S(u) = erf(u)'
    ENDIF
    IF (THOLES.EQ.2) THEN
       CALL WRNDIE(-3,'<DRUDE>','UNAVAILABLE SHAPE')
    ENDIF

    !     Fill segment list SEGLST with values corresonding to current PSF.
    DO I=1,NSEG
       DO J=NICTOT(I)+1,NICTOT(I+1)
          DO K=IBASE(J)+1,IBASE(J+1)
             SEGLST(K)=I
          ENDDO
       ENDDO
    ENDDO

    !     Fill residue list RESLST with values corresponding to current PSF.
    DO I=1,NRES
       DO J=IBASE(I)+1,IBASE(I+1)
          RESLST(J)=I
       ENDDO
    ENDDO

    !     Fill group list GRPLST with values corresponding to current PSF.
    DO I=1,NGRP
       DO J=IGPBS(I)+1,IGPBS(I+1)
          GRPLST(J)=I*10 
          ATMLST(J)=J*10 ! allow NDRUDE to be inserted between existing atoms
       ENDDO
    ENDDO

    NDRUDE = 0
#ifdef KEY_RESIZE
    na0=natom
#endif
    DO I = 1,NRES
       DO J = IBASE(I)+1,IBASE(I+1)
          RESLST(J) = I
          IF (ISLCT(J).EQ.1) THEN

             CALL ATOMID(J,SEGX,RESX,RESNX,TYPX)
             PTYP = 'D'//TYPX(1:3)
             IF (PRNLEV.GT.5) THEN
                WRITE(OUTU,'(8A,1X,F9.6)') &
                     'Adding Drude particle ',PTYP, &
                     ' on ',TYPX(1:idleng),SEGX(1:idleng),' ', &
                     RESX(1:idleng),RESNX(1:idleng),WMAIN(J)
             ENDIF
             NDRUDE = NDRUDE+1

             IF(NATOM+1.GT.MAXA) THEN
#ifndef KEY_RESIZE
                WRITE(OUTU,*) MAXA,NATOM
                CALL WRNDIE(-4,'<DRUDE>', &
                     'MAXIMUM NUMBER OF ATOMS EXCEEDED')
#endif                
             ELSE
                NATOM = NATOM+1
#ifdef KEY_RESIZE
                if (natom .gt. na0) then
                   na0=natom+drsz0
                   call resize_psf('drude.F90','DRUDE1','NATOM',na0, .true.)
                   call resize_coord('drude.F90','DRUDE1',na0, .true.)
                   call resize_coordc('drude.F90','DRUDE1',na0, .true.)
                   call resize_deriv('drude.F90','DRUDE1',na0, .true.)
                endif
#endif                
                FOUND = .FALSE.
                DO II = 1,NATC
                   IF (PTYP.EQ.ATC(II)) THEN
                      IAC(NATOM) = II
                      FOUND = .TRUE.
                   ENDIF
                ENDDO
                IF (.NOT.FOUND) THEN
                   WRITE(OUTU,*) PTYP
                   CALL WRNDIE(-3,'<DRUDE>', &
                        'ATOM TYPE UNKOWN FOR DRUDE')
                ENDIF
                ALPHA = WMAIN(J)
                IF (ALPHA.EQ.ZERO) THEN
                   CALL WRNDIE(-3,'<DRUDE>','ZERO POLARIZABILITY')
                ENDIF

                !                 Look for MASS
                IF (DMASS.EQ.ZERO) THEN
                   AMASS(NATOM) = ARMASS(IAC(NATOM))
                ELSE
                   AMASS(NATOM) = DMASS
                ENDIF
                IF (PRNLEV.GT.5) THEN
                   WRITE(OUTU,'(4X,A,F12.6)') &
                        'Using MASS   = ',AMASS(NATOM)
                ENDIF

                !                 Look for KDRUDE
                FOUND = .FALSE.
                DO INCB = 1,NCB
                   I1 = SQRT(TWO*KCB(INCB))+HALF
                   J1 = KCB(INCB)-I1*(I1-1)/2
                   IF ( (ATC(IAC(J)).EQ.ATC(I1) .AND. &
                        ATC(IAC(NATOM)).EQ.ATC(J1)) .OR. &
                        (ATC(IAC(J)).EQ.ATC(J1).AND. &
                        ATC(IAC(NATOM)).EQ.ATC(I1)) ) THEN
                      IF (KDRUDE.NE.ZERO) THEN
                         CBC(INCB) = KDRUDE
                      ENDIF
                      IF (PRNLEV.GT.5) THEN
                         WRITE(OUTU,'(4X,A,F12.6)') &
                              'Using KDRUDE = ',CBC(INCB)
                      ENDIF
                      FOUND = .TRUE.
                      IF (PRNLEV.GT.5) THEN
                         WRITE(OUTU,70) INCB,ATC(I1),ATC(J1), &
                              KCB(INCB),CBC(INCB),CBB(INCB)
70                       FORMAT(6X,I4,2X,A4,' - ',A4,3X,I7,3F10.3)
                      ENDIF
                      IF (CBC(INCB).EQ.ZERO) THEN
                         CALL WRNDIE(-3,'<DRUDE>', &
                              'BOND FORCE CONSTANT ZERO')
                      ENDIF
                      KDRUDE1 = CBC(INCB)
                   ENDIF
                ENDDO
                IF (.NOT.FOUND) THEN
                   WRITE(OUTU,*) ATC(IAC(J)), ATC(IAC(NATOM))
                   CALL WRNDIE(-3,'<DRUDE>','UNDEFINED BOND')
                ENDIF

                !                 Set charge
                CG(NATOM) = int(SQRT(2*ABS(ALPHA)*KDRUDE1/CCELEC)*10000.0 + 0.5)/10000.0
                !              CG(NATOM) = SQRT(2*ABS(ALPHA)*KDRUDE1/CCELEC)
                CG(NATOM) = SIGN(CG(NATOM),ALPHA) ! copy sign of alpha


                ATYPE(NATOM) = PTYP
                RSCLF(NATOM) = ONE
#if KEY_WCA==1
                WCA(NATOM) = ONE         
#endif
                IMOVE(NATOM) = 0
                RESLST(NATOM) = I
                SEGLST(NATOM) = GETSEG(I,NICTOT,NSEG)
                IBLO(NATOM) = NNB
                ! No explicit nonbonded exclusions are
                ! allowed for this atom for now.
                ATMLST(NATOM) = ATMLST(J)+1
                GRPLST(NATOM) = GRPLST(J)
                ! Assign to the same group number as
                ! heavy atom to which it belongs
                X(NATOM) = X(J)
                Y(NATOM) = Y(J)
                Z(NATOM) = Z(J)
                !                 Modify mass and charge of heavy atom
                AMASS(J) = AMASS(J)-AMASS(NATOM)
                IF (AMASS(J).LE.ZERO) THEN
                   CALL WRNDIE(-3,'<DRUDE>', &
                        'DRUDE MASS LARGER THAN TOTAL MASS OF ATOM')
                ENDIF
                CG(J) = CG(J)-CG(NATOM)

                IF ((IUNIT.GT.0) .AND. (IOLEV.GE.0)) THEN
                   WRITE(IUNIT,100) ATC(IAC(J)),ATC(IAC(NATOM)), &
                        KDRUDE1,ZERO,' ! I1 -- I2 BOND between ', &
                        ATYPE(J)(1:idleng),ATYPE(NATOM)(1:idleng)
                ENDIF

             ENDIF
          ENDIF ! IF (ISLCT(J).EQ.1)
       ENDDO
    ENDDO
#ifdef KEY_RESIZE
    if (natom .lt. na0) then
       call resize_psf('drude.F90','DRUDE1','NATOM',natom, .true.)
       call resize_coord('drude.F90','DRUDE1',natom, .true.)
       call resize_coordc('drude.F90','DRUDE1',natom, .true.)
       call resize_deriv('drude.F90','DRUDE1',natom, .true.)
    endif
#endif
    !     Now define a map as well as an inverse map to map the atom numbers.
    !     First, sort GRPLST, using quicksort (SORTP conserves the order of
    !     equal elements, this is essential at this point!)
    !
    CALL SORTP(NATOM,INVMAP,ORDER,ATMLST,1,0,0,0,0,0,0)

    DO I = 1,NATOM
       MAP(INVMAP(I)) = I
       ISDRUDE(I) = .FALSE.
    ENDDO

    !     Complete mapping of PSF, INTCR, COORD, COORDC, CNST
    !
    if (cnst_is_allocated) call allocate_cnst(natom)

    CALL MAPIC(MAP,INVMAP,SEGLST,RESLST,GRPLST,LINB,LIBLO, &
         MARK,.FALSE.)

    IF(PRNLEV.GT.2) THEN
       WRITE(OUTU,'(20X,I4,1X,A)') NDRUDE,' Drude particles were added.'
    ENDIF

    IF (PRNLEV.GT.6) THEN
       WRITE(OUTU,*)
       DO I = 1,NGRP
          WRITE(OUTU,'(A)') 'GROUP'
          DO J = IGPBS(I)+1,IGPBS(I+1)
             CALL ATOMID(J,SEGX,RESX,RESNX,TYPX)
             WRITE(OUTU,'(I4,3(1X,A,1X))') J,SEGX(1:idleng), &
                  RESX(1:idleng),TYPX(1:idleng)
          ENDDO
       ENDDO
    ENDIF

    !     Put in the bonds

    WRITE(OUTU,*)
    NBONDX = NBOND
    J = NATOM-NDRUDE
#ifdef KEY_RESIZE
    na0=nbond
#endif    
    DO I = 1,NATOM-NDRUDE
       IF (ISLCT(I).EQ.1) THEN
          J = J+1
          NBOND = NBOND+1
#ifdef KEY_RESIZE
          if (nbond .gt. na0) then
             na0=nbond+drsz0
             call resize_psf('drude.F90','DRUDE1','NBOND',na0, .true.)
          endif
#endif               
          IB(NBOND) = MAP(I)
          JB(NBOND) = MAP(J)
          ISDRUDE(MAP(J)) = .TRUE.
       ENDIF
    ENDDO
    NBDRUDE = NBOND ! Keep in memory for subroutine CODES

    !     Find the 1-2-3-4 exclusions pairs involving drudes
    !     and set all possible bonds

    DO I = NBONDX+1,NBONDX+NDRUDE
       I1 = JB(I)
       I2 = IB(I)
       I3 = 0
       I4 = 0
       DO J = 1,NBONDX

          IF((IB(I).EQ.IB(J)))THEN
             I3 = JB(J)
             NBOND = NBOND+1
#ifdef KEY_RESIZE
          if (nbond .gt. na0) then
             na0=nbond+drsz0
             call resize_psf('drude.F90','DRUDE1','NBOND',na0, .true.)
          endif
#endif               
             IB(NBOND) = I1   !ici
             JB(NBOND) = I3   !ici
             IF (IUNIT.GT.0) THEN
                WRITE(IUNIT,100) ATC(IAC(I1)),ATC(IAC(I3)),ZERO,ZERO, &
                     ' ! I1 -- I3 BOND between ', &
                     ATYPE(I1)(1:idleng),ATYPE(I3)(1:idleng)
             ENDIF
          ELSEIF(IB(I).EQ.JB(J))THEN
             I3 = IB(J)
             NBOND = NBOND+1
#ifdef KEY_RESIZE
          if (nbond .gt. na0) then
             na0=nbond+drsz0
             call resize_psf('drude.F90','DRUDE1','NBOND',na0, .true.)
          endif
#endif               
             IB(NBOND) = I1   !ici
             JB(NBOND) = I3   !ici
             IF (IUNIT.GT.0) THEN
                WRITE(IUNIT,100) ATC(IAC(I1)),ATC(IAC(I3)),ZERO,ZERO, &
                     ' ! I1 -- I3 BOND between ', &
                     ATYPE(I1)(1:idleng),ATYPE(I3)(1:idleng)
             ENDIF
          ENDIF

          DO K = NBONDX+1,NBONDX+NDRUDE
             IF( ((IB(I).EQ.IB(J)).AND.(JB(J).EQ.IB(K))) .OR. &
                  ((IB(I).EQ.JB(J)).AND.(IB(J).EQ.IB(K))) )THEN
                IF (K.GT.I) THEN
                   I4 = JB(K)
                   NBOND = NBOND+1
#ifdef KEY_RESIZE
          if (nbond .gt. na0) then
             na0=nbond+drsz0
             call resize_psf('drude.F90','DRUDE1','NBOND',na0, .true.)
          endif
#endif               
                   IB(NBOND) = I1 !ici
                   JB(NBOND) = I4 !ici
                   IF (IUNIT.GT.0) THEN
                      WRITE(IUNIT,100) ATC(IAC(I1)),ATC(IAC(I4)), &
                           ZERO,ZERO,' ! I1 -- I4 BOND between ', &
                           ATYPE(I1)(1:idleng),ATYPE(I4)(1:idleng)
                   ENDIF
                ENDIF
             ENDIF
          ENDDO

       ENDDO
    ENDDO
100 FORMAT(A,2X,A,2F10.3,4X,3(1X,A))
#ifdef KEY_RESIZE
    if (nbond .lt. na0) &
         call resize_psf('drude.F90','DRUDE1','NBOND',nbond, .true.)
#endif 


    !     NOLABEL flag
    !     Builds Drude particles without labelling them
    !     (We are actually unlabelling them...)
    !     (Provided for backward compatibility only)
    NNOTDRUDE = 0
    IF (INDXA(COMLYN, COMLEN, 'NOLA').GT.0) THEN
       CALL WRNDIE(0,'<DRUDE>','NOLABEL FLAG: DO NOT USE')
       J = NATOM-NDRUDE
       DO I = 1,NATOM-NDRUDE
          IF (ISLCT(I).EQ.1) THEN
             J = J+1
             IF (ISDRUDE(MAP(J))) THEN
                NNOTDRUDE = NNOTDRUDE+1
                ISDRUDE(MAP(J)) = .FALSE.
             ENDIF
          ENDIF
       ENDDO
    ENDIF
    NDRUDE = NDRUDE-NNOTDRUDE

    !     Fill ALPHADP, THOLEI
    DO I = 1,NATOM
       ALPHADP(I) = ZERO
       IF (ISDRUDE(I).AND.THOLEA.GT.ZERO) THEN
          IF (WCOMP(I-1).EQ.ZERO) THEN
             IF (ATC(IAC(I)).NE.'DOH2') THEN
                WRITE(OUTU,*) 'Warning from DRUDE: ZERO THOLEI on ', &
                     ATC(IAC(I-1)), (I-1)
                CALL WRNDIE(5,'<DRUDE>','ZERO THOLEI')
             ENDIF
          ENDIF
          THOLEI(I-1) = WCOMP(I-1)
          ALPHADP(I-1) = WMAIN(I-1)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE DRUDE1

  SUBROUTINE EHYPER(NATOM,ISDRUDE,HORDER, &
       KHYPER,RHYPER,EHYPERPOL,X,Y,Z,DX,DY,DZ)
    !     Computes hyper polarizability energy and forces for Drude
    !     no second derivative yet.
    use dimens_fcm
    use number
    use consta
    use stream ! for OUTU,PRNLEV
#if KEY_BLOCK==1
    use block_fcm
    use pert
#endif /*  BLOCK*/
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, natoml, atoml
#endif
    !     Arguments
    INTEGER NATOM
    LOGICAL ISDRUDE(*)
    REAL(chm_real) EHYPERPOL
    REAL(chm_real) X(*),Y(*),Z(*)
    REAL(chm_real) RX,RY,RZ,R2,R
    REAL(chm_real) DUDR
    REAL(chm_real) DX(*),DY(*),DZ(*)
    INTEGER HORDER
    REAL(chm_real) KHYPER,RHYPER
    INTEGER I
    integer ia, iend
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real) COEF
#endif /*  BLOCK*/


#if KEY_DOMDEC==1
    if (q_domdec) then
       iend = natoml
    else
#endif
       iend = natom
#if KEY_DOMDEC==1
    endif
#endif

    do ia=1,iend
#if KEY_DOMDEC==1
       if (q_domdec) then
          i = atoml(ia)
       else
#endif
          i = ia
#if KEY_DOMDEC==1
       endif
#endif      
       IF(ISDRUDE(I)) THEN
          RX=X(I)-X(I-1)
          RY=Y(I)-Y(I-1)
          RZ=Z(I)-Z(I-1)
          R2=RX*RX+RY*RY+RZ*RZ
          R=SQRT(R2)
          IF(R.GT.RHYPER)THEN

#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
             IBL = IBLCKP(I)
             JBL = IBLCKP(I-1)
             IF (JBL.LT.IBL) THEN
                KK=JBL
                JBL=IBL
                IBL=KK
             ENDIF
             KK=IBL+JBL*(JBL-1)/2
             COEF = BLCOEB(KK)
             IF(QNOBO) COEF = 1.0
             DUDR=COEF*KHYPER*HORDER*(R-RHYPER)**(HORDER-1)/R
             EHYPERPOL=EHYPERPOL+COEF*KHYPER*(R-RHYPER)**HORDER
       ELSE
#endif /*  BLOCK*/
             DUDR=KHYPER*HORDER*(R-RHYPER)**(HORDER-1)/R
             EHYPERPOL=EHYPERPOL+KHYPER*(R-RHYPER)**HORDER
#if KEY_BLOCK==1 /*blocker*/
       ENDIF
#endif /*  BLOCK*/

             DX(I)=DX(I)+DUDR*RX
             DY(I)=DY(I)+DUDR*RY
             DZ(I)=DZ(I)+DUDR*RZ
             DX(I-1)=DX(I-1)-DUDR*RX
             DY(I-1)=DY(I-1)-DUDR*RY
             DZ(I-1)=DZ(I-1)-DUDR*RZ
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE EHYPER
  !
  SUBROUTINE ETHOLE(NATOM, &
       ISDRUDE,ALPHADP,THOLEI,CG,  &
       X,Y,Z, I,J, VALID, EN, &
       DFAA,DFDA,DFAD,DFDD, &
       SECOND,DDFAA,DDFDA,DDFAD,DDFDD,COUNTER )
    !     
    !     Computes Thole energy and forces for the (I,J) pair.
    !     Can also compute second derivatives.
    !     
    !     EN = U
    !     DF = (dU/dR)/R
    !     DDF = (d2U/dR2)/R^2
    !     
    use dimens_fcm
    use number
    use consta
    use stream ! for OUTU,PRNLEV
    use pert 

    !     Arguments
    INTEGER NATOM
    LOGICAL ISDRUDE(*)
    real(chm_real)  ALPHADP(*),THOLEI(*)
    real(chm_real)  CG(*)             ! INPUT   Charges
    real(chm_real)  X(*),Y(*),Z(*)    ! INPUT   Atomic positions
    INTEGER I,J               ! INPUT   Indices of the two particles
    LOGICAL VALID             ! OUTPUT  TRUE if (I,J) pair contributes
    real(chm_real)  EN                ! OUTPUT  Thole energy
    real(chm_real)  DFAA              ! OUTPUT  Atom-Atom
    real(chm_real)  DFDA              ! OUTPUT  Drude-Atom
    real(chm_real)  DFAD              ! OUTPUT  Atom-Drude
    real(chm_real)  DFDD              ! OUTPUT  Drude-Drude
    LOGICAL SECOND            ! INPUT   TRUE to compute the 2nd derivative
    real(chm_real)  DDFAA             ! OUTPUT  Atom-Atom
    real(chm_real)  DDFDA             ! OUTPUT  Drude-Atom
    real(chm_real)  DDFAD             ! OUTPUT  Atom-Drude
    real(chm_real)  DDFDD             ! OUTPUT  Drude-Drude
    INTEGER COUNTER

    !     Locals
    real(chm_real)  QD1,QD2,QQ, ALPHA1,ALPHA2,AA
    real(chm_real)  RX,RY,RZ,R,AU,EXPAU,HFAUP1,DF

    VALID = .FALSE.
    EN = ZERO

    !      IF (QTHOLE) THEN
    IF (I.LT.NATOM .AND. J.LT.NATOM &
         .AND. ISDRUDE(I+1) .AND. ISDRUDE(J+1)) THEN
       VALID = .TRUE.
       ALPHA1 = ALPHADP(I)
       ALPHA2 = ALPHADP(J)
       AA = ALPHA1*ALPHA2/(THOLEI(I) + THOLEI(J))**6
       QD1 = CG(I+1)
       QD2 = CG(J+1)

       QQ = QD1*QD2 * CCELEC
       IF (PRNLEV.GT.7) WRITE(OUTU,'(A,2I5,4F11.7)')'ETHOLE ', &
            I,J,ALPHA1,ALPHA2,QD1,QD2

       !           Atom-Atom
       RX = X(I)-X(J)
       RY = Y(I)-Y(J)
       RZ = Z(I)-Z(J)
       EN = EN + E_THOLE(QQ,AA,RX,RY,RZ, DFAA,SECOND,DDFAA)
       !           Atom-Drude
       RX = X(I)-X(J+1)
       RY = Y(I)-Y(J+1)
       RZ = Z(I)-Z(J+1)
       EN = EN + E_THOLE(-QQ,AA,RX,RY,RZ, DFAD,SECOND,DDFAD)
       !           Drude-Atom
       RX = X(I+1)-X(J)
       RY = Y(I+1)-Y(J)
       RZ = Z(I+1)-Z(J)
       EN = EN + E_THOLE(-QQ,AA,RX,RY,RZ, DFDA,SECOND,DDFDA)
       !           Drude-Drude
       RX = X(I+1)-X(J+1)
       RY = Y(I+1)-Y(J+1)
       RZ = Z(I+1)-Z(J+1)
       EN = EN + E_THOLE(QQ,AA,RX,RY,RZ, DFDD,SECOND,DDFDD)
    ENDIF
    !      ENDIF
    RETURN
  END SUBROUTINE ETHOLE

  SUBROUTINE ETHOLE2(I,J,X,Y,Z,DX,DY,DZ,ISDRUDE,ALPHADP, &
       THOLEIJ,CG,ENTHOLE)
    !
    !     Computes Thole energy and forces for the (I,J) pair.
    !     Changes for shielded divalent ions (Benoit Roux, Dec 15 2007)
    !
    !     ENTHOLE = U
    !     DF = (dU/dR)/R
    !
    use dimens_fcm
    use number
    use consta
#if KEY_BLOCK==1
    use block_fcm
    use pert
#endif /*  BLOCK*/

    !     Arguments
    INTEGER I,J               ! INPUT   Indices of the two particles
    real(chm_real)  X(*),Y(*),Z(*)    ! INPUT   Atomic positions
    real(chm_real)  DX(*),DY(*),DZ(*) ! INPUT/OUTPUT
    LOGICAL ISDRUDE(*)
    real(chm_real)  ALPHADP(*),THOLEIJ
    real(chm_real)  CG(*)             ! INPUT   Charges
    real(chm_real)  ENTHOLE           ! OUTPUT  Thole energy

    !     Locals
    real(chm_real)  DFAA              ! Atom-Atom
    real(chm_real)  DFDA              ! Drude-Atom
    real(chm_real)  DFAD              ! Atom-Drude
    real(chm_real)  DFDD              ! Drude-Drude
    real(chm_real)  QD1,QD2,QQ, ALPHA1,ALPHA2,AA
    real(chm_real)  RX,RY,RZ,R,AU,EXPAU,NORM,POLYAU1,POLYAU2
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real) COEF
#endif /*  BLOCK*/

    ENTHOLE = ZERO

    ! Thole-shielding (the direct QQ/R coulomb is already included elsewhere...)
    ! We want:
    !  U        = (-QQ/R)   + QQ/R   [1 - (1+AU/2)exp(-AU)]      = QQ/R   [- (1+AU/2)exp(-AU)]
    ! (dU/dR)/R = (+QQ/R^3) + QQ/R^3 [(1+AU+AU^2/2)exp(-AU) - 1] = QQ/R^3 [(1+AU+AU^2/2)exp(-AU)]

    ! Default Thole:
    !           AA = ALPHADP(I)*ALPHADP(J)/(THOLEI(I) + THOLEI(J))**6

    ! Pair-specific Thole:
    AA = ALPHADP(I)*ALPHADP(J)/(THOLEIJ)**6

    NORM = AA**(-SIXTH)

    !           write(*,*)
    !    &      'ETHOLE2 ','atom ij',I,J,' CG=',cg(i),cg(i+1),cg(j),cg(j+1),
    !    &      'ALPHA=',ALPHADP(I),ALPHADP(J),' THOLEIJ=',THOLEIJ,AA

#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             IF (JBL.LT.IBL) THEN
                KK=JBL
                JBL=IBL
                IBL=KK
             ENDIF
             KK=IBL+JBL*(JBL-1)/2
             COEF = BLCOEE(KK)
       ENDIF
#endif /*  BLOCK*/

    !           Atom-Atom
    QQ = CG(I)*CG(J) * CCELEC
    RX = X(I)-X(J)
    RY = Y(I)-Y(J)
    RZ = Z(I)-Z(J)
    R = SQRT(RX*RX+RY*RY+RZ*RZ)
    AU = NORM*R
    EXPAU = EXP(-AU)
    POLYAU1 = ONE+AU*HALF
    POLYAU2 = ONE+AU*(ONE+HALF*AU)
#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
            ENTHOLE = ENTHOLE + COEF * QQ/R * (-POLYAU1*EXPAU)
            DFAA = COEF * QQ/R**3 * (POLYAU2*EXPAU)
       ELSE
#endif /*  BLOCK*/
    ENTHOLE = ENTHOLE + QQ/R * (-POLYAU1*EXPAU)
    !           write(*,*) 'atom-atom ',r,QQ/R * (-POLYAU1*EXPAU)
    DFAA = QQ/R**3 * (POLYAU2*EXPAU)
#if KEY_BLOCK==1 /*blocker*/
       ENDIF
#endif /*  BLOCK*/
    DX(I)=DX(I)+RX*DFAA
    DY(I)=DY(I)+RY*DFAA
    DZ(I)=DZ(I)+RZ*DFAA
    DX(J)=DX(J)-RX*DFAA
    DY(J)=DY(J)-RY*DFAA
    DZ(J)=DZ(J)-RZ*DFAA

    !           Atom-Drude
    QQ = CG(I)*CG(J+1) * CCELEC
    RX = X(I)-X(J+1)
    RY = Y(I)-Y(J+1)
    RZ = Z(I)-Z(J+1)
    R = SQRT(RX*RX+RY*RY+RZ*RZ)
    AU = NORM*R
    EXPAU = EXP(-AU)
    POLYAU1 = ONE+AU*HALF
    POLYAU2 = ONE+AU*(ONE+HALF*AU)
#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
            ENTHOLE = ENTHOLE + COEF * QQ/R * (-POLYAU1*EXPAU)
            DFAD = COEF * QQ/R**3 * (POLYAU2*EXPAU)
       ELSE
#endif /*  BLOCK*/
    ENTHOLE = ENTHOLE + QQ/R * (-POLYAU1*EXPAU)
    !           write(*,*) 'atom-drude ',r,QQ/R * (-POLYAU1*EXPAU)
    DFAD = QQ/R**3 * (POLYAU2*EXPAU)
#if KEY_BLOCK==1 /*blocker*/
       ENDIF
#endif /*  BLOCK*/
    DX(I  )=DX(I  )+RX*DFAD
    DY(I  )=DY(I  )+RY*DFAD
    DZ(I  )=DZ(I  )+RZ*DFAD
    DX(J+1)=DX(J+1)-RX*DFAD
    DY(J+1)=DY(J+1)-RY*DFAD
    DZ(J+1)=DZ(J+1)-RZ*DFAD

    !           Drude-Atom
    QQ = CG(I+1)*CG(J) * CCELEC
    RX = X(I+1)-X(J)
    RY = Y(I+1)-Y(J)
    RZ = Z(I+1)-Z(J)
    R = SQRT(RX*RX+RY*RY+RZ*RZ)
    AU = NORM*R
    EXPAU = EXP(-AU)
    POLYAU1 = ONE+AU*HALF
    POLYAU2 = ONE+AU*(ONE+HALF*AU)
#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
            ENTHOLE = ENTHOLE + COEF * QQ/R * (-POLYAU1*EXPAU)
            DFDA = COEF * QQ/R**3 * (POLYAU2*EXPAU)
       ELSE
#endif /*  BLOCK*/
    ENTHOLE = ENTHOLE + QQ/R * (-POLYAU1*EXPAU)
    !           write(*,*) 'drude-atom ',r,QQ/R * (-POLYAU1*EXPAU)
    DFDA = QQ/R**3 * (POLYAU2*EXPAU)
#if KEY_BLOCK==1 /*blocker*/
       ENDIF
#endif /*  BLOCK*/
    DX(I+1)=DX(I+1)+RX*DFDA
    DY(I+1)=DY(I+1)+RY*DFDA
    DZ(I+1)=DZ(I+1)+RZ*DFDA
    DX(J  )=DX(J  )-RX*DFDA
    DY(J  )=DY(J  )-RY*DFDA
    DZ(J  )=DZ(J  )-RZ*DFDA

    !           Drude-Drude
    QQ = CG(I+1)*CG(J+1) * CCELEC
    RX = X(I+1)-X(J+1)
    RY = Y(I+1)-Y(J+1)
    RZ = Z(I+1)-Z(J+1)
    R = SQRT(RX*RX+RY*RY+RZ*RZ)
    AU = NORM*R
    EXPAU = EXP(-AU)
    POLYAU1 = ONE+AU*HALF
    POLYAU2 = ONE+AU*(ONE+HALF*AU)
#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
            ENTHOLE = ENTHOLE + COEF * QQ/R * (-POLYAU1*EXPAU)
            DFDD = COEF * QQ/R**3 * (POLYAU2*EXPAU)
       ELSE
#endif /*  BLOCK*/
    ENTHOLE = ENTHOLE + QQ/R * (-POLYAU1*EXPAU)
    !           write(*,*) 'drude-drude ',r,QQ/R * (-POLYAU1*EXPAU)
    DFDD = QQ/R**3 * (POLYAU2*EXPAU)
#if KEY_BLOCK==1 /*blocker*/
       ENDIF
#endif /*  BLOCK*/
    DX(I+1)=DX(I+1)+RX*DFDD
    DY(I+1)=DY(I+1)+RY*DFDD
    DZ(I+1)=DZ(I+1)+RZ*DFDD
    DX(J+1)=DX(J+1)-RX*DFDD
    DY(J+1)=DY(J+1)-RY*DFDD
    DZ(J+1)=DZ(J+1)-RZ*DFDD

    !           write(*,*) 'ENTHOLE ',ENTHOLE
    RETURN
  END SUBROUTINE ETHOLE2

  FUNCTION E_THOLE( QQ,AA,RX,RY,RZ, DF,SECOND,DDF )  &
       result(e_thole_1)
    !     
    !     Returns the screened Coulomb interaction.
    !     Computes the force and, in option, the second derivative.
    !     
    use number
    !     Arguments
    real(chm_real) e_thole_1
    real(chm_real),intent(in) ::  QQ,AA             ! INPUT
    real(chm_real),intent(in) ::  RX,RY,RZ          ! INPUT
    real(chm_real),intent(out) ::  DF                ! OUTPUT
    LOGICAL,intent(in) :: SECOND            ! INPUT
    real(chm_real),intent(out) ::  DDF               ! OUTPUT
    !     Locals
    real(chm_real)  NORM,R,AU,EXPAU,POLYAU

    NORM = AA**(-SIXTH)
    R = SQRT(RX*RX+RY*RY+RZ*RZ)
    AU = NORM*R
    EXPAU = EXP(-AU)

    !     U = QQ/R [1 - (1+AU/2)exp(-AU)]
    POLYAU = ONE+AU*HALF
    E_THOLE_1 = QQ/R * (ONE-POLYAU*EXPAU)
    !     (dU/dR)/R = QQ/R^3 [(1+AU+AU^2/2)exp(-AU) - 1]
    POLYAU = ONE+AU*(ONE+HALF*AU)
    DF = QQ/R**3 * (POLYAU*EXPAU-ONE)
    IF (SECOND) THEN
       POLYAU = ONE+AU*(ONE+AU*(HALF+PT25*AU))
       !        (d2U/dR2)/R^2 = 2QQ/R^5 [1-(1+AU+AU^2/2+AU^4/4)exp(-AU)]
       DDF = TWO*QQ/R**5 * (ONE-POLYAU*EXPAU)
    ENDIF

    RETURN
  END FUNCTION E_THOLE

  SUBROUTINE THOLE_NB(ISDRUDE,ALPHADP,THOLEI,CG, &
       ISTART, NBTHOLP, &
       NBTHOL1, NBTHOL2, NBTHOL3, &
       NBTHOLXIJ, &
       EN, X,Y,Z, DX,DY,DZ)
    !
    !     Pair-specific nonbonded Thole interaction
    !
    use number,only:zero
    integer ISTART, NBTHOLP
    integer NBTHOL1(*), NBTHOL2(*), NBTHOL3(*)
    real(chm_real)  NBTHOLXIJ(*)

    !     NBTHOLXIJ  numerical values of the pairwise Thole factor from the parameter file
    !     NBTHOLP    number of atom pairs in the mini nonbonded list of pair-specific Thole
    !     NBTHOL1    first atom in the mini nonbonded list of pair-specific Thole
    !     NBTHOL2    second atom in the mini nonbonded list of pair-specific Thole
    !     NBTHOL3    pointer for the pairwise Thole factor in NBTHOLXIJ

    real(chm_real)  EN
    LOGICAL ISDRUDE(*)
    real(chm_real)  ALPHADP(*),THOLEI(*)
    real(chm_real)  CG(*)
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  DX(*),DY(*),DZ(*)

    !     Local variables
    INTEGER I,J
    real(chm_real)  EBTH, EBTHOLE
    real(chm_real) dist

    ! Pair-specific thole shielding (used for divalent ions and cation-anion pairs, B. Roux, Oct 2010)

    EBTHOLE = zero

    do i=ISTART,NBTHOLP
       j = NBTHOL3(i)
       CALL ETHOLE2(NBTHOL1(i),NBTHOL2(i),X,Y,Z,DX,DY,DZ, &
            ISDRUDE,ALPHADP,NBTHOLXIJ(j),CG,EBTH)
       EBTHOLE = EBTHOLE + EBTH
    enddo

    EN = EN + EBTHOLE

    return
  end subroutine thole_nb

  SUBROUTINE THOLE_NBX(ISDRUDE,ALPHADP,THOLEI,CG,  &
       EN, INB14,IBLO14, &
       X,Y,Z, DX,DY,DZ, DD1,IUPT,QSECD, &
       NATOM )
    !     
    !     Thole interaction for the nonbonded exclusion list
    !     
    use dimens_fcm
    use number
    use consta
    use parallel
#if KEY_BLOCK==1
    use block_fcm
    use pert
#endif /*  BLOCK*/
    !     
    !     Arguments
    real(chm_real)  EN                ! INPUT/OUTPUT
    INTEGER INB14(*)          ! INPUT
    INTEGER IBLO14(*)         ! INPUT
    LOGICAL ISDRUDE(*)
    real(chm_real)  ALPHADP(*),THOLEI(*)
    real(chm_real)  CG(*)             ! INPUT
    real(chm_real)  X(*),Y(*),Z(*)    ! INPUT
    real(chm_real)  DX(*),DY(*),DZ(*) ! INPUT/OUTPUT
    real(chm_real)  DD1(*)            ! INPUT/OUTPUT
    INTEGER IUPT(*)           ! INPUT
    LOGICAL QSECD             ! INPUT
    INTEGER NATOM             ! INPUT 
    !     Locals
    INTEGER I,J,IX,NXI,NXIMAX
    LOGICAL VALID
    real(chm_real)  EBTHOLE
    real(chm_real)  DFAA,DFDA,DFAD,DFDD
    real(chm_real)  DDFAA,DDFDA,DDFAD,DDFDD
    INTEGER II,JJ,IADD
    real(chm_real)  RX,RY,RZ,DXI,DYI,DZI,R2,A
    INTEGER COUNTER
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real) COEF
#endif /*  BLOCK*/

    !      IF (.NOT.QTHOLE) RETURN

    COUNTER = ZERO

#if KEY_PARALLEL==1
    DO I = MYNODP,NATOM,NUMNOD
#else /**/
    DO I = 1,NATOM
#endif 
       IF (I.GT.1) THEN
          NXI = IBLO14(I-1)+1
       ELSE
          NXI = 1
       ENDIF
       NXIMAX = IBLO14(I)
       DO IX = NXI,NXIMAX
          J = INB14(IX)
          ! whuch inbe is right ???            IF (J.LT.0) GOTO 122
          IF (J.LE.0) GOTO 122
          !           At this point, (I,J) is an excluded pair
          COUNTER = COUNTER + 1
          CALL ETHOLE(NATOM, &
               ISDRUDE,ALPHADP,THOLEI,CG,  &
               X,Y,Z, I,J, VALID, EBTHOLE, &
               DFAA,DFDA,DFAD,DFDD, &
               QSECD,DDFAA,DDFDA,DDFAD,DDFDD,COUNTER )
          IF (VALID) THEN

#if KEY_BLOCK==1 /*blocker*/
             IF (QBLOCK) THEN
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL.LT.IBL) THEN
                      KK=JBL
                      JBL=IBL
                      IBL=KK
                   ENDIF
                   KK=IBL+JBL*(JBL-1)/2
                   COEF = BLCOEE(KK)

                   EBTHOLE=COEF*EBTHOLE
                   DFAA=COEF*DFAA
                   DFDA=COEF*DFDA
                   DFAD=COEF*DFAD
                   DFDD=COEF*DFDD
                   IF(QSECD) THEN
                     DDFAA=COEF*DDFAA
                     DDFDA=COEF*DDFDA
                     DDFAD=COEF*DDFAD
                     DDFDD=COEF*DDFDD
                   ENDIF
             ENDIF
#endif /*  BLOCK*/

             EN = EN+EBTHOLE
             !              Atom-Atom
             RX = X(I)-X(J)
             RY = Y(I)-Y(J)
             RZ = Z(I)-Z(J)
             DXI = RX*DFAA
             DYI = RY*DFAA
             DZI = RZ*DFAA
             DX(I)=DX(I)+DXI 
             DY(I)=DY(I)+DYI
             DZ(I)=DZ(I)+DZI
             DX(J)=DX(J)-DXI
             DY(J)=DY(J)-DYI
             DZ(J)=DZ(J)-DZI
             IF (QSECD) THEN
#if KEY_PARALLEL==1
                IF(NUMNOD.GT.1)CALL WRNDIE(-5,'<DRUDE>', &
                     'Second derivatives not supported in parallel.')
#endif 
                R2 = ONE/(RX*RX+RY*RY+RZ*RZ)
                DDFAA = DDFAA-DFAA*R2
                IF (J.LT.I) THEN
                   JJ=3*I-2
                   II=3*J-2
                ELSE
                   JJ=3*J-2
                   II=3*I-2
                ENDIF
                !                 XX
                A=RX*RX*DDFAA+DFAA
                IADD=IUPT(II)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ
                DD1(IADD)=DD1(IADD)+A
                !                 YY
                A=RY*RY*DDFAA+DFAA
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 ZZ
                A=RZ*RZ*DDFAA+DFAA
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+2)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 XY
                A=RX*RY*DDFAA
                IADD=IUPT(II)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 XZ
                A=RX*RZ*DDFAA
                IADD=IUPT(II)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 YZ
                A=RY*RZ*DDFAA
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+2
                DD1(IADD)=DD1(IADD)+A
             ENDIF
             !              Atom-Drude
             RX = X(I)-X(J+1)
             RY = Y(I)-Y(J+1)
             RZ = Z(I)-Z(J+1)
             DX(I  )=DX(I  )+RX*DFAD
             DY(I  )=DY(I  )+RY*DFAD
             DZ(I  )=DZ(I  )+RZ*DFAD
             DX(J+1)=DX(J+1)-RX*DFAD
             DY(J+1)=DY(J+1)-RY*DFAD
             DZ(J+1)=DZ(J+1)-RZ*DFAD
             IF (QSECD) THEN
                R2 = ONE/(RX*RX+RY*RY+RZ*RZ)
                DDFAD = DDFAD-DFAD*R2
                IF ((J+1).LT.I) THEN
                   JJ=3*I-2
                   II=3*(J+1)-2
                ELSE
                   JJ=3*(J+1)-2
                   II=3*I-2
                ENDIF
                !                 XX
                A=RX*RX*DDFAD+DFAD
                IADD=IUPT(II)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ
                DD1(IADD)=DD1(IADD)+A
                !                 YY
                A=RY*RY*DDFAD+DFAD
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 ZZ
                A=RZ*RZ*DDFAD+DFAD
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+2)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 XY
                A=RX*RY*DDFAD
                IADD=IUPT(II)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 XZ
                A=RX*RZ*DDFAD
                IADD=IUPT(II)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 YZ
                A=RY*RZ*DDFAD
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+2
                DD1(IADD)=DD1(IADD)+A
             ENDIF
             !              Drude-Atom
             RX = X(I+1)-X(J)
             RY = Y(I+1)-Y(J)
             RZ = Z(I+1)-Z(J)
             DX(I+1)=DX(I+1)+RX*DFDA
             DY(I+1)=DY(I+1)+RY*DFDA
             DZ(I+1)=DZ(I+1)+RZ*DFDA
             DX(J  )=DX(J  )-RX*DFDA
             DY(J  )=DY(J  )-RY*DFDA
             DZ(J  )=DZ(J  )-RZ*DFDA
             IF (QSECD) THEN
                R2 = ONE/(RX*RX+RY*RY+RZ*RZ)
                DDFDA = DDFDA-DFDA*R2
                IF (J.LT.(I+1)) THEN
                   JJ=3*(I+1)-2
                   II=3*J-2
                ELSE
                   JJ=3*J-2
                   II=3*(I+1)-2
                ENDIF
                !                 XX
                A=RX*RX*DDFDA+DFDA
                IADD=IUPT(II)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ
                DD1(IADD)=DD1(IADD)+A
                !                 YY
                A=RY*RY*DDFDA+DFDA
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 ZZ
                A=RZ*RZ*DDFDA+DFDA
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+2)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 XY
                A=RX*RY*DDFDA
                IADD=IUPT(II)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 XZ
                A=RX*RZ*DDFDA
                IADD=IUPT(II)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 YZ
                A=RY*RZ*DDFDA
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+2
                DD1(IADD)=DD1(IADD)+A
             ENDIF
             !              Drude-Drude
             RX = X(I+1)-X(J+1)
             RY = Y(I+1)-Y(J+1)
             RZ = Z(I+1)-Z(J+1)
             DX(I+1)=DX(I+1)+RX*DFDD
             DY(I+1)=DY(I+1)+RY*DFDD
             DZ(I+1)=DZ(I+1)+RZ*DFDD
             DX(J+1)=DX(J+1)-RX*DFDD
             DY(J+1)=DY(J+1)-RY*DFDD
             DZ(J+1)=DZ(J+1)-RZ*DFDD
             IF (QSECD) THEN
                R2 = ONE/(RX*RX+RY*RY+RZ*RZ)
                DDFDD = DDFDD-DFDD*R2
                IF ((J+1).LT.(I+1)) THEN
                   JJ=3*(I+1)-2
                   II=3*(J+1)-2
                ELSE
                   JJ=3*(J+1)-2
                   II=3*(I+1)-2
                ENDIF
                !                 XX
                A=RX*RX*DDFDD+DFDD
                IADD=IUPT(II)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ
                DD1(IADD)=DD1(IADD)+A
                !                 YY
                A=RY*RY*DDFDD+DFDD
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 ZZ
                A=RZ*RZ*DDFDD+DFDD
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+2)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 XY
                A=RX*RY*DDFDD
                IADD=IUPT(II)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+1
                DD1(IADD)=DD1(IADD)+A
                !                 XZ
                A=RX*RZ*DDFDD
                IADD=IUPT(II)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+2
                DD1(IADD)=DD1(IADD)+A
                !                 YZ
                A=RY*RZ*DDFDD
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+2
                DD1(IADD)=DD1(IADD)+A
             ENDIF
          ENDIF
122       CONTINUE
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE THOLE_NBX

  SUBROUTINE EANISOTROPY(EANIS,X,Y,Z,DX,DY,DZ,NATOM)
    !
    !     EANSI    - anisotropic drude harmonic potential
    !     X,Y,Z    - Coordinates for energy evaluation
    !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
    !     NATOM    - Number of atoms
    !
    !     Author: Ed Harder
    !
    use dimens_fcm
    use number
    use stream
    use aniso_fcm
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_aniso,only:nanisolist, anisolist, make_anisolist_current
#endif
#if KEY_BLOCK==1
    use block_fcm
    use pert  
#endif /*  BLOCK*/

    real(chm_real) EANIS
    INTEGER NATOM
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    !
    INTEGER IANISO,I,J,L,M,N,P
    real(chm_real) u1x, u1y, u1z
    real(chm_real) u2x, u2y, u2z
    real(chm_real) drx, dry, drz
    real(chm_real) u1mag, u2mag, dpar, dperp

    real(chm_real) KPAR0, KPERP0, KISO0
    integer ii, ii_end

#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real) COEF
#endif /*  BLOCK*/

#if KEY_DOMDEC==1
    if (q_domdec) then
       call make_anisolist_current()
       ii_end = nanisolist
    else
#endif
       ii_end = naniso
#if KEY_DOMDEC==1
    endif
#endif

    EANIS=ZERO

    !do ianiso=1,naniso
    do ii = 1, ii_end
       ianiso = ii
#if KEY_DOMDEC==1
       if (q_domdec) then
          ianiso = anisolist(ii)%ind
       endif
#endif
       I = LSTANI1(IANISO)
       J = I + 1
       L = LSTANI2(IANISO)
       M = LSTANI3(IANISO)
       N = LSTANI4(IANISO)

       !      write(*,*) I,J,L,M,N,
       !     &           K11(IANISO),K22(IANISO), K33(IANISO)

       !     set up parallel and perpendicular unit vectors

       u1x = X(I) - X(L)
       u1y = Y(I) - Y(L)
       u1z = Z(I) - Z(L)
       u2x = X(M) - X(N) 
       u2y = Y(M) - Y(N)
       u2z = Z(M) - Z(N)

       u1mag = sqrt(u1x*u1x + u1y*u1y &
            +  u1z*u1z)
       u2mag = sqrt(u2x*u2x + u2y*u2y &
            + u2z*u2z)
       u1x = u1x/u1mag
       u1y = u1y/u1mag
       u1z = u1z/u1mag
       u2x = u2x/u2mag
       u2y = u2y/u2mag
       u2z = u2z/u2mag

       !     drude displacement vector

       drx = X(J) - X(I) 
       dry = Y(J) - Y(I)
       drz = Z(J) - Z(I)

       !     multiply force constants by 2 for consistency with CHARMMM bond FF

       KPAR0  = 2*K11(IANISO)
       KPERP0 = 2*K22(IANISO)
       KISO0  = 2*K33(IANISO)

       !     perpendicular and parallel drude displ vectors
       dpar = drx*u1x + dry*u1y + drz*u1z
       dperp = drx*u2x + dry*u2y + drz*u2z

#if KEY_BLOCK==1 /*blocker*/
       IF (QBLOCK) THEN
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             IF (JBL.LT.IBL) THEN
                KK=JBL
                JBL=IBL
                IBL=KK
             ENDIF
             KK=IBL+JBL*(JBL-1)/2
             COEF = BLCOEB(KK)
             IF(QNOBO) COEF = 1.0

             KPAR0 = COEF * KPAR0
             KPERP0 = COEF * KPERP0
             KISO0 = COEF * KISO0
       ENDIF
#endif /*  BLOCK*/


       !     aniso spring energy
       !     kpar: reduces response along carbonyl vector
       !     kperp:  reduces response perp to bond vector (reg in and out of plane response)
       
       EANIS = EANIS + 0.5*KPAR0*dpar*dpar + 0.5*KPERP0*dperp*dperp  &
            + 0.5*KISO0*(drx*drx + dry*dry + drz*drz)

       !     iso spring force 

       DX(I) = DX(I) - KISO0*drx   
       DY(I) = DY(I) - KISO0*dry   
       DZ(I) = DZ(I) - KISO0*drz   

       DX(J) = DX(J) + KISO0*drx   
       DY(J) = DY(J) + KISO0*dry   
       DZ(J) = DZ(J) + KISO0*drz   

       !     par/perp spring forces

       DX(I)=DX(I)+KPAR0*dpar*(-u1x + (drx  &
            - u1x*(drx*u1x+dry*u1y+drz*u1z))/u1mag) &
            - KPERP0*dperp*u2x 
       DY(I)=DY(I)+KPAR0*dpar*(-u1y + (dry &
            - u1y*(drx*u1x+dry*u1y+drz*u1z))/u1mag) &
            - KPERP0*dperp*u2y
       DZ(I)=DZ(I)+KPAR0*dpar*(-u1z + (drz &
            - u1z*(drx*u1x+dry*u1y+drz*u1z))/u1mag) &
            - KPERP0*dperp*u2z

       DX(J) =  DX(J)+KPAR0*dpar*u1x + KPERP0*dperp*u2x
       DY(J) =  DY(J)+KPAR0*dpar*u1y + KPERP0*dperp*u2y
       DZ(J) =  DZ(J)+KPAR0*dpar*u1z + KPERP0*dperp*u2z

       DX(L) =  DX(L)+KPAR0*dpar*(-drx  &
            + u1x*(drx*u1x+dry*u1y+drz*u1z))/u1mag  
       DY(L) =  DY(L)+KPAR0*dpar*(-dry &
            + u1y*(drx*u1x+dry*u1y+drz*u1z))/u1mag
       DZ(L) =  DZ(L)+KPAR0*dpar*(-drz &
            + u1z*(drx*u1x+dry*u1y+drz*u1z))/u1mag

       DX(M) = DX(M)+KPERP0*dperp*(drx  &
            - u2x*(drx*u2x+dry*u2y+drz*u2z))/u2mag   
       DY(M) = DY(M)+KPERP0*dperp*(dry  &
            - u2y*(drx*u2x+dry*u2y+drz*u2z))/u2mag  
       DZ(M) = DZ(M)+KPERP0*dperp*(drz   &
            - u2z*(drx*u2x+dry*u2y+drz*u2z))/u2mag

       DX(N) = DX(N)+KPERP0*dperp*(-drx  &
            + u2x*(drx*u2x+dry*u2y+drz*u2z))/u2mag   
       DY(N) = DY(N)+KPERP0*dperp*(-dry  &
            + u2y*(drx*u2x+dry*u2y+drz*u2z))/u2mag  
       DZ(N) = DZ(N)+KPERP0*dperp*(-drz   &
            + u2z*(drx*u2x+dry*u2y+drz*u2z))/u2mag

    ENDDO
    RETURN
  END SUBROUTINE EANISOTROPY

end module drude

