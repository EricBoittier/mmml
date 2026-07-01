module dynanal
  use chm_kinds
  use dimens_fcm
  implicit none

contains

  SUBROUTINE AVECOR(X,Y,Z,WMAIN,NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,ISLCT, &
       QPRINT,QPAX,RPAX, &
       ORIENT,LNORO,XCOMP,YCOMP,ZCOMP,WCOMP,JSLCT, &
       LMASS,LWMAIN,AMASS)
    !
    !     THIS ROUTINE GETS THE AVERAGE COORDINATES FROM A DYNAMICS RUN
    !
    !     By Bernard R. Brooks   9/22/1983
    !

#if KEY_CHEQ==1
    use cheq,only:qcg         
#endif
    use number
    use ctitla
    use stream
    use cvio
    use image
    use memory
    use corsubs,only:pkmass,rotls1
    use chutil,only:qhavcrd

    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real4),allocatable,dimension(:) :: ITEMP
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: CCG        
#endif
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    INTEGER NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP
    INTEGER ISLCT(*)
    LOGICAL QPRINT,QPAX
    real(chm_real) RPAX(15,*)
    !
    real(chm_real) XS,YS,ZS,FLUC,XX,YY,ZZ
    real(chm_real) SCR(24),AMOM(6)
    INTEGER NCOORD,ISTATS,I,IUNIT,J
    INTEGER IFILE,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER NSLCT
    real(chm_real) XXAVE,YYAVE,ZZAVE,ANAVE
    !
    ! To re-orient the coordinate 
    LOGICAL ORIENT,LNORO,LMASS,LWMAIN
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)
    real(chm_real) AMASS(*)
    real(chm_real),allocatable,dimension(:) :: BMASS
    INTEGER JSLCT(*),NPR
    integer,allocatable,dimension(:,:) :: ATOMPR
    !
    !     tec3 11/97 -- compute xtlabc averages (i.e. lattice averages)
    !     and xucell fluctuations (i.e. standard deviations of box
    !     distances and angles).  This is set up to work even in absence
    !     of selected atoms and is a useful routine to call prior to
    !     unfolding the trajectories.  NOTE: UXCELL is updated with the
    !     averaged box parameters after the call!
    !
    real(chm_real) AVGXTL(6), AVGUCL(6), VARXTL(6)

    real(chm_real) DELTA
    CHARACTER(len=4) :: HDR1='COOR', HDR2='CORD'

    real(chm_real),allocatable,dimension(:),target :: space_rl
    real(chm_real),pointer,dimension(:) :: xsum,ysum,zsum,xref,yref,zref,xfluc,yfluc,zfluc

    call chmalloc('dynanal.src','avecor','atompr',9*natom,crl=space_rl)
    xsum  => space_rl(0*natom+1:1*natom)
    ysum  => space_rl(1*natom+1:2*natom)
    zsum  => space_rl(2*natom+1:3*natom)
    xref  => space_rl(3*natom+1:4*natom)
    yref  => space_rl(4*natom+1:5*natom)
    zref  => space_rl(5*natom+1:6*natom)
    xfluc => space_rl(6*natom+1:7*natom)
    yfluc => space_rl(7*natom+1:8*natom)
    zfluc => space_rl(8*natom+1:9*natom)



    if(orient) then
       call chmalloc('dynanal.src','avecor','atompr',2,natom,intg=atompr)
       call chmalloc('dynanal.src','avecor','bmass',natom,crl=bmass)
    else
       call chmalloc('dynanal.src','avecor','atompr',1,1,intg=atompr)
       call chmalloc('dynanal.src','avecor','bmass',1,crl=bmass)
    endif

    DO I=1,NATOM
       XSUM(I)=ZERO
       YSUM(I)=ZERO
       ZSUM(I)=ZERO
       XFLUC(I)=ZERO
       YFLUC(I)=ZERO
       ZFLUC(I)=ZERO
    ENDDO
    !
    IF(QPAX) THEN
       DO I=1,NATOM
          DO J=1,3
             RPAX(J,I)=ZERO
          ENDDO
       ENDDO
    ENDIF
    !
    !     Setup the orient option assuming that the reference structure is 
    !     in the COMP set
    !
    IF(ORIENT)THEN
       NPR=0
       DO I=1,NATOM
          IF(JSLCT(I) == 1)THEN
             NPR=NPR+1
             ATOMPR(1,NPR)=I
             ATOMPR(2,NPR)=I
          ENDIF
       enddo
       CALL PKMASS(AMASS,AMASS,BMASS,ATOMPR,NPR,LMASS,LWMAIN,WMAIN)
    ENDIF
    !
    NCOORD=0
    call chmalloc('dynanal.src','AVECOR','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('dynanal.src','AVECOR','ITEMP',NATOM,cr4=ITEMP)
#if KEY_CHEQ==1
    call chmalloc('dynanal.src','AVECOR','CCG',NATOM,crl=CCG)      
#endif
    IUNIT=FIRSTU
    ISTATS=1
    DO WHILE(ISTATS >= 0)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CCG,QCG,                        & 
#endif
            ITEMP,NATOM,IFREAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NCOORD=NCOORD+1
       !
       IF(ORIENT)THEN
          ! Re-orient the main coordinate set with respect to selected atoms
          ! in the comp
          !lni Check that comparison coordinates are present if ORIEnt is requested
          IF(NCOORD == 1 .AND. .NOT. QHAVCRD(NATOM,JSLCT,XCOMP)) THEN
             CALL WRNDIE(1,'<AVECOR>', &
                  'NO COMP. COORD. FOR ORIENT! USING FIRST FRAME.')
             DO I =1,NATOM
                XCOMP(I)=X(I)
                YCOMP(I)=Y(I)
                ZCOMP(I)=Z(I)
             ENDDO
          ENDIF
          CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM, &
               ATOMPR,NPR,BMASS,QPRINT,LNORO)
       ENDIF
       !
       IF(NCOORD == 1) THEN
          NSLCT=0
          DO I=1,NATOM
             IF(ISLCT(I) == 1) THEN
                NSLCT=NSLCT+1
                XREF(I)=X(I)
                YREF(I)=Y(I)
                ZREF(I)=Z(I)
             ENDIF
          ENDDO
          DO I=1,6
             AVGXTL(I) = ZERO
             AVGUCL(I) = ZERO
             VARXTL(I) = ZERO
          ENDDO
          IF(NSLCT == 0 .AND. XTLABC(1) == ZERO)then
             call chmdealloc('dynanal.src','avecor','atompr',9*natom,crl=space_rl)
             RETURN
          endif
       ENDIF
       !
       !     NOW ADD IN THE CONTRIBUTION FOR THIS COORDINATE SET
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             XS=X(I)-XREF(I)
             YS=Y(I)-YREF(I)
             ZS=Z(I)-ZREF(I)
             XSUM(I)=XS+XSUM(I)
             YSUM(I)=YS+YSUM(I)
             ZSUM(I)=ZS+ZSUM(I)
             XFLUC(I)=XS*XS+XFLUC(I)
             YFLUC(I)=YS*YS+YFLUC(I)
             ZFLUC(I)=ZS*ZS+ZFLUC(I)
             IF(QPAX) THEN
                RPAX(1,I)=RPAX(1,I)+XS*YS
                RPAX(2,I)=RPAX(2,I)+XS*ZS
                RPAX(3,I)=RPAX(3,I)+YS*ZS
             ENDIF
          ENDIF
       ENDDO
       DO I=1,6
          AVGXTL(I) = AVGXTL(I) + XTLABC(I)
          AVGUCL(I) = AVGUCL(I) + XUCELL(I)
          VARXTL(I) = VARXTL(I) + XUCELL(I)*XUCELL(I)
       ENDDO
       !
    ENDDO
    call chmdealloc('dynanal.src','AVECOR','IFREAT',NATOM,intg=IFREAT)
    call chmdealloc('dynanal.src','AVECOR','ITEMP',NATOM,cr4=ITEMP)
#if KEY_CHEQ==1
    call chmdealloc('dynanal.src','AVECOR','CCG',NATOM,crl=CCG)      
#endif

    ! Done processing coordinate sets
    !
    ! Now scale the sum appropriately
    !
    XXAVE=ZERO
    YYAVE=ZERO
    ZZAVE=ZERO
    ANAVE=ZERO
    !
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          XS=XSUM(I)/NCOORD
          YS=YSUM(I)/NCOORD
          ZS=ZSUM(I)/NCOORD
          X(I)=XS+XREF(I)
          Y(I)=YS+YREF(I)
          Z(I)=ZS+ZREF(I)
          XX=(XFLUC(I)/NCOORD)-(XS**2)
          YY=(YFLUC(I)/NCOORD)-(YS**2)
          ZZ=(ZFLUC(I)/NCOORD)-(ZS**2)
          FLUC=XX+YY+ZZ
          IF(FLUC > ZERO) THEN
             WMAIN(I)=SQRT(FLUC)
          ELSE
             WMAIN(I)=ZERO
          ENDIF
          !
          ! Save principal axis data
          IF(QPAX) THEN
             AMOM(1)=XX
             AMOM(2)=(RPAX(1,I)/NCOORD)-XS*YS
             AMOM(3)=(RPAX(2,I)/NCOORD)-XS*ZS
             AMOM(4)=YY
             AMOM(5)=(RPAX(3,I)/NCOORD)-YS*ZS
             AMOM(6)=ZZ
             !        
             CALL DIAGQ(3,3,AMOM,RPAX(1,I),SCR(4),SCR(7),SCR(10), &
                  SCR(13),RPAX(10,I),SCR(16),SCR(19),SCR(22),0)
             !        
             RPAX(13,I)=X(I)
             RPAX(14,I)=Y(I)
             RPAX(15,I)=Z(I)
             XS=RPAX(12,I)
             YS=RPAX(11,I)
             ZS=RPAX(10,I)
             FLUC=0.5*(YS+ZS)
             IF(FLUC > ZERO) THEN
                FLUC=(XS/FLUC)**HALF-ONE
             ELSE
                FLUC=ZERO
             ENDIF
             IF(QPRINT .AND. PRNLEV >= 3) &
                  WRITE(OUTU,45) I,XS,YS,ZS,FLUC
45           FORMAT(' Atom',I5,' PAX moments:',3F12.5, &
                  '  Anisotropy:',F12.6)
             IF(PRNLEV > 5) THEN
                WRITE(OUTU,44)'X vector',RPAX(7,I),RPAX(8,I),RPAX(9,I)
                WRITE(OUTU,44)'Y vector',RPAX(4,I),RPAX(5,I),RPAX(6,I)
                WRITE(OUTU,44)'Z vector',RPAX(1,I),RPAX(2,I),RPAX(3,I)
44              FORMAT(10X,A,3F12.6)
             ENDIF
             XXAVE=XXAVE+XS
             YYAVE=YYAVE+YS
             ZZAVE=ZZAVE+ZS
             ANAVE=ANAVE+FLUC
          ENDIF
          !
       ENDIF
    ENDDO
    !
    IF(QPAX) THEN
       XXAVE=XXAVE/NSLCT
       YYAVE=YYAVE/NSLCT
       ZZAVE=ZZAVE/NSLCT
       ANAVE=ANAVE/NSLCT
       FLUC=0.5*(YYAVE+ZZAVE)
       IF(FLUC > ZERO) THEN
          FLUC=(XXAVE/FLUC)**HALF-ONE
       ELSE
          FLUC=ZERO
       ENDIF
       IF(PRNLEV >= 3) WRITE(OUTU,46) XXAVE,YYAVE,ZZAVE,ANAVE,FLUC
46     FORMAT(/' Averages: PAX moments:',3F12.5,'  Anisotropy:',F12.6, &
            ' from moments:',F12.6)
    ENDIF
    !
    IF(AVGXTL(1)  >  ZERO) THEN
       DO I=1,6
          AVGXTL(I) = AVGXTL(I) / NCOORD
          AVGUCL(I) = AVGUCL(I) / NCOORD
          VARXTL(I) = VARXTL(I) / NCOORD - AVGUCL(I)*AVGUCL(I)
          IF (VARXTL(I)  <  ZERO) THEN
             VARXTL(I) = ZERO
          ELSE
             VARXTL(I) = SQRT(VARXTL(I))
          ENDIF
       ENDDO

       CALL XTLLAT(XUCELL, AVGXTL)
       if (prnlev >= 2) then
          WRITE (OUTU,56) 'LAVE ', &
               ' A     = ',XUCELL(1),' B    = ',XUCELL(2),' C     = ',XUCELL(3)
          WRITE (OUTU,56) 'LAVE ', &
               ' Alpha = ',XUCELL(4),' Beta = ',XUCELL(5),' Gamma = ',XUCELL(6)

          WRITE (OUTU,56) 'LFLC ', &
               ' A     = ',VARXTL(1),' B    = ',VARXTL(2),' C     = ',VARXTL(3)
          WRITE (OUTU,56) 'LFLC ', &
               ' Alpha = ',VARXTL(4),' Beta = ',VARXTL(5),' Gamma = ',VARXTL(6)
       endif

       CALL XTLMSR(XUCELL)
       DO I=1,6
          XTLABC(I) = AVGXTL(I)
       ENDDO

    ENDIF
56  FORMAT(6X,A4,3(A,F10.5))
    call chmdealloc('dynanal.src','avecor','atompr',9*natom,crl=space_rl)

    RETURN
  END SUBROUTINE AVECOR

  SUBROUTINE PAXANL(X,Y,Z,NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,ISLCT, &
       RPAX,AMASS,QPRNT)
    !
    !     This routine gets the moments alaong the PAX defined by a previous
    !     COOR DYNA PAX command.
    !
    !     By Bernard R. Brooks   7/3/89
    !

#if KEY_CHEQ==1
    use cheq,only:qcg    
#endif
    use number
    use memory
    use ctitla
    use stream
    use cvio

    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real4),allocatable,dimension(:) :: ITEMP
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: CCG      
#endif
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real),dimension(:,:),allocatable,target :: sumspace
    real(chm_real),pointer,dimension(:,:) :: XSUM,YSUM,ZSUM
    INTEGER NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP
    INTEGER ISLCT(*)
    real(chm_real) RPAX(15,*)
    real(chm_real)  AMASS(*)
    LOGICAL QPRNT
    !
    real(chm_real) XAVE(6),YAVE(6),ZAVE(6),AAVE(6)
    real(chm_real) XABS(6),YABS(6),ZABS(6)
    real(chm_real) XS,YS,ZS,XR,YR,ZR,SKEW,KURT,ANIS1,ANIS2,RMSF,DENOM
    INTEGER NCOORD,ISTATS,I,IUNIT,J
    INTEGER IFILE,NFREAT,ISTEP,NDEGF,NSAVV
    INTEGER NSLCT
    LOGICAL QPRINT
    real(chm_real) DELTA
    CHARACTER(len=4) :: HDR1='COOR', HDR2='CORD'

    call chmalloc('dynanal.src','paxanl','sumspace',4,3*natom,crl=sumspace)
    xsum => sumspace(1:4,         1: natom)
    ysum => sumspace(1:4,   natom+1:2*natom)
    zsum => sumspace(1:4, 2*natom+1:3*natom)

    QPRINT=(QPRNT .AND. PRNLEV >= 2)
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          NSLCT=NSLCT+1
          XSUM(1:4,I)=ZERO
          YSUM(1:4,I)=ZERO
          ZSUM(1:4,I)=ZERO
       ENDIF
    ENDDO
    !
    IF(NSLCT == 0) then
       call chmdealloc('dynanal.src','paxanl','sumspace',4,3*natom,crl=sumspace)
       RETURN
    endif
    !
    NCOORD=0
    call chmalloc('dynanal.src','PAXANL','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('dynanal.src','PAXANL','ITEMP',NATOM,cr4=ITEMP)
#if KEY_CHEQ==1
    call chmalloc('dynanal.src','PAXANL','CCG',NATOM,crl=CCG)      
#endif
    IUNIT=FIRSTU
    ISTATS=1
    DO WHILE(ISTATS >= 0)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CCG,QCG,                          & 
#endif
            ITEMP,NATOM,IFREAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NCOORD=NCOORD+1
       !
       ! Now add in the contribution for this coordinate set
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             XS=X(I)-RPAX(13,I)
             YS=Y(I)-RPAX(14,I)
             ZS=Z(I)-RPAX(15,I)
             XR = XS*RPAX(7,I) + YS*RPAX(8,I) + ZS*RPAX(9,I)
             YR = XS*RPAX(4,I) + YS*RPAX(5,I) + ZS*RPAX(6,I)
             ZR = XS*RPAX(1,I) + YS*RPAX(2,I) + ZS*RPAX(3,I)
             ZR =-ZR
             DO J=1,4
                XSUM(J,I)=XSUM(J,I)+XR**J
                YSUM(J,I)=YSUM(J,I)+YR**J
                ZSUM(J,I)=ZSUM(J,I)+ZR**J
             ENDDO
          ENDIF
       ENDDO
       !
    ENDDO
    call chmdealloc('dynanal.src','PAXANL','IFREAT',NATOM,intg=IFREAT)
    call chmdealloc('dynanal.src','PAXANL','ITEMP',NATOM,cr4=ITEMP)
#if KEY_CHEQ==1
    call chmdealloc('dynanal.src','PAXANL','CCG',NATOM,crl=CCG)      
#endif

    ! Done processing coordinate sets
    !
    ! Now print out the results
    !
    DO J=1,6
       XAVE(J)=ZERO
       YAVE(J)=ZERO
       ZAVE(J)=ZERO
       XABS(J)=ZERO
       YABS(J)=ZERO
       ZABS(J)=ZERO
       AAVE(J)=ZERO
    ENDDO
    !
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          DO J=1,4
             XSUM(J,I)=XSUM(J,I)/NCOORD
             YSUM(J,I)=YSUM(J,I)/NCOORD
             ZSUM(J,I)=ZSUM(J,I)/NCOORD
             XAVE(J)=XAVE(J)+XSUM(J,I)
             YAVE(J)=YAVE(J)+YSUM(J,I)
             ZAVE(J)=ZAVE(J)+ZSUM(J,I)
             XABS(J)=XABS(J)+ABS(XSUM(J,I))
             YABS(J)=YABS(J)+ABS(YSUM(J,I))
             ZABS(J)=ZABS(J)+ABS(ZSUM(J,I))
          ENDDO
          !
          IF(QPRINT .and. prnlev >= 2) WRITE(OUTU,25)
25        FORMAT(' PAX moments for atom',I5)
          !
          ! Compute rms fluctuations and anisotropy values
          RMSF=XSUM(2,I)+YSUM(2,I)+ZSUM(2,I)
          IF(RMSF > ZERO) THEN
             RMSF=SQRT(RMSF)
          ELSE
             RMSF=ZERO
          ENDIF
          !
          DENOM=0.5*(YSUM(2,I)+ZSUM(2,I))
          IF(DENOM <= ZERO) THEN
             ANIS1=ZERO
             ANIS2=ZERO
          ELSE
             ANIS1=(XSUM(2,I)/DENOM)**HALF-ONE
             ANIS2=(YSUM(2,I)/DENOM)**HALF-ONE
          ENDIF
          AAVE(1)=AAVE(1)+ANIS1
          AAVE(2)=AAVE(2)+ANIS2
          AAVE(3)=AAVE(3)+RMSF
          AAVE(4)=AAVE(4)+RMSF*AMASS(I)
          AAVE(5)=AAVE(5)+AMASS(I)
          !
          ! Compute skew and kurtosis values
          SKEW=XSUM(3,I)/(ABS(XSUM(2,I))**1.5)
          KURT=XSUM(4,I)/(XSUM(2,I)**2)-3.0
          XAVE(5)=XAVE(5)+SKEW
          XAVE(6)=XAVE(6)+KURT
          XABS(5)=XABS(5)+ABS(SKEW)
          XABS(6)=XABS(6)+ABS(KURT)
          IF(QPRINT .and. prnlev >= 2) WRITE(OUTU,26) (XSUM(J,I),J=1,4),SKEW,KURT
          !
          SKEW=YSUM(3,I)/(ABS(YSUM(2,I))**1.5)
          KURT=YSUM(4,I)/(YSUM(2,I)**2)-3.0
          YAVE(5)=YAVE(5)+SKEW
          YAVE(6)=YAVE(6)+KURT
          YABS(5)=YABS(5)+ABS(SKEW)
          YABS(6)=YABS(6)+ABS(KURT)
          IF(QPRINT .and. prnlev >= 2) WRITE(OUTU,27) (YSUM(J,I),J=1,4),SKEW,KURT
          !
          SKEW=ZSUM(3,I)/(ZSUM(2,I)**1.5)
          KURT=ZSUM(4,I)/(ABS(ZSUM(2,I))**2)-3.0
          ZAVE(5)=ZAVE(5)+SKEW
          ZAVE(6)=ZAVE(6)+KURT
          ZABS(5)=ZABS(5)+ABS(SKEW)
          ZABS(6)=ZABS(6)+ABS(KURT)
          IF(QPRINT .and. prnlev >= 2) WRITE(OUTU,28) (ZSUM(J,I),J=1,4),SKEW,KURT
          !
          IF(QPRINT .and. prnlev >= 2) WRITE(OUTU,38) RMSF,ANIS1,ANIS2
          !
26        FORMAT('    <X>=',F14.5,'  <X2>=',F14.5,'  <X3>=',F14.5, &
               '  <X4>=',F14.5,'  Skew=',F14.5,'  Kurtosis=',F14.5)
27        FORMAT('    <Y>=',F14.5,'  <Y2>=',F14.5,'  <Y3>=',F14.5, &
               '  <Y4>=',F14.5,'  Skew=',F14.5,'  Kurtosis=',F14.5)
28        FORMAT('    <Z>=',F14.5,'  <Z2>=',F14.5,'  <Z3>=',F14.5, &
               '  <Z4>=',F14.5,'  Skew=',F14.5,'  Kurtosis=',F14.5)
38        FORMAT('    RMS fluctuation=',F14.5,'  Anisotropy1=',F14.5, &
               '  Anisotropy2=',F14.5)
39        FORMAT('    Mass weighted RMS fluctuation=',F14.5, &
               '  Average mass=',F14.5)
          !
       ENDIF
    ENDDO
    !
    DO J=1,6
       XAVE(J)=XAVE(J)/NSLCT
       YAVE(J)=YAVE(J)/NSLCT
       ZAVE(J)=ZAVE(J)/NSLCT
       XABS(J)=XABS(J)/NSLCT
       YABS(J)=YABS(J)/NSLCT
       ZABS(J)=ZABS(J)/NSLCT
       AAVE(J)=AAVE(J)/NSLCT
    ENDDO
    !
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,29)
29     FORMAT(/' PAX moments averages:')
       WRITE(OUTU,26) (XAVE(J),J=1,6)
       WRITE(OUTU,27) (YAVE(J),J=1,6)
       WRITE(OUTU,28) (ZAVE(J),J=1,6)
       WRITE(OUTU,30)
30     FORMAT(' PAX moments absolute value averages:')
       WRITE(OUTU,26) (XABS(J),J=1,6)
       WRITE(OUTU,27) (YABS(J),J=1,6)
       WRITE(OUTU,28) (ZABS(J),J=1,6)
       !
       WRITE(OUTU,38) AAVE(3),AAVE(1),AAVE(2)
       WRITE(OUTU,39) AAVE(4)/AAVE(5),AAVE(5)
    ENDIF
    !
    call chmdealloc('dynanal.src','paxanl','sumspace',4,3*natom,crl=sumspace)
    RETURN
  END SUBROUTINE PAXANL
end module dynanal

