  SUBROUTINE RBQUAS(X,Y,Z,NAT3,XREF,YREF,ZREF,NUNIT,FIRSTU,NSKIP, &
       IUNBAS,IUNTRN,NDIM,NFREQ,AMASS,DDV,DDM,DDF,DDEV,DDSCR, &
       DD5,DDS,DDV2,NADD,TEMP,NBEGN,NSTOP)
    !
    ! This routine does a reduced basis quasiharmonic analysis and
    ! saves the eigenvectors in the original basis. The basis functions
    ! must be on a file pointed to by IUNBAS.
    !
    ! Bernard R. Brooks      4-FEB-1987
    !
  use chm_kinds
  use exfunc
  use number
  use cvio
  use memory
  use consta
  use ctitla
  use redbas_m
  use stream
  use timerm
  use vibio
  use machutil,only:timrb,timre

    implicit none
    INTEGER NAT3,NFREQ,NADD,IUNBAS,IUNTRN,NDIM
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    INTEGER NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(NAT3)
    real(chm_real) DD5(*),DDS(*),DDV2(*)
    real(chm_real) TEMP

    real(chm_real) DELTA
    real(chm_real) RSTEP,RTEMP,RSMAL
    INTEGER NCOORD,ISTATS,IUNIT
    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    INTEGER NFREAT,IFILE,ISTEP,NDEGF,NSAVV
    INTEGER NN,I,J,IPT

    INTEGER ICNTRL(20)
    CHARACTER(len=4) :: HDR
    INTEGER NATOM,NATP
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8

    CHARACTER(len=4) :: HDR1,HDR2
    DATA HDR1,HDR2/'COOR','CORD'/

    RSMAL=0.00001
    RTEMP=TEMP
    RTEMP=KBOLTZ*RTEMP
    NATOM=NAT3/3

    IF(TIMER.GT.0) CALL TIMRB
    IF(PRNLEV.GE.2) WRITE(OUTU,33)
33  FORMAT(' QUASI: Generating the reduced basis fluctuation matrix.')

    NN=(NDIM*(NDIM+1))/2
    DD5(1:NN)=0.0
    DDM(1:NATOM)=1.0/DDM(1:NATOM)

    NCOORD=0
    call chmalloc('rbquas.src','RBQUAS','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('rbquas.src','RBQUAS','ITEMP',NATOM,cr4=ITEMP)
    IUNIT=FIRSTU
    ISTATS=1

    DO WHILE (ISTATS.GE.0)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
            ITEMP,NATOM,IFREAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NCOORD=NCOORD+1
       !
       ! COMPUTE COORDINATE DIFFERENCES
       !
       IPT=0
       DO I=1,NATOM
          DDSCR(IPT+1)=(X(I)-XREF(I))*DDM(I)
          DDSCR(IPT+2)=(Y(I)-YREF(I))*DDM(I)
          DDSCR(IPT+3)=(Z(I)-ZREF(I))*DDM(I)
          IPT=IPT+3
       ENDDO
       !
       ! NOW ADD IN THE CONTRIBUTION FOR THIS COORDINATE SET
       !
       DO I=1,NDIM
          DDS(I)=0.0
          DO J=1,NAT3
             DDS(I)=DDS(I)+DDV(J,I)*DDSCR(J)
          ENDDO
       ENDDO

       IPT=0
       DO I=1,NDIM
          DO J=I,NDIM
             IPT=IPT+1
             DD5(IPT)=DD5(IPT)+DDS(I)*DDS(J)
          ENDDO
       ENDDO

    ENDDO
    call chmdealloc('rbquas.src','RBQUAS','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('rbquas.src','RBQUAS','IFREAT',NATOM,intg=IFREAT)
    !
    ! NOW SCALE THE MATRIX APPROPRIATELY
    !
    RSTEP=NCOORD
    RSTEP=1.0/RSTEP
    IPT=0
    DO I=1,NDIM
       DO J=I,NDIM
          IPT=IPT+1
          DD5(IPT)=DD5(IPT)*RSTEP
       ENDDO
    ENDDO

    IF(TIMER.GT.0) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,39)
39     FORMAT('      Timing for this step:')
       CALL TIMRE
       CALL TIMRB
    ENDIF

    IF(PRNLEV.GE.2) WRITE(OUTU,35)
35  FORMAT(' QUASI: Diagonalizing the reduced force', &
         ' constant matrix')

    IH1=1
    NATP=NDIM+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(NDIM,NFREQ,DD5,DDV2,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)

    DO I=1,NFREQ
       DDEV(I)=DDS(I)
       IF (ABS(DDEV(I)).LT.RSMAL) DDEV(I)=SIGN(RSMAL,DDEV(I))
       DDEV(I)=RTEMP/DDEV(I)
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    IF(TIMER.GT.0) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,39)
       CALL TIMRE
       CALL TIMRB
    ENDIF

    IF(IUNTRN.GT.0) THEN
       ! SAVE TRANSFORMATION MATRIX FOR EXTERNAL BACKTRANSFORMATION
       IF(PRNLEV.GE.2) WRITE(OUTU,46) IUNTRN
46     FORMAT(' QUASI: Saving the transformation matrix to unit',I4)
       CALL WRTNMD(.FALSE.,1,NDIM,NDIM,DDV2,DDSCR,DDEV,IUNTRN,AMASS)
    ENDIF

    IF(PRNLEV.GE.2) WRITE(OUTU,36)
36  FORMAT(' QUASI: Backtransforming to the original basis')

    IF(IOLEV.GT.0) THEN
       IF (reallow) THEN      
          REWIND (UNIT=IUNBAS)
       ENDIF                  
       READ(IUNBAS) HDR,ICNTRL
       CALL RDTITL(TITLEB,NTITLB,IUNBAS,-1)
       READ(IUNBAS)
       READ(IUNBAS)
    ENDIF

    CALL RBDIA2(NAT3,NFREQ,NDIM,DDV,DDV2,DDSCR,IUNBAS)

    IF(TIMER.GT.0) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,39)
       CALL TIMRE
       CALL TIMRB
    ENDIF

    IF(PRNLEV.GE.2) WRITE(OUTU,37)
37  FORMAT(' QUASI: Calculation completed')

    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,625)
625    FORMAT(/15X,'FREQUENCIES'/)
       CALL FREQPR(DDF,NFREQ,OUTU)
    ENDIF

  END SUBROUTINE RBQUAS

