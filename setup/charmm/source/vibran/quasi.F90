  SUBROUTINE QUASI(XREF,YREF,ZREF,NATOM,NDIM,NFREQ,TEMP, &
       DDV,DDM,DDF,DDEV,DD5,DDS,DDSCR,NADD, &
       X,Y,Z,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,LTHERMO, &
       IFIRST,ILAST,NSET1,IND1,LRESI,FCUT, &
       LNOTP,LNOMA,LNORM,LMEAN,NCOORD)
    !
    !     SETS UP AND DIAGONALIZES THE SECOND DERIVATIVE MATRIX
    !
    !     By Bernard R. Brooks   9/19/1983
    !     Modified by Nathan Desdouits and Arnaud Blondel Jun 2012
    !                 to incorporate PCA calculation.
    !
  use chm_kinds
  use consta
  use stream
  use timerm
  use memory
  use chutil,only:atomid
  use machutil,only:wrttim
  use vector, only: normall
  use param_store, only: get_param
    implicit none
    INTEGER NDIM,NFREQ,NADD
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    real(chm_real) DDV(NDIM,*),DDM(*),DDF(*),DDEV(*)
    real(chm_real) DD5(*),DDS(*),DDSCR(*)
    real(chm_real) TEMP,FCUT
    real(chm_real) RTEMP,RSMAL,SSUM,S,HSUM,H,CSUM,CV,STEP
    real(chm_real),allocatable,dimension(:) :: DD6
    LOGICAL LTHERMO,LRESI,LNOMA,LNORM,LMEAN,LNOTP
    INTEGER IFIRST,ILAST,NSET1,IND1(*)
    INTEGER NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,NCOORD
    INTEGER NATOM,NATP,I,IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    INTEGER NA1,NF1,II,JJ,IPT,N,I1,I2,J,ISTART,NSMALL
    CHARACTER(len=8) :: SID,RID,REN,AC,SIDO,RIDO,RENO

    RSMAL=0.00001
    RTEMP=TEMP
    RTEMP=KBOLTZ*RTEMP
    IF(RTEMP.LE.0.0)THEN
       RTEMP=-1.0
    ENDIF

    IF(TIMER.GT.0) CALL WRTTIM('TIME INTO QUASI')
    CALL QUASI2(XREF,YREF,ZREF,X,Y,Z,NATOM, &
         NUNIT,FIRSTU,NSKIP,DD5,DDM,DDSCR,NBEGN,NSTOP, &
         NSET1,IND1,LNOMA,LMEAN,NCOORD)
    IF(TIMER.GT.0) CALL WRTTIM('TIME TO COLLECT FLUCTUATIONS')

    IH1=1
    NATP=NDIM+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    IF(LRESI.AND.LTHERMO.AND.(RTEMP.GT.0))THEN
       !
       ! Do the subsets first so that the full spectrum is available on exit
       !
       STEP=-1.0
       NA1=0
       SSUM=0.0
       HSUM=0.0
       CSUM=0.0
       N=0
       ISTART=1 
       WRITE(OUTU,'(/5X,A,/3(A4,2X),2X,3(A8,4X),1X,2(A7,3X))') &
            'Thermodynamic analysis/residue. Other correlations neglected', &
            'SEGI','RESN','RESI','Entropy','Enthalpy','Heatcap', &
            'Atm/res','Ign.frq'
       CALL ATOMID(IND1(1),SIDO,RIDO,RENO,AC)

       call chmalloc('quasi.src','QUASI','DD6',NDIM*(NDIM+1)/2,crl=DD6)
       DO I=1,NSET1
          CALL ATOMID(IND1(I),SID,RID,REN,AC)
          IF(RID.NE.RIDO)THEN
             ! At a new residue; sum up results for previous residue, reset counters
             N=3*N
             IPT=0  
             DO I1=1,N
                DO I2=I1,N
                   ! CHECK THAT WE GET RIGHT PART OF DD5!!!!!!
                   IPT=IPT+1
                   II=ISTART+I1-1
                   JJ=ISTART+I2-1
                   JJ=(II-1)*NDIM-(II-2)*(II-1)/2 +JJ-II+1 
                   DD6(IPT)=DD5(JJ)
                ENDDO
             ENDDO
             NF1=N
             CALL DIAGQ(N,NF1,DD6,DDV,DDS(IH2),DDS(IH3), &
                  DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NA1)
             DO J=1,NF1
                DDEV(J)=DDS(J)
                IF (ABS(DDEV(J)).LT.RSMAL)  &
                     DDEV(J)=SIGN(RSMAL,DDEV(J))
                DDEV(J)=RTEMP/DDEV(J)
                DDF(J)=CNVFRQ*SQRT(ABS(DDEV(J)))
                IF(DDEV(J).LT.0.0) DDF(J)=-DDF(J)
             ENDDO

             CALL THERMV(1,NF1,DDF,TEMP,STEP,.FALSE.,FCUT,NSMALL)
             call get_param('STOT', S)
             SSUM=SSUM+S
             call get_param('HTOT', H)
             HSUM=HSUM+H
             call get_param('CTOT', CV)
             CSUM=CSUM+CV
             WRITE(OUTU,200) &
                  SIDO(1:idleng),RENO(1:idleng),RIDO(1:idleng), &
                  S,H,CV,N/3,NSMALL
200          FORMAT(3(A,2X),3G12.4,2I10)

             ISTART=3*I-2
             N=1
             SIDO=SID
             RIDO=RID
             RENO=REN    
          ELSE
             N=N+1
          ENDIF
       ENDDO

       ! The last residue
       N=3*N
       IPT=0  
       DO I1=1,N
          DO I2=I1,N
             ! CHECK THAT WE GET RIGHT PART OF DD5!!!!!!
             IPT=IPT+1
             II=ISTART+I1-1
             JJ=ISTART+I2-1
             ! Position of (II,JJ) element in symmetrically stored NDIM x NDIM matrix:
             JJ=(II-1)*NDIM-(II-2)*(II-1)/2 +JJ-II+1 
             DD6(IPT)=DD5(JJ)
          ENDDO
       ENDDO
       NF1=N
       CALL DIAGQ(N,NF1,DD6,DDV,DDS(IH2),DDS(IH3), &
            DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NA1)
       DO J=1,NF1
          DDEV(J)=DDS(J)
          IF (ABS(DDEV(J)).LT.RSMAL) DDEV(J)=SIGN(RSMAL,DDEV(J))
          DDEV(J)=RTEMP/DDEV(J)
          DDF(J)=CNVFRQ*SQRT(ABS(DDEV(J)))
          IF(DDEV(J).LT.0.0) DDF(J)=-DDF(J)
       ENDDO
       call chmdealloc('quasi.src','QUASI','DD6',NDIM*(NDIM+1)/2,crl=DD6)

       CALL THERMV(1,NF1,DDF,TEMP,STEP,.FALSE.,FCUT,NSMALL)
       call get_param('STOT', S)
       SSUM=SSUM+S
       call get_param('HTOT', H)
       HSUM=HSUM+H
       call get_param('CTOT', CV)
       CSUM=CSUM+CV
       WRITE(OUTU,200) &
            SIDO(1:idleng),RENO(1:idleng),RIDO(1:idleng), &
            S,H,CV,N/3,NSMALL
       IF(TIMER.GT.0) CALL WRTTIM('TIME TO DIAGONALIZE SUBMATRICES')
    ENDIF

    ! And now the full matrix   
    CALL DIAGQ(NDIM,NFREQ,DD5,DDV,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)
    IF(TIMER.GT.0) CALL WRTTIM('TIME TO DIAGONALIZE FULL MATRIX')

    IF(LNORM)THEN
       DO I=1,NFREQ
          CALL NORMALL(DDV(1,I),NDIM)
       ENDDO
    ENDIF

    IF(PRNLEV.GE.4) WRITE(OUTU,595)
595 FORMAT(/15X,'DIAGONALIZATION COMPLETED')

    DO I=1,NFREQ
       DDEV(I)=DDS(I)
       IF (ABS(DDEV(I)).LT.RSMAL) DDEV(I)=SIGN(RSMAL,DDEV(I))
       IF(RTEMP.LE.0.0)THEN
          DDF(I)=ABS(DDEV(I))
       ELSE
          DDEV(I)=RTEMP/DDEV(I)
          DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       ENDIF
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    IF(LTHERMO)THEN
       STEP=-1.0
       CALL THERMV(IFIRST,ILAST,DDF,TEMP,STEP,.FALSE.,FCUT,NSMALL)
       call get_param('STOT', S)
       call get_param('HTOT', H)
       call get_param('CTOT', CV)
       IF(PRNLEV.GE.2)THEN
          IF(LRESI)THEN
             WRITE(OUTU,'(/3X,A,3X,3G12.4,2I10)')  &
                  'All selected',S,H,CV,NSET1,NSMALL
             WRITE(OUTU,'(//5X,A,G12.4/5X,A,G12.4/)')  &
                  'Sum of residue entropies (Sres)=',SSUM, &
                  '          Difference Sfull-Sres=',S-SSUM
          ELSE
             WRITE(OUTU,'(/5X,3(2X,A,G12.4),/5X,A,I6,/5X,A,I6/)')  &
                  'Entropy=',S,'Enthalpy=',H,'Heat capacity=',CV, &
                  'Number of atoms=',NSET1, &
                  'Number of ignored freq =',NSMALL
          ENDIF
       ENDIF
    ENDIF
    IF(PRNLEV.GE.4)THEN
       WRITE(OUTU,625)
625    FORMAT(/15X,'FREQUENCIES'/)
       CALL FREQPR(DDF,NFREQ,OUTU)
    ENDIF
    RETURN
  END SUBROUTINE QUASI

  SUBROUTINE QUASI2(XREF,YREF,ZREF,X,Y,Z,NATOM, &
       NUNIT,FIRSTU,NSKIP,DD5,DDM,DDSCR,NBEGN,NSTOP, &
       NSET1,IND1,LNOMA,LMEAN,NCOORD)
    !
    !             THIS ROUTINE WILL SET UP A FLUCTUATION
    !      MATRIX FOR THE SOULUTION OF THE QUASIHARMONIC EQUATIONS
    !
    !     By Bernard R. Brooks   9/19/1983
    !        Introduction of PCA calculation by ND & AB.
    !
  use chm_kinds
  use exfunc
  use number
  use cvio
  use memory
  use ctitla
    implicit none

    real(chm_real) DD5(*),DDM(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    real(chm_real) DDSCR(*)
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,NSET1,IND1(*)
    real(chm_real) DELTA
    real(chm_real) DDMI
    INTEGER ICOORD,NCOORD,ISTATS,IUNIT
    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    real(chm_real),allocatable,dimension(:) :: DDSSP
    INTEGER NFREAT,IFILE,ISTEP,NDEGF,NSAVV
    INTEGER NDIM,NN,I,J,K,IPT
    LOGICAL LNOMA,LMEAN
    CHARACTER(len=4) :: HDR1,HDR2
    DATA HDR1,HDR2/'COOR','CORD'/
    !
    !     INITIALIZE THE SECULAR EQUATION MATRIX
    !

    NDIM=NSET1*3
    NN=(NDIM*(NDIM+1))/2
    DD5(1:NN)=0.0
    IF(LNOMA) THEN
       DDM(1:NATOM)=1.0
    ELSE
       DDM(1:NATOM)=1.0/DDM(1:NATOM)
    ENDIF
    IF(LMEAN)THEN
       XREF(1:NATOM)=0.0
       YREF(1:NATOM)=0.0
       ZREF(1:NATOM)=0.0
    ENDIF

    call chmalloc('quasi.src','QUASI2','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('quasi.src','QUASI2','ITEMP',NATOM,cr4=ITEMP)
    IF(LMEAN) THEN
       call chmalloc('quasi.src','QUASI2','DDSSP',3*NCOORD*NATOM,crl=DDSSP)
    ELSE
       call chmalloc('quasi.src','QUASI2','DDSSP',1,crl=DDSSP)
    ENDIF

    IUNIT=FIRSTU
    ISTATS=1
    ICOORD=0


    IPT=0
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
       ICOORD=ICOORD+1
       !
       !     COMPUTE COORDINATE DIFFERENCES
       !
       IF(LMEAN) THEN
          DO I=1,NSET1
             J=IND1(I)
             DDSSP(IPT+1)=X(J)
             DDSSP(IPT+2)=Y(J)
             DDSSP(IPT+3)=Z(J)
             IPT=IPT+3
             XREF(J)=XREF(J)+X(J)
             YREF(J)=YREF(J)+Y(J)
             ZREF(J)=ZREF(J)+Z(J)
          ENDDO
       ELSE
          IPT=0
          DO I=1,NSET1
             J=IND1(I)
             DDSCR(IPT+1)=(X(J)-XREF(J))*DDM(J)
             DDSCR(IPT+2)=(Y(J)-YREF(J))*DDM(J)
             DDSCR(IPT+3)=(Z(J)-ZREF(J))*DDM(J)
             IPT=IPT+3
          ENDDO
          !
          !     NOW ADD IN THE CONTRIBUTION FOR THIS COORDINATE SET
          !
          IPT=0
          DO I=1,NDIM
             DO J=I,NDIM
                IPT=IPT+1
                DD5(IPT)=DD5(IPT)+DDSCR(I)*DDSCR(J)
             ENDDO
          ENDDO
       ENDIF
    ENDDO


    DDMI=1.0/ICOORD
    IF(LMEAN) THEN
       DO J=1,NATOM
          XREF(J)=XREF(J)/ICOORD
          YREF(J)=YREF(J)/ICOORD
          ZREF(J)=ZREF(J)/ICOORD
       ENDDO
       IPT=0
       DO K=1,ICOORD
          DO I=1,NSET1
             J=IND1(I)
             DDSSP(IPT+1)=(DDSSP(IPT+1)-XREF(J))*DDM(J)
             DDSSP(IPT+2)=(DDSSP(IPT+2)-YREF(J))*DDM(J)
             DDSSP(IPT+3)=(DDSSP(IPT+3)-ZREF(J))*DDM(J)
             IPT=IPT+3
          ENDDO
       ENDDO
       IPT=0
       DO I=1,NDIM
          DO J=I,NDIM
             IPT=IPT+1
             DO K=0,ICOORD-1
                DD5(IPT)=DD5(IPT)+DDSSP(I+K*3*NATOM)*DDSSP(J+K*3*NATOM)
             ENDDO
             DD5(IPT)=DD5(IPT)*DDMI
          ENDDO
       ENDDO
    ELSE
       !
       !     NOW SCALE THE MATRIX APPROPRIATELY
       !
       IPT=0
       DO I=1,NDIM
          DO J=I,NDIM
             IPT=IPT+1
             DD5(IPT)=DD5(IPT)*DDMI
          ENDDO
       ENDDO

    ENDIF

    call chmdealloc('quasi.src','QUASI2','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('quasi.src','QUASI2','IFREAT',NATOM,intg=IFREAT)
    IF(LMEAN) THEN
       call chmdealloc('quasi.src','QUASI2','DDSSP',3*NCOORD*NATOM,crl=DDSSP)
    ELSE
       call chmdealloc('quasi.src','QUASI2','DDSSP',1,crl=DDSSP)
    ENDIF
  END SUBROUTINE QUASI2

  SUBROUTINE SQUAS(XREF,YREF,ZREF,NATOM,NDIM,NDIM2,NFREQ,TEMP, &
       DDV,DDM,DDF,DDEV,DD5,DDS,DDSCR,NADD, &
       X,Y,Z,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP, &
       IFIRST,ILAST,NSET1,IND1,FCUT, &
       LNOTP,LNOMA,NCOORD,LNORM,LMEAN)
    !
    !     SETS UP AND DIAGONALIZES THE SECOND DERIVATIVE MATRIX
    !
    !     Inspired By Bernard R. Brooks (9/19/1983),
    !                 Nathan Desdouits and Arnaud Blondel Jun 2012.
    !
  use chm_kinds
  use number, only: zero
  use consta
  use stream
  use timerm
  use memory
  use chutil,only:atomid
  use machutil,only:wrttim
  use vector, only: normall
    implicit none
    INTEGER NDIM,NDIM2,NFREQ,NADD
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    real(chm_real) DDV(NDIM2,*),DDM(*),DDF(*),DDEV(*)
    real(chm_real) DD5(*),DDS(*),DDSCR(*)
    real(chm_real) TEMP,FCUT
    real(chm_real) RTEMP,RSMAL,SSUM,S,HSUM,H,CSUM,CV,STEP
    real(chm_real),allocatable,dimension(:) :: DDV2
    real(chm_real),allocatable,dimension(:) :: DDSSP
    LOGICAL LNOMA,LNORM,LMEAN,LNOTP
    INTEGER IFIRST,ILAST,NSET1,IND1(*)
    INTEGER NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP
    INTEGER NCOORD,NATOM,NATP,I,IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    INTEGER NA1,NF1,II,JJ,IPT,N,I1,I2,J,K,ISTART,NSMALL
    CHARACTER(len=8) :: SID,RID,REN,AC,SIDO,RIDO,RENO

    RSMAL=0.00001
    RTEMP=TEMP
    IF(LNOTP)THEN
       RTEMP=-1.
    ELSE
       RTEMP=KBOLTZ*RTEMP
    ENDIF

    call chmalloc('quasi.src','SQUAS','DDSSP',NDIM2*NDIM,crl=DDSSP)
    call chmalloc('quasi.src','SQUAS','DDV2',NDIM*NDIM,crl=DDV2)

    IF(TIMER.GT.0) CALL WRTTIM('TIME INTO SQUAS')
    CALL SQUAS2(XREF,YREF,ZREF,X,Y,Z,NDIM,NDIM2,NATOM, &
         NUNIT,FIRSTU,NSKIP,DD5,DDSSP,DDM,DDSCR,NBEGN,NSTOP, &
         NSET1,IND1,NCOORD,LNOMA,LMEAN)
    IF(TIMER.GT.0) CALL WRTTIM('TIME TO COLLECT FLUCTUATIONS')

    IH1=1
    NATP=NDIM+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP

    ! And now the full matrix   
    CALL DIAGQ(NDIM,NFREQ,DD5,DDV2,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)
    IF(TIMER.GT.0) CALL WRTTIM('TIME TO DIAGONALIZE FULL MATRIX')

    ! Then, compute the normal modes in coordinate space
    ddv(1:ndim2,1:nfreq)=zero ! must be initialized ?? MH-14
    DO I=1,NFREQ        ! I freq number  (NFREQ)
       IPT=(I-1)*NDIM
       DO J=1,NDIM2     ! J atom number  (NAT3=NDIM2)
          IPT=IPT+1
          DO K=1,NDIM   ! K frame number (NCOORD=NDIM)
             DDV(J,I)=DDV(J,I)+DDV2((I-1)*NDIM+K)*DDSSP((K-1)*NDIM2+J)
          ENDDO
       ENDDO
    ENDDO

    IF(LNORM)THEN
       DO I=1,NFREQ
          CALL NORMALL(DDV(1,I),NDIM2)
       ENDDO
    ENDIF

    IF(PRNLEV.GE.4) WRITE(OUTU,595)
595 FORMAT(/15X,'DIAGONALIZATION COMPLETED')

    DO I=1,NFREQ
       DDEV(I)=DDS(I)
       IF (ABS(DDEV(I)).LT.RSMAL) DDEV(I)=SIGN(RSMAL,DDEV(I))
       IF(LNOTP)THEN
          DDF(I)=ABS(DDEV(I))
       ELSE
          DDEV(I)=RTEMP/DDEV(I)
          DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       ENDIF
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    IF(PRNLEV.GE.4)THEN
       WRITE(OUTU,625)
625    FORMAT(/15X,'FREQUENCIES'/)
       CALL FREQPR(DDF,NFREQ,OUTU)
    ENDIF
    call chmdealloc('quasi.src','SQUAS','DDSSP',NDIM2*NDIM,crl=DDSSP)
    call chmdealloc('quasi.src','SQUAS','DDV2',NDIM*NDIM,crl=DDV2)
    RETURN
  END SUBROUTINE SQUAS

  SUBROUTINE SQUAS2(XREF,YREF,ZREF,X,Y,Z,NDIM,NDIM2,NATOM, &
       NUNIT,FIRSTU,NSKIP,DD5,DDSSP,DDM,DDSCR,NBEGN,NSTOP, &
       NSET1,IND1,NCOORD,LNOMA,LMEAN)
    !
    !             THIS ROUTINE WILL SET UP A Covariance
    !      MATRIX FOR THE SOULUTION OF THE Principal component analysis
    !
    !     Inspired By Bernard R. Brooks (9/19/1983),
    !                 Nathan Desdouits and Arnaud Blondel Jun 2012.
    !
  use chm_kinds
  use exfunc
  use number
  use cvio
  use memory
  use ctitla
  use stream
    implicit none

    real(chm_real) DD5(*),DDM(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    real(chm_real) DDSCR(*),DDSSP(*)
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,NSET1,IND1(*)
    real(chm_real) DELTA
    real(chm_real) DDMI
    INTEGER ICOORD,NCOORD,ISTATS,IUNIT
    integer,allocatable,dimension(:) :: IFREAT
    real(chm_real4),allocatable,dimension(:) :: ITEMP
    LOGICAL LNOMA,LMEAN
    INTEGER NFREAT,IFILE,ISTEP,NDEGF,NSAVV
    INTEGER NDIM,NDIM2,NN,I,J,K,IPT
    CHARACTER(len=4) :: HDR1,HDR2
    DATA HDR1,HDR2/'COOR','CORD'/
    !
    !     INITIALIZE THE SECULAR EQUATION MATRIX
    !

    !NDIM=NSET1*3
    NN=(NDIM*(NDIM+1))/2
    DD5(1:NN)=0.0
    IF(LNOMA) THEN
       DDM(1:NATOM)=1.0
    ELSE
       DDM(1:NATOM)=1.0/DDM(1:NATOM)
    ENDIF
    ICOORD=0
    call chmalloc('quasi.src','SQUAS2','IFREAT',NATOM,intg=IFREAT)
    call chmalloc('quasi.src','SQUAS2','ITEMP',NATOM,cr4=ITEMP)
    IUNIT=FIRSTU
    ISTATS=1
    IPT=0
    IF(LMEAN)THEN
       XREF(1:NATOM)=0.0
       YREF(1:NATOM)=0.0
       ZREF(1:NATOM)=0.0
    ENDIF
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
       ICOORD=ICOORD+1
       IF(ICOORD.GT.NCOORD) THEN
         WRITE(OUTU,991)'READING FRAME ',ICOORD, &
          ' BUT WAS EXPECTING ',NCOORD,' FRAMES.'
991      FORMAT(A14,I6,A19,I6,A8)
         CALL WRNDIE(1,'<VIBRAN>','TOO MUCH FRAMES IN TRAJ.')
         ICOORD=ICOORD-1
         GOTO 990
       ENDIF
       !
       !     COMPUTE COORDINATE DIFFERENCES
       ! 
       IF(LMEAN)THEN
          DO I=1,NSET1
             J=IND1(I)
             DDSSP(IPT+1)=X(J)
             DDSSP(IPT+2)=Y(J)
             DDSSP(IPT+3)=Z(J)
             IPT=IPT+3
             XREF(J)=XREF(J)+X(J)
             YREF(J)=YREF(J)+Y(J)
             ZREF(J)=ZREF(J)+Z(J)
          ENDDO
       ELSE
          DO I=1,NSET1
             J=IND1(I)
             DDSSP(IPT+1)=(X(J)-XREF(J))*DDM(J)
             DDSSP(IPT+2)=(Y(J)-YREF(J))*DDM(J)
             DDSSP(IPT+3)=(Z(J)-ZREF(J))*DDM(J)
             IPT=IPT+3
          ENDDO
       ENDIF
    ENDDO
990 CONTINUE
    IF(ICOORD.LT.NCOORD) THEN
      WRITE(OUTU,992)'READ ',ICOORD, &
       ' FRAMES BUT WAS EXPECTING ',NCOORD,' FRAMES.'
992      FORMAT(A5,I6,A26,I6,A8)
      CALL WRNDIE(1,'<VIBRAN>','TOO FEW FRAMES IN TRAJ.')
    ENDIF
    IF(LMEAN)THEN
       DO J=1,NATOM
          XREF(J)=XREF(J)/ICOORD
          YREF(J)=YREF(J)/ICOORD
          ZREF(J)=ZREF(J)/ICOORD
       ENDDO
       IPT=0
       DO K=1,ICOORD
          DO I=1,NSET1
             J=IND1(I)
             DDSSP(IPT+1)=(DDSSP(IPT+1)-XREF(J))*DDM(J)
             DDSSP(IPT+2)=(DDSSP(IPT+2)-YREF(J))*DDM(J)
             DDSSP(IPT+3)=(DDSSP(IPT+3)-ZREF(J))*DDM(J)
             IPT=IPT+3
          ENDDO
       ENDDO
    ENDIF
    IPT=0
    ! matrix here corresponds to the secular matrix (M.M') but the
    ! matrices used to build are transposed (=> M'.M). The expression should
    ! be proportional to the average displacement hence weighting by 1/ICOORD.
    DDMI=1.0/ICOORD
    DO I=0,ICOORD-1
       DO J=I,ICOORD-1
          IPT=IPT+1
          DD5(IPT)=0.0
          DO K=1,NDIM2
             DD5(IPT)=DD5(IPT)+DDSSP(I*NDIM2+K)*DDSSP(J*NDIM2+K)
          ENDDO
          DD5(IPT)=DD5(IPT)*DDMI
       ENDDO
    ENDDO
    call chmdealloc('quasi.src','SQUAS2','ITEMP',NATOM,cr4=ITEMP)
    call chmdealloc('quasi.src','SQUAS2','IFREAT',NATOM,intg=IFREAT)

  END SUBROUTINE SQUAS2


