module vibsub
  implicit none
contains

  SUBROUTINE NORMDS(X,Y,Z,NAT3,BNBND,BIMAG,NFREQ,LNOMA,AMASS, &
       DDV,DDM,DDF,DDEV,NADD,LRAISE, &
       LFINIT,STEP,LDSCF,LENTRO, &
       NSAVDD1,QREST) ! JZ_UW12
    !
    ! SETS UP AND DIAGONALIZES THE SECOND DERIVATIVE MATRIX
    !
    ! By Bernard R. Brooks   1982
    ! Victor Anisimov        2004, Numerical IR spectrum for Drudes
    !                              and Vibrational Entropy term
    !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use deriv
  use consta
  use energym
  use stream
  use memory
  use vibcom, only: gensd2

    implicit none

    INTEGER NAT3,NFREQ,NADD
    type(nonbonddatastructure) :: BNBND
    type(imagedatastructure) :: BIMAG
    real(chm_real) :: X(:), Y(:), Z(:)
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*)

    LOGICAL LNOMA,LRAISE,QMASWT
    LOGICAL LFINIT,LDSCF,LENTRO
    real(chm_real)  STEP
    ! JZ_UW12
    INTEGER NSAVDD1
    LOGICAL QREST

    INTEGER NATOM,NATP,N6,I
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    real(chm_real),allocatable,dimension(:) :: DD1,DDS
    real(chm_real),allocatable,dimension(:) :: XREF,YREF,ZREF
    real(chm_real),allocatable,dimension(:) :: DXF,DYF,DZF

    NATOM=NAT3/3

    QMASWT=(.NOT.LNOMA)

    N6=(NAT3*(NAT3+1))/2
    call chmalloc('vibsub.src','NORMDS','DD1',N6,crl=DD1)
    DD1(1:N6)=0.0

    IF(LFINIT) THEN
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       call chmalloc('vibsub.src','NORMDS','XREF',NATOM,crl=XREF)
       call chmalloc('vibsub.src','NORMDS','YREF',NATOM,crl=YREF)
       call chmalloc('vibsub.src','NORMDS','ZREF',NATOM,crl=ZREF)
       call chmalloc('vibsub.src','NORMDS','DXF',NATOM,crl=DXF)
       call chmalloc('vibsub.src','NORMDS','DYF',NATOM,crl=DYF)
       call chmalloc('vibsub.src','NORMDS','DZF',NATOM,crl=DZF)

       CALL GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF, &
            DD1,DXF,DYF,DZF,STEP,LDSCF, &
            NSAVDD1,QREST) ! JZ_UW12

       call chmdealloc('vibsub.src','NORMDS','DZF',NATOM,crl=DZF)
       call chmdealloc('vibsub.src','NORMDS','DYF',NATOM,crl=DYF)
       call chmdealloc('vibsub.src','NORMDS','DXF',NATOM,crl=DXF)
       call chmdealloc('vibsub.src','NORMDS','ZREF',NATOM,crl=ZREF)
       call chmdealloc('vibsub.src','NORMDS','YREF',NATOM,crl=YREF)
       call chmdealloc('vibsub.src','NORMDS','XREF',NATOM,crl=XREF)
    ELSE
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
    ENDIF

    !! MH 2024:  this is wrong in most cases, but check if there is a reason
    !! that this is here - for now we leave it in the comments
    !! maybe some code that is not used in the standard pref.dat keys ??
    !! then this needs additional protection
!!#!!#if KEY_PARALLEL==1
!!#!!    CALL VDGBR(DX,DY,DZ,1)
!!#!!    CALL GCOMB(DD1,N6)
!!#!!#endif 

    IF(PRNLEV.GE.2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
            1, ZERO, ZERO, .TRUE.)
    ENDIF

    ! mass weight the Hessian.
    call chmalloc('vibsub.src','NORMDS','DDS',(NAT3+1)*8,crl=DDS)
    CALL RAISE(LRAISE,LRAISE,NAT3,NATOM,QMASWT,AMASS,DD1,DDM, &
         DDS,X,Y,Z,0,0,0,.FALSE.,QMASWT)

    IH1=1
    NATP=NAT3+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(NAT3,NFREQ,DD1,DDV,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)

    DO I=1,NFREQ
       DDEV(I)=DDS(I)
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO
    call chmdealloc('vibsub.src','NORMDS','DDS',(NAT3+1)*8,crl=DDS)
    call chmdealloc('vibsub.src','NORMDS','DD1',N6,crl=DD1)

    IF(PRNLEV.LT.2) RETURN
    WRITE(OUTU,595)
595 FORMAT(/15X,'DIAGONALIZATION COMPLETED')
    WRITE(OUTU,625)
625 FORMAT(/15X,'FREQUENCIES'/)
    CALL FREQPR(DDF,NFREQ,OUTU)

    IF(LENTRO) THEN
       CALL ENTROPDS(X,Y,Z,NAT3,NFREQ,AMASS,DDF,NADD,LRAISE)
    ENDIF

    RETURN
  END SUBROUTINE NORMDS

  SUBROUTINE SUBSDS(X,Y,Z,NAT3,BNBND,BIMAG,NFREQ,LNOMA,AMASS, &
       DDV,DDM,DDF,DDEV,ISLCT,NADD, &
       LRAISE,LFINIT,STEP,LDSCF,LENTRO)
    !
    ! SETS UP AND DIAGONALIZES THE SECOND DERIVATIVE MATRIX
    !
    ! By Bernard R. Brooks   2007
    !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use deriv
  use consta
  use energym
  use stream
  use memory
  use vibcom, only: gensd2

    implicit none

    INTEGER NAT3,NFREQ,NADD,ISLCT(*)
    type(nonbonddatastructure) :: BNBND
    type(imagedatastructure) :: BIMAG
    real(chm_real) :: X(:), Y(:), Z(:)
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*)

    LOGICAL LNOMA,LRAISE,QMASWT
    LOGICAL LFINIT,LDSCF,LENTRO
    real(chm_real)  STEP

    INTEGER NATOM,NATP,N6,I
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    real(chm_real),allocatable,dimension(:) :: DD1,DDS
    real(chm_real),allocatable,dimension(:) :: XREF,YREF,ZREF
    real(chm_real),allocatable,dimension(:) :: DXF,DYF,DZF
    INTEGER ISPACE,JSPACE,KSPACE,N3E,N3S,NSLCT
    integer,allocatable,dimension(:) :: IOFF
    real(chm_real),allocatable,dimension(:) :: HEE,HSS,VHEE
    real(chm_real),allocatable,dimension(:) :: TMS,TME
    integer,allocatable,dimension(:) :: I3SEL

    NATOM=NAT3/3

    QMASWT=(.NOT.LNOMA)

    N6=(NAT3*(NAT3+1))/2
    call chmalloc('vibsub.src','SUBSDS','DD1',N6,crl=DD1)
    DD1(1:N6)=0.0

    IF(LFINIT) THEN
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       call chmalloc('vibsub.src','SUBSDS','XREF',NATOM,crl=XREF)
       call chmalloc('vibsub.src','SUBSDS','YREF',NATOM,crl=YREF)
       call chmalloc('vibsub.src','SUBSDS','ZREF',NATOM,crl=ZREF)
       call chmalloc('vibsub.src','SUBSDS','DXF',NATOM,crl=DXF)
       call chmalloc('vibsub.src','SUBSDS','DYF',NATOM,crl=DYF)
       call chmalloc('vibsub.src','SUBSDS','DZF',NATOM,crl=DZF)

       CALL GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF, &
            DD1,DXF,DYF,DZF,STEP,LDSCF, &
            3*NATOM,.FALSE.) ! JZ_UW12

       call chmdealloc('vibsub.src','SUBSDS','DZF',NATOM,crl=DZF)
       call chmdealloc('vibsub.src','SUBSDS','DYF',NATOM,crl=DYF)
       call chmdealloc('vibsub.src','SUBSDS','DXF',NATOM,crl=DXF)
       call chmdealloc('vibsub.src','SUBSDS','ZREF',NATOM,crl=ZREF)
       call chmdealloc('vibsub.src','SUBSDS','YREF',NATOM,crl=YREF)
       call chmdealloc('vibsub.src','SUBSDS','XREF',NATOM,crl=XREF)
    ELSE
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
    ENDIF

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'DD1 matrix'
    write (6,'(12F15.5)') (DD1(i),i=1,12)
#endif 

    !! MH 2024:  this is wrong in most cases, but check if there is a reason
    !! that this is here - for now we leave it in the comments
    !! maybe some code that is not used in the standard pref.dat keys ??
    !! then this needs additional protection
!!#!!#if KEY_PARALLEL==1
!!#!!    CALL VDGBR(DX,DY,DZ,1)
!!#!!    CALL GCOMB(DD1,N6)
!!#!!#endif 

    IF(PRNLEV.GE.2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
            1, ZERO, ZERO, .TRUE.)
    ENDIF
    !
    ! mass weight the Hessian.
    ! CALL RAISE(LRAISE,LRAISE,NAT3,NATOM,QMASWT,AMASS,DD1,DDM,
    ! 1           DDS,X,Y,Z,0,0,0,.FALSE.,QMASWT)
    !
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) NSLCT=NSLCT+1
    ENDDO
    !
    ! allocate needed space
    !
    N3S=NSLCT*3
    N3E=NAT3-N3S
    IF(NFREQ.GT.N3S-NADD) NFREQ=N3S-NADD
    IF(NFREQ.LE.0) RETURN

#if KEY_DEBUGNMD==1
    write(6,'(5I10)') nat3,n3s,n3e,nfreq
#endif 

    call chmalloc('vibsub.src','SUBSDS','DDS',(NAT3+1)*8,crl=DDS)
    call chmalloc('vibsub.src','SUBSDS','TMS',N3S,crl=TMS)
    call chmalloc('vibsub.src','SUBSDS','TME',N3E,crl=TME)
    call chmalloc('vibsub.src','SUBSDS','I3SEL',NAT3,intg=I3SEL)
    call chmalloc('vibsub.src','SUBSDS','IOFF',NAT3,intg=IOFF)
    ISPACE=MAX(N3S**2,(N3E*(N3E+1))/2)
    call chmalloc('vibsub.src','SUBSDS','HEE',ISPACE,crl=HEE)
    JSPACE=MAX((N3S*(N3S+1))/2,N3S*NFREQ)
    call chmalloc('vibsub.src','SUBSDS','HSS',JSPACE,crl=HSS)
    KSPACE=MAX(N3E**2,N3E*N3S)
    call chmalloc('vibsub.src','SUBSDS','VHEE',KSPACE,crl=VHEE)

    CALL SUBSDS2(NAT3,NFREQ,LNOMA,AMASS, &
         DDV,DDM,DDF,DDEV,DD1,DDS,ISLCT,NSLCT,NADD, &
         N3S,N3E,TMS,TME, &
         I3SEL,IOFF, &
         HEE,HSS,VHEE,VHEE, &
         DD1,HEE,HEE,HSS,DD1)

    call chmdealloc('vibsub.src','SUBSDS','IOFF',NAT3,intg=IOFF)
    call chmdealloc('vibsub.src','SUBSDS','I3SEL',NAT3,intg=I3SEL)
    call chmdealloc('vibsub.src','SUBSDS','TME',N3E,crl=TME)
    call chmdealloc('vibsub.src','SUBSDS','TMS',N3S,crl=TMS)
    call chmdealloc('vibsub.src','SUBSDS','HEE',ISPACE,crl=HEE)
    call chmdealloc('vibsub.src','SUBSDS','HSS',JSPACE,crl=HSS)
    call chmdealloc('vibsub.src','SUBSDS','VHEE',KSPACE,crl=VHEE)
    call chmdealloc('vibsub.src','SUBSDS','DDS',(NAT3+1)*8,crl=DDS)
    call chmdealloc('vibsub.src','SUBSDS','DD1',N6,crl=DD1)

    IF(PRNLEV.LT.2) RETURN
    WRITE(OUTU,595)
595 FORMAT(/15X,'SUBSYSTEM NORMAL MODE ANALYSIS COMPLETED')
    WRITE(OUTU,625)
625 FORMAT(/15X,'FREQUENCIES'/)
    CALL FREQPR(DDF,NFREQ,OUTU)

    IF(LENTRO) THEN
       CALL ENTROPDS(X,Y,Z,NAT3,NFREQ,AMASS,DDF,NADD,LRAISE)
    ENDIF

    RETURN
  END SUBROUTINE SUBSDS

  SUBROUTINE SUBSDS2(NAT3,NFREQ,LNOMA,AMASS, &
       DDV,DDM,DDF,DDEV,DD1,DDS, &
       ISLCT,NSLCT,NADD,N3S,N3E,TMS,TME,I3SEL,IOFF, &
       HEE,HSS,VHEE,AMSE,TMSS,VMSS,VHSS,XSS,XSE)
    !
    ! Do the subsystem analysis
    !
    ! By Bernard R. Brooks   2007
    !
  use chm_kinds
  use dimens_fcm
  use number
  use vector
  use consta
  use stream

    implicit none

    INTEGER NAT3,NFREQ,NADD,ISLCT(*),NSLCT,N3S,N3E,I3SEL(*),IOFF(*)
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*)
    real(chm_real) DD1(*),DDS(*)
    real(chm_real) TMS(*),TME(*),HEE(*),HSS(*),VHEE(N3E,N3E), &
         AMSE(N3E,N3S)
    real(chm_real) TMSS(*),VMSS(N3S,N3S),VHSS(N3S,N3S)
    real(chm_real) XSS(N3S,N3S),XSE(N3E,N3S)
    !
    !  Array reuse via space allocation:
    !     DD1 -> TMSS -> XSE
    !     HEE -> VMSS -> VHSS
    !     HSS -> XSS
    !     VHEE -> AMSE
    !
    LOGICAL LNOMA,QMASWT

    INTEGER NATOM,NATP,N6,I,J,K
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    INTEGER IS,IE,JS,JE,KS,KE,II,JJ,IPT,JPT,KPT
    real(chm_real) VCUT,AMS
    CHARACTER(len=17) :: FMT

    VCUT=TENM5
    NATOM=NAT3/3

#if KEY_DEBUGNMD==1
    write(6,'(5I10)') nat3,n3s,n3e,nfreq
#endif 

    ! fill I3SEL
    IPT=0  ! N3  index
    JPT=0  ! N3s index
    KPT=0  ! N3e index
    DO I=1,NATOM
       I3SEL(IPT+1)=ISLCT(I)
       I3SEL(IPT+2)=ISLCT(I)
       I3SEL(IPT+3)=ISLCT(I)
       IPT=IPT+3
       IF(ISLCT(I).EQ.1) THEN
          TMS(JPT+1)=AMASS(I)
          TMS(JPT+2)=AMASS(I)
          TMS(JPT+3)=AMASS(I)
          JPT=JPT+3
       ELSE
          TME(KPT+1)=AMASS(I)
          TME(KPT+2)=AMASS(I)
          TME(KPT+3)=AMASS(I)
          KPT=KPT+3
       ENDIF
    ENDDO
    !
    ! Extract HEE and HSS
    !
    IPT=0   ! H index
    JPT=0   ! HEE index
    KPT=0   ! HSS index
    DO I=1,NAT3
       IF(I3SEL(I).EQ.0) THEN
          DO J=I,NAT3
             IPT=IPT+1
             IF(I3SEL(J).EQ.0) THEN
                JPT=JPT+1
                HEE(JPT)=DD1(IPT)
             ENDIF
          ENDDO
       ELSE
          DO J=I,NAT3
             IPT=IPT+1
             IF(I3SEL(J).NE.0) THEN
                KPT=KPT+1
                HSS(KPT)=DD1(IPT)
             ENDIF
          ENDDO
       ENDIF
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'ISLCT'
    write (6,'(12I5)') (ISLCT(j),j=1,natom)
    write(6,'(A)') ' '
    write(6,'(A)') 'I3SEL'
    write (6,'(12I5)') (I3SEL(j),j=1,nat3)
    write(6,'(A)') ' '

    write(6,'(A)') 'DD1 matrix'
    ipt=0
    do i=1,nat3
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(DD1(j),j=ipt+i,ipt+nat3)
       ipt=ipt+nat3-i
    enddo
    write(6,'(A)') ' '

    write(6,'(A)') 'Hee matrix'
    ipt=0
    do i=1,n3e
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(HEE(j),j=ipt+i,ipt+n3e)
       ipt=ipt+n3e-i
    enddo
    write(6,'(A)') ' '
    write(6,'(A)') 'Hss matrix'
    ipt=0
    do i=1,n3s
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(HSS(j),j=ipt+i,ipt+n3s)
       ipt=ipt+n3s-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! start to calculate Hee**-1
    ! diagonalize Hee
    !
    IH1=1
    NATP=N3E+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(N3E,N3E,HEE,VHEE,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDEV,DDS(IH6),DDS(IH7),DDS(IH8),0)


#if KEY_DEBUGNMD==1
    write(6,'(A)') 'EV-Hee matrix'
    write(6,'(12E15.5)') (DDEV(j),j=1,n3e)
    write(6,'(A)') ' '
#endif 
    !
    ! invert the eigenvalues
    !
    DO K=1,N3E
       IF(ABS(DDEV(K)).LT.VCUT) DDEV(K)=SIGN(VCUT,DDEV(K))
       DDEV(K)=ONE/DDEV(K)
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'EV-Hee**-1 matrix'
    write(6,'(12E15.5)') (DDEV(j),j=1,n3e)
    write(6,'(A)') ' '
#endif 

    !
    ! GET Hee**-1
    !
    IPT=0
    DO I=1,N3E
       DO J=I,N3E
          IPT=IPT+1
          HEE(IPT)=0
          DO K=1,N3E
             HEE(IPT)=HEE(IPT)+DDEV(K)*VHEE(I,K)*VHEE(J,K)
          ENDDO
       ENDDO
    ENDDO


#if KEY_DEBUGNMD==1
    write(6,'(A)') 'Hee**-1 matrix'
    ipt=0
    do i=1,n3e
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(HEE(j),j=ipt+i,ipt+n3e)
       ipt=ipt+n3e-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! calculate A = Hes * Hee**-1
    !

    DO IS=1,N3S
       DO KE=1,N3E
          AMSE(KE,IS)=ZERO
       ENDDO
    ENDDO
    IPT=0
    DO I=1,N3E
       IOFF(I)=IPT
       IPT=IPT+N3E-I
    ENDDO

    IPT=0     ! Hse index
    IE=0
    IS=0
    DO II=1,NAT3
       IF(I3SEL(II).EQ.0) THEN
          IE=IE+1
          JS=IS
          DO JJ=II,NAT3
             IPT=IPT+1
             IF(I3SEL(JJ).GT.0) THEN
                JS=JS+1
                DO KE=1,N3E
                   IF(IE.LT.KE) THEN
                      KPT=IOFF(IE)+KE
                   ELSE
                      KPT=IOFF(KE)+IE
                   ENDIF
                   AMSE(KE,JS)=AMSE(KE,JS)+DD1(IPT)*HEE(KPT)
                ENDDO
             ENDIF
          ENDDO
       ELSE
          IS=IS+1
          JE=IE
          DO JJ=II,NAT3
             IPT=IPT+1
             IF(I3SEL(JJ).EQ.0) THEN
                JE=JE+1
                DO KE=1,N3E
                   IF(JE.LT.KE) THEN
                      KPT=IOFF(JE)+KE
                   ELSE
                      KPT=IOFF(KE)+JE
                   ENDIF
                   AMSE(KE,IS)=AMSE(KE,IS)+DD1(IPT)*HEE(KPT)
                ENDDO
             ENDIF
          ENDDO
       ENDIF
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'AMSE matrix'
    do i=1,n3s
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(AMSE(j,i),j=1,n3e)
    enddo
    write(6,'(A)') ' '
#endif 
    !
    !  Calculate Hss effective
    !
    IPT=0
    DO I=1,N3S
       IOFF(I)=IPT
       IPT=IPT+N3S-I
    ENDDO

    IPT=0     ! Hse index
    IE=0
    IS=0
    DO II=1,NAT3
       IF(I3SEL(II).EQ.0) THEN
          IE=IE+1
          JS=IS
          DO JJ=II,NAT3
             IPT=IPT+1
             IF(I3SEL(JJ).GT.0) THEN
                JS=JS+1
                DO KS=1,N3S
                   IF(JS.LE.KS) THEN
                      KPT=IOFF(JS)+KS
                      HSS(KPT)=HSS(KPT)-AMSE(IE,KS)*DD1(IPT)
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
       ELSE
          IS=IS+1
          JE=IE
          DO JJ=II,NAT3
             IPT=IPT+1
             IF(I3SEL(JJ).EQ.0) THEN
                JE=JE+1
                DO KS=1,N3S
                   IF(IS.LE.KS) THEN
                      KPT=IOFF(IS)+KS
                      HSS(KPT)=HSS(KPT)-AMSE(JE,KS)*DD1(IPT)
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
       ENDIF
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'Hss-eff matrix'
    ipt=0
    do i=1,n3s
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(HSS(j),j=ipt+i,ipt+n3s)
       ipt=ipt+n3s-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! calculate TMss
    !
    IPT=0
    DO IS=1,N3S
       DO JS=IS,N3S
          IPT=IPT+1
          TMSS(IPT)=ZERO
          DO KE=1,N3E
             TMSS(IPT)=TMSS(IPT)+AMSE(KE,IS)*AMSE(KE,JS)*TME(KE)
          ENDDO
          IF(IS.EQ.JS) TMSS(IPT)=TMSS(IPT)+TMS(IS)
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'TMss matrix'
    ipt=0
    do i=1,n3s
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(TMSS(j),j=ipt+i,ipt+n3s)
       ipt=ipt+n3s-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! Calculate TM**-0.5
    ! start with diagonalize TMss
    !
    IH1=1
    NATP=N3S+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(N3S,N3S,TMSS,VMSS,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDEV,DDS(IH6),DDS(IH7),DDS(IH8),0)

    DO K=1,N3S
       IF(ABS(DDEV(K)).LT.VCUT) DDEV(K)=SIGN(VCUT,DDEV(K))
       DDEV(K)=ONE/SQRT(DDEV(K))
    ENDDO
    !
    ! GET TMSS**-1/2
    !
    IPT=0
    DO I=1,N3S
       DO J=I,N3S
          IPT=IPT+1
          TMSS(IPT)=ZERO
          DO K=1,N3S
             TMSS(IPT)=TMSS(IPT)+DDEV(K)*VMSS(I,K)*VMSS(J,K)
          ENDDO
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'TMss**-0,5 matrix'
    ipt=0
    do i=1,n3s
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(TMSS(j),j=ipt+i,ipt+n3s)
       ipt=ipt+n3s-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! calculate mass weighted Hss-eff
    !
    DO IS=1,N3S
       DO JS=1,N3S
          VMSS(JS,IS)=ZERO
          DO KS=1,N3S
             IF(JS.LT.KS) THEN
                IPT=IOFF(JS)+KS
             ELSE
                IPT=IOFF(KS)+JS
             ENDIF
             IF(IS.LT.KS) THEN
                JPT=IOFF(IS)+KS
             ELSE
                JPT=IOFF(KS)+IS
             ENDIF
             VMSS(JS,IS)=VMSS(JS,IS)+TMSS(IPT)*HSS(JPT)
          ENDDO
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'VMSS matrix'
    do i=1,n3s
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(VMSS(j,i),j=1,n3s)
    enddo
    write(6,'(A)') ' '
#endif 

    IPT=0
    DO IS=1,N3S
       DO JS=IS,N3S
          IPT=IPT+1
          HSS(IPT)=ZERO
          DO KS=1,N3S
             IF(JS.LT.KS) THEN
                JPT=IOFF(JS)+KS
             ELSE
                JPT=IOFF(KS)+JS
             ENDIF
             HSS(IPT)=HSS(IPT)+TMSS(JPT)*VMSS(IS,KS)
          ENDDO
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'Hss-eff matrix'
    ipt=0
    do i=1,n3s
       FMT='(I5, 30X,42F10.5)'
       write (FMT(5:7),'(I3)') 10*i+1
       write (6,FMT) i,(HSS(j),j=ipt+i,ipt+n3s)
       ipt=ipt+n3s-i
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! Diagonalize mass weighted Hss-eff
    !
    IH1=1
    NATP=N3S+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(N3S,NFREQ,HSS,VHSS,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDEV,DDS(IH6),DDS(IH7),DDS(IH8),NADD)

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'EV-Hss-eff matrix'
    write(6,'(12E15.5)') (DDEV(j),j=1,n3s)
    write(6,'(A)') ' '

    write(6,'(A)') 'VHss matrix'
    do i=1,n3s
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(VHSS(j,i),j=1,n3s)
    enddo
    write(6,'(A)') ' '
#endif 

    !
    ! convert to cartesian x = t**-0.5 * u
    !
    DO IS=1,NFREQ
       DO JS=1,N3S
          XSS(JS,IS)=ZERO
          DO KS=1,N3S
             IF(JS.LT.KS) THEN
                JPT=IOFF(JS)+KS
             ELSE
                JPT=IOFF(KS)+JS
             ENDIF
             XSS(JS,IS)=XSS(JS,IS)+TMSS(JPT)*VHSS(KS,IS)
          ENDDO
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'Xss matrix'
    do i=1,n3s
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(XSS(j,i),j=1,n3s)
    enddo
    write(6,'(A)') ' '
#endif 


    !
    ! calculate cartesian for the external part
    !
    DO IS=1,NFREQ
       DO JE=1,N3E
          XSE(JE,IS)=ZERO
          DO KS=1,N3S
             XSE(JE,IS)=XSE(JE,IS)-XSS(KS,IS)*AMSE(JE,KS)
          ENDDO
       ENDDO
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'Xse matrix'
    do i=1,n3s
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(XSE(j,i),j=1,n3e)
    enddo
    write(6,'(A)') ' '
#endif 

    DO I=1,NFREQ
       IPT=0  ! N3  index
       JPT=0  ! N3s index
       KPT=0  ! N3e index
       DO J=1,NATOM
          AMS=SQRT(AMASS(J))
          IF(ISLCT(J).EQ.1) THEN
             DDV(IPT+1,I)=XSS(JPT+1,I)*AMS
             DDV(IPT+2,I)=XSS(JPT+2,I)*AMS
             DDV(IPT+3,I)=XSS(JPT+3,I)*AMS
             JPT=JPT+3
          ELSE
             DDV(IPT+1,I)=XSE(KPT+1,I)*AMS
             DDV(IPT+2,I)=XSE(KPT+2,I)*AMS
             DDV(IPT+3,I)=XSE(KPT+3,I)*AMS
             KPT=KPT+3
          ENDIF
          IPT=IPT+3
       ENDDO
       CALL NORMALL(DDV(1,I),NAT3)

       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

#if KEY_DEBUGNMD==1
    write(6,'(A)') 'DDF matrix'
    write(6,'(12E15.5)') (DDF(j),j=1,nfreq)
    write(6,'(A)') ' '

    write(6,'(A)') 'DDV matrix'
    do i=1,nfreq
       FMT='(I5,1X,42F10.5)'
       write (6,FMT) i,(ddv(j,i),j=1,nat3)
    enddo
    write(6,'(A)') ' '
#endif 

    RETURN
  END SUBROUTINE SUBSDS2

  SUBROUTINE ENTROPDS(X,Y,Z,NAT3,NFREQ,AMASS, &
       DDF,NADD,LRAISE)
    !
    ! Do an entropy calculation, if requested
    !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use deriv
  use consta
  use energym
  use stream
  use memory
  use entropy
  use rgyr_mod,only:coriner
  use param_store, only: set_param

    implicit none

    INTEGER NAT3,NFREQ,NADD
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) AMASS(*)
    real(chm_real) DDF(*)

    LOGICAL LRAISE

    ! INTEGER NATOM,NATP,N6,I
    ! INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    ! INTEGER XREF,YREF,ZREF,DXF,DYF,DZF
    INTEGER I,NATOM

    ! Entropy variables
    real(chm_real) CONST, A, EA
    integer,allocatable,dimension(:) :: ISLCT
    INTEGER NISLCT
    integer imd,jmd                            ! yw050725 fix for raise on

    NATOM=NAT3/3
    !
    ! Entropy calculation
    !
    ! CONST = h * c / (k_B * T)
    !     H=6.62606876D0         Planck constant
    !     C=0.299792458D0        Speed of light
    !     KB=1.3806503D0         Bolyzmann constant
    !     R=8.314472D0 / 4.184 = 1.9872065 cal/(mol*K)  Universal gas constant
    !

    ! This IF block is for a test purpose only
    IF(UTEST.GT.0) THEN
       ! Replace CHARMM coordinates and frequencies by the ones read from external file
       DO I=1,NATOM
          READ(UTEST,*) X(I),Y(I),Z(I)
       ENDDO
       DO I=7,NFREQ
          READ(UTEST,*) DDF(I)
       ENDDO
       ! This file should be read no more than once
       UTEST=0
    ENDIF

    ! Vibrational Entropy
    CONST=1.43877522D0 / TK
    SVIB=0.0D0
    ! skip first 6 trivial modes
    imd=7
    IF(NADD.GT.0) then
       imd=7-NADD
       if(imd.le.0) imd=1
    ENDIF
    jmd=nfreq
    if(lraise) then
       imd=1
       if(nfreq.gt.3*natom-6) jmd=3*natom-6
    endif
    DO I=imd,jmd
       A=DDF(I)*CONST
       EA=EXP(A)
       SVIB=SVIB+A/(EA-1.0D0)-LOG(1.0D0-1.0D0/EA)
    ENDDO
    SVIB=SVIB*1.9872065D0
    !
    ! Rotational and Translational Entropies are calculated in CORINER()
    !
    NISLCT = NATOM
    call chmalloc('vibsub.src','ENTROPDS','ISLCT',NISLCT,intg=ISLCT)
    ISLCT(1:NATOM)=1
    CALL CORINER(NATOM,X,Y,Z,AMASS,ISLCT,.TRUE.)
    call chmdealloc('vibsub.src','ENTROPDS','ISLCT',NISLCT,intg=ISLCT)
    SSUM=SROT+SVIB+STRAN

    WRITE(OUTU,'(1X,A,F10.3/1X,A,F10.3)') &
         '  Vibrational   = ', SVIB, &
         '  Total         = ', SSUM

    ! Save Entropy component in CHARMM variables
    call set_param('SVIB',SVIB)
    call set_param('SSUM',SSUM)

    RETURN
  END SUBROUTINE ENTROPDS

  SUBROUTINE VIBFLU(ISTRT,ISTOP,IUNIT,NAT3, &
       DDV,DDM,DDF,DDEV,DDSCR, &
       LNOMA,LNONO,ITYPE,RTYPE,TFREQ, &
       LQUAN,LVERB,ISLCT,X,Y,Z,XNEW,YNEW,ZNEW, &
       XNORM,YNORM,ZNORM,AMASS,LATOM,LIC,LUSER, &
       NRES,IBASE,ATYPE, &
       LENIC,B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    ! COMPUTES ATOM AND IC FLUCTUATIONS
    !
    ! By Bernard R. Brooks   1983
    !
  use chm_kinds
  use intcor_module
  use intcor2,only:writic,bildc,intder
  use stream
  use dimens_fcm
  use memory
  use chutil,only:atomid
  use machutil,only:die
    implicit none

    INTEGER ISTRT,ISTOP,IUNIT,NAT3,ITYPE,NRES
    real(chm_real) RTYPE,TFREQ
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) X(*),Y(*),Z(*),XNEW(*),YNEW(*),ZNEW(*)
    INTEGER ISLCT(*),IBASE(*)
    character(len=*) ATYPE(*)
    LOGICAL LATOM,LIC,LUSER
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)

    LOGICAL LNOMA,LNONO,LQUAN,LVERB,LTFREQ

    ! LOCAL STORAGE
    real(chm_real),allocatable,dimension(:) :: B1ICT,B2ICT,T1ICT,T2ICT,PICT
    INTEGER NATOM,J,INORM
    real(chm_real) UX,UY,UZ,UR,AMASST,DLAST,DPRES
    real(chm_real) CONST,Q,DFLUC,DMRMS,U
    real(chm_real) DELTA
    real(chm_real) AFACT
    CHARACTER(len=8) :: ISID,IRID,IREN,IIAC

    LOGICAL LNOMAX
    DATA LNOMAX/.TRUE./

    ! Conversion factor from CM-1 to Kelvin
    CONST=1.4390
    NATOM=NAT3/3

    IF(IUNIT.EQ.OUTU .AND. PRNLEV.LT.2) RETURN
    IF(IUNIT.NE.OUTU .AND. IOLEV.LT.0)  RETURN

    WRITE(IUNIT,625)
625 FORMAT(/15X,'NORMAL MODE FLUCTUATIONS'/)

    IF(LIC) THEN
       WRITE(IUNIT,18)
18     FORMAT(' INTERNAL COORDINATE FLUCTUATIONS WILL BE COMPUTED')
       call chmalloc('vibsub.src','VIBFLU','B1ICT',LENIC,crl=B1ICT)
       call chmalloc('vibsub.src','VIBFLU','B2ICT',LENIC,crl=B2ICT)
       call chmalloc('vibsub.src','VIBFLU','T1ICT',LENIC,crl=T1ICT)
       call chmalloc('vibsub.src','VIBFLU','T2ICT',LENIC,crl=T2ICT)
       call chmalloc('vibsub.src','VIBFLU','PICT',LENIC,crl=PICT)
       B1ICT(:)=0.0
       B2ICT(:)=0.0
       T1ICT(:)=0.0
       T2ICT(:)=0.0
       PICT(:)=0.0
       DELTA=1.0
    ENDIF

    IF(LUSER) THEN
       WRITE(IUNIT,19)
19     FORMAT(' USER FLUCTUATIONS WILL BE COMPUTED')
       CALL USEFLU(0,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT,IUNIT)
    ENDIF

    IF(LATOM) THEN
       WRITE(IUNIT,21)
21     FORMAT(' ATOM FLUCTUATIONS WILL BE COMPUTED')
       IF(LNOMAX) THEN
          WRITE(IUNIT,23)
23        FORMAT(' NO MASS WEIGHTING WILL BE USED')
       ELSE
          WRITE(IUNIT,24)
24        FORMAT(' MASS WEIGHTING WILL BE USED')
       ENDIF

       AMASST=0.0
       DO J=1,NATOM
          XNEW(J)=0.0
          YNEW(J)=0.0
          ZNEW(J)=0.0
          IF(ISLCT(J).GT.0) THEN
             IF(LNOMAX) THEN
                AMASST=AMASST+1.0
             ELSE
                AMASST=AMASST+AMASS(J)
             ENDIF
          ENDIF
       ENDDO
    ENDIF

    DPRES=0.0
    DO INORM=ISTRT,ISTOP

       CALL MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV(1,INORM),DDM,DDSCR, &
            DDEV(INORM),.TRUE.,ITYPE,RTYPE,LNOMA,LNONO, &
            TFREQ,AFACT,LTFREQ)

       WRITE(IUNIT,45) INORM,DDF(INORM),AFACT
45     FORMAT(' MODE',I4,' FREQ=',F8.2,' FACTOR=',F12.6)

       IF(LQUAN) THEN
          IF(ITYPE.NE.1) CALL DIE
          Q=CONST*DDF(INORM)/RTYPE
          IF(Q.GT.0.0001) THEN
             IF(Q.GT.50.0) THEN
                Q=Q*0.5
             ELSE
                Q=Q*0.5*(EXP(Q)+1.0)/(EXP(Q)-1.0)
             ENDIF
             Q=SQRT(Q)
             WRITE(IUNIT,81) Q
81           FORMAT(' QUANTUM SCALING FACTOR IS',F12.6)
             DO J=1,NATOM
                XNORM(J)=XNORM(J)*Q
                YNORM(J)=YNORM(J)*Q
                ZNORM(J)=ZNORM(J)*Q
             ENDDO
          ENDIF
       ENDIF

       IF (LATOM) THEN
          DFLUC=0.0
          DO J=1,NATOM
             IF(ISLCT(J).GT.0) THEN
                UX=XNORM(J)
                UY=YNORM(J)
                UZ=ZNORM(J)
                UR=UX*UX+UY*UY+UZ*UZ
                UR=0.5*UR
                XNEW(J)=XNEW(J)+0.5*UX*UX
                YNEW(J)=YNEW(J)+0.5*UY*UY
                ZNEW(J)=ZNEW(J)+0.5*UZ*UZ
                IF(LVERB) THEN
                   CALL ATOMID(J,ISID,IRID,IREN,IIAC)
                   WRITE(IUNIT,83) INORM,J,ISID(1:idleng), &
                        IRID(1:idleng),IIAC(1:idleng),UR
83                 FORMAT(2I5,3(1X,A),4F15.6)
                ENDIF
                IF(.NOT.LNOMAX) UR=UR*AMASS(J)
                DFLUC=DFLUC+UR
             ENDIF
          ENDDO
          DMRMS=(DFLUC/AMASST)
          DLAST=DPRES
          DPRES=DPRES+DMRMS

          U=100.0*(DPRES-DLAST)/DPRES
          WRITE(IUNIT,85) DMRMS,DPRES,U
85        FORMAT(15X,' RMS=',F12.6,' TOTAL=',F13.6,' %INCR=',F12.6)
       ENDIF

       IF(LIC) THEN
          CALL INTDER(DELTA,XNORM,YNORM,ZNORM,X,Y,Z,LENIC,.FALSE., &
               B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
          DO J=1,LENIC
             B1ICT(J)=B1ICT(J)+0.5*B1IC(J)*B1IC(J)
             B2ICT(J)=B2ICT(J)+0.5*B2IC(J)*B2IC(J)
             T1ICT(J)=T1ICT(J)+0.5*T1IC(J)*T1IC(J)
             T2ICT(J)=T2ICT(J)+0.5*T2IC(J)*T2IC(J)
             PICT(J)=PICT(J)+0.5*PIC(J)*PIC(J)
          ENDDO
          IF(LVERB) THEN
             DO J=1,LENIC
                B1IC(J)=SQRT(0.5*B1IC(J)*B1IC(J))
                B2IC(J)=SQRT(0.5*B2IC(J)*B2IC(J))
                T1IC(J)=SQRT(0.5*T1IC(J)*T1IC(J))
                T2IC(J)=SQRT(0.5*T2IC(J)*T2IC(J))
                PIC(J)=SQRT(0.5*PIC(J)*PIC(J))
             ENDDO
             CALL WRITIC(1,icr_struct%lenic,&
                  -1,0,IUNIT, &
                  icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR)
          ENDIF
       ENDIF

       IF(LUSER) THEN
          CALL USEFLU(INORM,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT,IUNIT)
       ENDIF

    ENDDO

    IF (LATOM) THEN
       WRITE(IUNIT,82) SQRT(DPRES)
82     FORMAT(' AVERAGE FLUCTUATION OVER SELECTED ATOMS IS',F12.4)
    ENDIF

    IF(LIC) THEN
       WRITE(IUNIT,84)
84     FORMAT(/' TOTAL FLUCTUATION FOR INTERNAL COORDINATES')
       DO J=1,LENIC
          B1IC(J)=SQRT(B1ICT(J))
          B2IC(J)=SQRT(B2ICT(J))
          T1IC(J)=SQRT(T1ICT(J))
          T2IC(J)=SQRT(T2ICT(J))
          PIC(J)=SQRT(PICT(J))
       ENDDO
       CALL WRITIC(1,icr_struct%LENIC,-1,0,IUNIT, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR)
       call chmdealloc('vibsub.src','VIBFLU','PICT',LENIC,crl=PICT)
       call chmdealloc('vibsub.src','VIBFLU','T2ICT',LENIC,crl=T2ICT)
       call chmdealloc('vibsub.src','VIBFLU','T1ICT',LENIC,crl=T1ICT)
       call chmdealloc('vibsub.src','VIBFLU','B2ICT',LENIC,crl=B2ICT)
       call chmdealloc('vibsub.src','VIBFLU','B1ICT',LENIC,crl=B1ICT)
    ENDIF

    IF(LUSER) THEN
       CALL USEFLU(-1,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT,IUNIT)
    ENDIF

    RETURN
  END SUBROUTINE VIBFLU

  SUBROUTINE USEFLU(IMODE,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT,IUNIT)
    !
    ! THIS ROUTINE PROCESSES A USER SUPPLIED FLUCTUATION SPECIFICATION
    !
    ! IMODE IS PROCESSED AS;
    !   0 - INITIALIZATION CALL
    !   N - LOOP CALL (GIVING MODE NUMBER)
    !  -1 - TERMINATION CALL
    !
  use chm_kinds
  use exfunc
  use memory
    implicit none

    INTEGER IMODE,NATOM
    real(chm_real) X(*),Y(*),Z(*),XNORM(*),YNORM(*),ZNORM(*)
    INTEGER ISLCT(*)
    INTEGER IUNIT
    !
    integer,allocatable,dimension(:) :: IAT
    real(chm_real),allocatable,dimension(:) :: IDDR
    INTEGER NIAT,I,NN
    SAVE NIAT,IAT,IDDR

    ! mkg 2009: reordered logic
    IF (IMODE == 0) THEN
       ! process-flucuation-initialization
       NIAT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) NIAT=NIAT+1
       ENDDO
       NN=(NIAT*(NIAT+1))/2
       call chmalloc('vibsub.src','USEFLU','IAT',NIAT,intg=IAT)
       call chmalloc('vibsub.src','USEFLU','IDDR',NN,crl=IDDR)
    ENDIF

    ! Unconditional: process-this-normal-mode
    CALL USEFL2(IMODE,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT, &
         IAT,NIAT,IDDR,IUNIT)

    IF (IMODE == -1) THEN
       ! process-fluctuation-termination
       NN=(NIAT*(NIAT+1))/2
       call chmdealloc('vibsub.src','USEFLU','IAT',NIAT,intg=IAT)
       call chmdealloc('vibsub.src','USEFLU','IDDR',NN,crl=IDDR)
    ENDIF

    RETURN
  END SUBROUTINE USEFLU

  SUBROUTINE USEFL2(IMODE,X,Y,Z,XNORM,YNORM,ZNORM,NATOM,ISLCT, &
       IAT,NIAT,DDR,IUNIT)
    !
    ! THIS ROUTINE PROCESSES A USER SUPPLIED FLUCTUATION SPECIFICATION
    !
    ! IMODE IS PROCESSED AS;
    !   0 - INITIALIZATION CALL
    !   N - LOOP CALL (GIVING MODE NUMBER)
    !  -1 - TERMINATION CALL
    !
  use chm_kinds
    implicit none

    INTEGER IMODE,NATOM
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER ISLCT(*)
    INTEGER IAT(*),NIAT
    real(chm_real) DDR(*)
    INTEGER IUNIT

    INTEGER I,J,IPT,II,JJ,IIP,JJP
    real(chm_real) DFX,DFY,DFZ,HX,HY,HZ,HR,A

    IF(IMODE.EQ.0) THEN
       ! process-flucuation-initialization-user
       J=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             J=J+1
             IAT(J)=I
          ENDIF
       ENDDO

       IPT=0
       DO I=1,NIAT
          DO J=1,I
             IPT=IPT+1
             DDR(IPT)=0.0
          ENDDO
       ENDDO

    ELSE IF(IMODE.EQ.-1) THEN
       ! process-fluctuation-termination-user
       DO I=1,NIAT
          II=IAT(I)
          IIP=(I*(I+1))/2
          DDR(IIP)=DDR(IIP)/2.0
          DDR(IIP)=SQRT(DDR(IIP))
          A=DDR(IIP)
          WRITE(IUNIT,31) I,II,A
31        FORMAT(2I5,F20.8)
       ENDDO

       IPT=0
       DO I=1,NIAT
          II=IAT(I)
          IIP=(I*(I+1))/2
          DO J=1,I
             JJ=IAT(J)
             JJP=(J*(J+1))/2
             IPT=IPT+1
             IF(II.NE.JJ) THEN
                HX=X(II)-X(JJ)
                HY=Y(II)-Y(JJ)
                HZ=Z(II)-Z(JJ)
                HR=SQRT(HX*HX+HY*HY+HZ*HZ)
                DDR(IPT)=DDR(IPT)/(DDR(IIP)*DDR(JJP))
                WRITE(IUNIT,26) I,J,II,JJ,DDR(IPT),HR
26              FORMAT(4I5,3F20.8)
             ENDIF
          ENDDO
       ENDDO

    ELSE
       ! process-this-normal-mode-user
       IPT=0
       DO I=1,NIAT
          II=IAT(I)
          DO J=1,I
             JJ=IAT(J)
             IPT=IPT+1

             DFX=XNORM(II)*XNORM(JJ)
             DFY=YNORM(II)*YNORM(JJ)
             DFZ=ZNORM(II)*ZNORM(JJ)
             DDR(IPT)=DDR(IPT)+DFX+DFY+DFZ
          ENDDO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE USEFL2

  SUBROUTINE VPAFL0(ISTRT,ISTOP,IUNIT,NAT3, &
       DDV,DDM,DDF,DDEV,DDSCR, &
       LNOMA,LNONO,ITYPE,RTYPE,TFREQ, &
       LQUAN,LVERB,ISLCT,XX,YY,ZZ,XY,XZ,YZ, &
       XNORM,YNORM,ZNORM,AMASS,LATOM,LUSER,LPAX,LGROU,LNOMS, &
       LSAVE,LCONT,NRES,IBASE,ATYPE)
    !
    ! Added October 1985 by B. Tidor.  Simply allocates space for
    ! arrays on STACK which are needed by VPAFL.
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream

    implicit none

    INTEGER ISTRT,ISTOP,IUNIT,NAT3,ITYPE,NRES
    real(chm_real) RTYPE,TFREQ
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) XX(*),YY(*),ZZ(*),XY(*),XZ(*),YZ(*)
    INTEGER ISLCT(*),IBASE(*)
    character(len=*) ATYPE(*)

    LOGICAL LNOMA,LNONO,LQUAN,LVERB
    LOGICAL LATOM,LUSER,LPAX,LGROU,LNOMS
    LOGICAL LSAVE,LCONT

    IF(IUNIT.EQ.OUTU .AND. PRNLEV.LT.2) RETURN
    IF(IUNIT.NE.OUTU .AND. IOLEV.LT.0)  RETURN

    CALL VPAFL(ISTRT,ISTOP,IUNIT,NAT3, &
         DDV,DDM,DDF,DDEV,DDSCR, &
         LNOMA,LNONO,ITYPE,RTYPE,TFREQ,LQUAN,LVERB, &
         ISLCT,XX,YY,ZZ,XY,XZ,YZ, &
         XNORM,YNORM,ZNORM,AMASS,LATOM,LUSER,LPAX,LGROU,LNOMS, &
         LSAVE,LCONT,NRES,IBASE,ATYPE)

    RETURN
  END SUBROUTINE VPAFL0

  SUBROUTINE VPAFL(ISTRT,ISTOP,IUNIT,NAT3, &
       DDV,DDM,DDF,DDEV,DDSCR, &
       LNOMA,LNONO,ITYPE,RTYPE,TFREQ,LQUAN,LVERB, &
       ISLCT,XX,YY,ZZ,XY,XZ,YZ, &
       XNORM,YNORM,ZNORM,AMASS,LATOM,LUSER,LPAX,LGROU,LNOMS, &
       LSAVE,LCONT,NRES,IBASE,ATYPE)
    !
    ! Added October 1985 by B. Tidor.  Adapted from routine VIBFLU
    ! (work of B. R. Brooks).  Computes atomic fluctuations from normal
    ! mode eigenvalues and eigenvectors and converts to principal axes
    ! if requested.
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  use stream
  use chutil,only:atomid
  use machutil,only:die
  use usermod,only: uspafl
    implicit none

    INTEGER ISTRT,ISTOP,IUNIT,NAT3,ITYPE,NRES
    real(chm_real) RTYPE,TFREQ
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) XX(*),YY(*),ZZ(*),XY(*),XZ(*),YZ(*)
    INTEGER ISLCT(*),IBASE(*)
    character(len=*) ATYPE(*)

    LOGICAL LNOMA,LNONO,LQUAN,LVERB,LTFREQ

    real(chm_real),allocatable,dimension(:) :: RR,FLU,VEC,EV
    real(chm_real),allocatable,dimension(:,:) :: PAX
    INTEGER NATOM,J,INORM
    real(chm_real) UX,UY,UZ,UR,AMASST,DLAST,DPRES
    real(chm_real) CONST,Q,DFLUC,DMRMS
    real(chm_real) DMASS,TSLCT
    real(chm_real) AFACT
    CHARACTER(len=8) :: ISID,IRID,IREN,IIAC
    real(chm_real),allocatable,dimension(:) :: A,B,P,W
    real(chm_real),allocatable,dimension(:) :: TA,TB,Y
    INTEGER NX,NFR,NA,I

    LOGICAL LNOMAX,LATOM,LUSER,LPAX,LGROU,LNOMS
    LOGICAL LSAVE,LCONT
    SAVE AMASST,DPRES
    DATA LNOMAX /.TRUE./
    !yw...05-Apr-91
    !
    ! CONVERSION FACTOR FROM CM-1 TO KELVIN
    CONST=1.4390
    NATOM=NAT3/3

    WRITE(IUNIT,625)
625 FORMAT(/15X,'FLUCTUATIONS FROM NORMAL MODES'/)

    IF(.NOT.LCONT) THEN

       IF(LUSER) THEN
          WRITE(IUNIT,19)
19        FORMAT(' USER FLUCTUATIONS WILL BE COMPUTED')
          CALL USPAFL(0,XX,YY,ZZ,XY,XZ,YZ, &
               XNORM,YNORM,ZNORM,NATOM,ISLCT)
       ENDIF

       IF(LATOM) THEN
          WRITE(IUNIT,21)
21        FORMAT(' ATOMIC FLUCTUATIONS WILL BE COMPUTED')
          AMASST=0.0

          DO J=1, NATOM
             XX(J)=0.0D0
             YY(J)=0.0D0
             ZZ(J)=0.0D0
             XY(J)=0.0D0
             XZ(J)=0.0D0
             YZ(J)=0.0D0
             IF(ISLCT(J).GT.0) THEN
                IF(LNOMAX) THEN
                   AMASST=AMASST+1.0
                ELSE
                   AMASST=AMASST+AMASS(J)
                ENDIF
             ENDIF
          ENDDO
       ENDIF

       IF(LGROU) THEN
          IF (LNOMA) WRITE(IUNIT,25)
25        FORMAT(' **** WARNING **** NOMASS IN VIBRAN NOT AVAILABLE', &
               ' WITH THIS OPERATION.')
          IF(LNOMS) THEN
             WRITE(IUNIT,26)
26           FORMAT('0THE FLUCTUATION OF THE NON-MASS-WEIGHTED', &
                  ' CENTER OF THE FOLLOWING ATOMS WILL BE COMPUTED')
          ELSE
             WRITE(IUNIT,27)
27           FORMAT('0THE FLUCTUATION OF THE MASS-WEIGHTED CENTER OF', &
                  ' THE FOLLOWING ATOMS WILL BE COMPUTED')
          ENDIF
          TSLCT=0.0D0
          DMASS=0.0D0
          DO J=1,NATOM
             IF (ISLCT(J).GT.0) THEN
                TSLCT=TSLCT+1.0D0
                DMASS=DMASS+1.0D0/(DDM(J)*DDM(J))
                CALL ATOMID(J,ISID,IRID,IREN,IIAC)
                WRITE(IUNIT,28) J,ISID(1:idleng), &
                     IRID(1:idleng),IIAC(1:idleng)
28              FORMAT(I5,3(1X,A))
             ENDIF
          ENDDO
          IF (.NOT.LNOMS) TSLCT=DMASS
          XX(1)=0.0D0
          YY(1)=0.0D0
          ZZ(1)=0.0D0
          XY(1)=0.0D0
          XZ(1)=0.0D0
          YZ(1)=0.0D0
       ENDIF

       DPRES=0.0
    ENDIF

    DO INORM=ISTRT,ISTOP

       IF(DDF(INORM).GT.TFREQ) THEN

          CALL MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV(1,INORM),DDM,DDSCR, &
               DDEV(INORM),.TRUE.,ITYPE,RTYPE,.TRUE.,LNONO, &
               TFREQ,AFACT,LTFREQ)

          !     WRITE(IUNIT,45) INORM,DDF(INORM),AFACT
          ! 45  FORMAT(' MODE', I4, ' FREQ=', F8.2, ' FACTOR=', F12.6)
          !
          IF(LQUAN) THEN
             IF (ITYPE.NE.1) CALL DIE
             Q=CONST*DDF(INORM)/RTYPE
             IF (Q.GT.0.0001) THEN
                IF(Q.GT.50.0) THEN
                   Q=Q*0.5
                ELSE
                   Q=Q*0.5*(EXP(Q)+1.0)/(EXP(Q)-1.0)
                ENDIF
                Q=SQRT(Q)
                !     WRITE(IUNIT,81) Q
                ! 81  FORMAT(' QUANTUM SCALING FACTOR IS', F12.6)
                DO J=1,NATOM
                   XNORM(J)=XNORM(J)*Q
                   YNORM(J)=YNORM(J)*Q
                   ZNORM(J)=ZNORM(J)*Q
                ENDDO
             ENDIF
          ENDIF

          IF(LATOM) THEN
             DFLUC=0.0
             DO J=1,NATOM
                IF (ISLCT(J).GT.0) THEN
                   UX=XNORM(J)
                   UY=YNORM(J)
                   UZ=ZNORM(J)
                   UR=UX*UX+UY*UY+UZ*UZ
                   UR=0.5*UR
                   XX(J)=XX(J)+0.5*UX*UX
                   YY(J)=YY(J)+0.5*UY*UY
                   ZZ(J)=ZZ(J)+0.5*UZ*UZ
                   XY(J)=XY(J)+0.5*UX*UY
                   XZ(J)=XZ(J)+0.5*UX*UZ
                   YZ(J)=YZ(J)+0.5*UY*UZ

                   IF (LVERB) THEN
                      CALL ATOMID(J,ISID,IRID,IREN,IIAC)
                      WRITE(IUNIT,83) INORM,J,ISID(1:idleng), &
                           IRID(1:idleng),IIAC(1:idleng),UR
83                    FORMAT(2I5, 3(1X,A), 4F15.6)
                   ENDIF
                   IF(.NOT.LNOMAX) UR=UR*AMASS(J)
                   DFLUC=DFLUC+UR
                ENDIF
             ENDDO
             DMRMS=(DFLUC/AMASST)
             DLAST=DPRES
             DPRES=DPRES+DMRMS

             !     U=100.0*(DPRES-DLAST)/DPRES
             !     WRITE(IUNIT,85) DMRMS, DPRES, U
             ! 85  FORMAT(15X,' RMS=',F12.6,' TOTAL=',F13.6,' %INCR=',F12.6)
          ENDIF

          IF(LUSER) THEN
             CALL USPAFL(INORM,XX,YY,ZZ,XY,XZ,YZ,XNORM,YNORM,ZNORM, &
                  NATOM,ISLCT)
          ENDIF

          IF(LGROU) THEN
             UX=0.0D0
             UY=0.0D0
             UZ=0.0D0
             DO J=1,NATOM
                IF (ISLCT(J).GT.0) THEN
                   IF(LNOMS) THEN
                      UX=UX+XNORM(J)*DDM(J)
                      UY=UY+YNORM(J)*DDM(J)
                      UZ=UZ+ZNORM(J)*DDM(J)
                   ELSE
                      UX=UX+XNORM(J)/DDM(J)
                      UY=UY+YNORM(J)/DDM(J)
                      UZ=UZ+ZNORM(J)/DDM(J)
                   ENDIF
                ENDIF
             ENDDO
             XX(1)=XX(1)+0.5*UX*UX
             YY(1)=YY(1)+0.5*UY*UY
             ZZ(1)=ZZ(1)+0.5*UZ*UZ
             XY(1)=XY(1)+0.5*UX*UY
             XZ(1)=XZ(1)+0.5*UX*UZ
             YZ(1)=YZ(1)+0.5*UY*UZ
          ENDIF

       ELSE
          WRITE(IUNIT,79) INORM
79        FORMAT('0MODE ', I5, ' SKIPPED IN SUMMATION --', &
               ' FREQ LESS THAN TFREQ')
       ENDIF
    ENDDO

    IF(.NOT.LSAVE) THEN

       call chmalloc('vibsub.src','VPAFL','RR',NATOM,crl=RR)
       call chmalloc('vibsub.src','VPAFL','PAX',9,NATOM,crl=PAX)
       IF(LATOM.AND.LPAX) THEN
          call chmalloc('vibsub.src','VPAFL','FLU',9,crl=FLU)
          call chmalloc('vibsub.src','VPAFL','VEC',9,crl=VEC)
          call chmalloc('vibsub.src','VPAFL','EV',3,crl=EV)
          call chmalloc('vibsub.src','VPAFL','A',4,crl=A)
          call chmalloc('vibsub.src','VPAFL','B',4,crl=B)
          call chmalloc('vibsub.src','VPAFL','P',4,crl=P)
          call chmalloc('vibsub.src','VPAFL','W',4,crl=W)
          call chmalloc('vibsub.src','VPAFL','TA',4,crl=TA)
          call chmalloc('vibsub.src','VPAFL','TB',4,crl=TB)
          call chmalloc('vibsub.src','VPAFL','Y',4,crl=Y)

          NX=3
          NFR=3
          NA=0
          DO J=1,NATOM
             IF(ISLCT(J).GT.0) THEN
                ! LOAD-AND-DIAGONALIZE-FLU-MATRIX
                FLU(1)=XX(J)
                FLU(2)=XY(J)
                FLU(3)=XZ(J)
                FLU(4)=YY(J)
                FLU(5)=YZ(J)
                FLU(6)=ZZ(J)
                FLU(7)=0.0D0
                FLU(8)=0.0D0
                FLU(9)=0.0D0
                CALL DIAGQ(NX,NFR,FLU,VEC,A,B,P, &
                     W,EV,TA,TB,Y,NA)

                RR(J)=SQRT(ABS(EV(1))+ABS(EV(2))+ABS(EV(3)))*DDM(J)
                XX(J)=SQRT( ABS( EV(1) ) )*DDM(J)
                YY(J)=SQRT( ABS( EV(2) ) )*DDM(J)
                ZZ(J)=SQRT( ABS( EV(3) ) )*DDM(J)
                IF (EV(1).LT.0.0) XX(J)=-XX(J)
                IF (EV(2).LT.0.0) YY(J)=-YY(J)
                IF (EV(3).LT.0.0) ZZ(J)=-ZZ(J)

                DO I=1,9
                   PAX(I,J)=VEC(I)
                ENDDO

             ENDIF
          ENDDO

          call chmdealloc('vibsub.src','VPAFL','Y',4,crl=Y)
          call chmdealloc('vibsub.src','VPAFL','TB',4,crl=TB)
          call chmdealloc('vibsub.src','VPAFL','TA',4,crl=TA)
          call chmdealloc('vibsub.src','VPAFL','W',4,crl=W)
          call chmdealloc('vibsub.src','VPAFL','P',4,crl=P)
          call chmdealloc('vibsub.src','VPAFL','B',4,crl=B)
          call chmdealloc('vibsub.src','VPAFL','A',4,crl=A)
          call chmdealloc('vibsub.src','VPAFL','EV',3,crl=EV)
          call chmdealloc('vibsub.src','VPAFL','VEC',9,crl=VEC)
          call chmdealloc('vibsub.src','VPAFL','FLU',9,crl=FLU)

          ! print-pax-atomic-fluctuations
          WRITE(IUNIT,87)
87        FORMAT('0ANISOTROPIC FLUCTUATIONS'/'0',23X,'F(R)   F(P1)  ', &
               ' F(P2)   F(P3)   P1(X)   P1(Y)   P1(Z)   P2(X)  ', &
               ' P2(Y)   P2(Z)   P3(X)   P3(Y)   P3(Z)' / ' ')
          DO J=1,NATOM
             IF(ISLCT(J).GT.0) THEN
                CALL ATOMID(J,ISID,IRID,IREN,IIAC)
                WRITE(IUNIT,90) J,ISID(1:idleng),IRID(1:idleng), &
                     IIAC(1:idleng),RR(J), &
                     XX(J),YY(J),ZZ(J),(PAX(I,J),I=1,9)
90              FORMAT(I5,3(1X,A),13F8.4)
             ENDIF
          ENDDO
       ENDIF

       IF(.NOT.LPAX.AND.LATOM) THEN
          WRITE(IUNIT,92)
92        FORMAT('0ISOTROPIC FLUCTUATIONS'/'0',28X,'F(R)',11X,'F(X)', &
               11X,'F(Y)',11X,'F(Z)' / ' ')
          DO J=1,NATOM
             IF(ISLCT(J).GT.0) THEN
                RR(J)=SQRT(XX(J)+YY(J)+ZZ(J))*DDM(J)
                XX(J)=SQRT(XX(J))*DDM(J)
                YY(J)=SQRT(YY(J))*DDM(J)
                ZZ(J)=SQRT(ZZ(J))*DDM(J)
                CALL ATOMID(J,ISID,IRID,IREN,IIAC)
                WRITE(IUNIT,95) J,ISID(1:idleng),IRID(1:idleng), &
                     IIAC(1:idleng),RR(J),XX(J),YY(J),ZZ(J)
95              FORMAT(I5, 3(1X,A), 4F15.6)
             ENDIF
          ENDDO
       ENDIF

       IF (LUSER) THEN
          CALL USPAFL(-1,XX,YY,ZZ,XY,XZ,YZ, &
               XNORM,YNORM,ZNORM,NATOM,ISLCT)
       ENDIF

       IF (LGROU) THEN
          RR(1)=SQRT( XX(1)+YY(1)+ZZ(1) )/TSLCT
          IF(LPAX) THEN
             call chmalloc('vibsub.src','VPAFL','FLU',9,crl=FLU)
             call chmalloc('vibsub.src','VPAFL','VEC',9,crl=VEC)
             call chmalloc('vibsub.src','VPAFL','EV',3,crl=EV)
             call chmalloc('vibsub.src','VPAFL','A',4,crl=A)
             call chmalloc('vibsub.src','VPAFL','B',4,crl=B)
             call chmalloc('vibsub.src','VPAFL','P',4,crl=P)
             call chmalloc('vibsub.src','VPAFL','W',4,crl=W)
             call chmalloc('vibsub.src','VPAFL','TA',4,crl=TA)
             call chmalloc('vibsub.src','VPAFL','TB',4,crl=TB)
             call chmalloc('vibsub.src','VPAFL','Y',4,crl=Y)

             NX=3
             NFR=3
             NA=0
             FLU(1)=XX(1)
             FLU(2)=XY(1)
             FLU(3)=XZ(1)
             FLU(4)=YY(1)
             FLU(5)=YZ(1)
             FLU(6)=ZZ(1)
             FLU(7)=0.0D0
             FLU(8)=0.0D0
             FLU(9)=0.0D0
             CALL DIAGQ(NX,NFR,FLU,VEC,A,B,P, &
                  W,EV,TA,TB,Y,NA)
             XX(1)=SQRT(ABS(EV(1)))/TSLCT
             YY(1)=SQRT(ABS(EV(2)))/TSLCT
             ZZ(1)=SQRT(ABS(EV(3)))/TSLCT
             IF (EV(1).LT.0.0) XX(1)=-XX(1)
             IF (EV(2).LT.0.0) YY(1)=-YY(1)
             IF (EV(3).LT.0.0) ZZ(1)=-ZZ(1)
             DO I=1,9
                PAX(I,1)=VEC(I)
             ENDDO
             WRITE(IUNIT,87)
             WRITE(IUNIT,100) RR(1),XX(1),YY(1),ZZ(1),(PAX(I,1),I=1,9)
100          FORMAT(20X,13F8.4)

             call chmdealloc('vibsub.src','VPAFL','Y',4,crl=Y)
             call chmdealloc('vibsub.src','VPAFL','TB',4,crl=TB)
             call chmdealloc('vibsub.src','VPAFL','TA',4,crl=TA)
             call chmdealloc('vibsub.src','VPAFL','W',4,crl=W)
             call chmdealloc('vibsub.src','VPAFL','P',4,crl=P)
             call chmdealloc('vibsub.src','VPAFL','B',4,crl=B)
             call chmdealloc('vibsub.src','VPAFL','A',4,crl=A)
             call chmdealloc('vibsub.src','VPAFL','EV',3,crl=EV)
             call chmdealloc('vibsub.src','VPAFL','VEC',9,crl=VEC)
             call chmdealloc('vibsub.src','VPAFL','FLU',9,crl=FLU)
          ELSE
             XX(1)=SQRT(XX(1))/TSLCT
             YY(1)=SQRT(YY(1))/TSLCT
             ZZ(1)=SQRT(ZZ(1))/TSLCT
             WRITE(IUNIT,92)
             WRITE(IUNIT,101) RR(1),XX(1),YY(1),ZZ(1)
101          FORMAT(20X,4F15.6)
          ENDIF
       ENDIF
       call chmdealloc('vibsub.src','VPAFL','PAX',9,NATOM,crl=PAX)
       call chmdealloc('vibsub.src','VPAFL','RR',NATOM,crl=RR)

    ENDIF
    RETURN
  END SUBROUTINE VPAFL

  SUBROUTINE EXPLNM(X,Y,Z,XNEW,YNEW,ZNEW,XNORM,YNORM,ZNORM, &
       XTEMP,YTEMP,ZTEMP,NSTEP,BNBND,BIMAG, &
       ESTORE,RSTORE,WSTORE,ISTORE,IUNIT,AFACT,LADJU,RBOLT,RGAUS, &
       DDEV,DDF,ISTRT,LSHAK,NATOM,AMASS,IMOVE,ISKP)
    !
    ! THIS ROUTINE EXPLORES THE ENERGY SURFACE FOR A PARTICULAR
    ! MOTION. ONLY A ONE DIMENTIONAL LINEAR SEARCH IS CURRENTLY
    ! PRESENT
    !
    ! By Bernard R. Brooks   1982
    !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use deriv
  use energym
  use consta
  use stream
  use holonom,only:holonoma

    implicit none

    INTEGER NSTEP,IUNIT,ISTRT,NATOM
    real(chm_real) RBOLT,RGAUS,AFACT
    type(nonbonddatastructure) :: BNBND
    type(imagedatastructure) :: BIMAG
    real(chm_real) AMASS(*)
    INTEGER IMOVE(*),ISKP(*)

    real(chm_real) :: X(:), Y(:), Z(:), XNEW(:), YNEW(:), ZNEW(:)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) XTEMP(*),YTEMP(*),ZTEMP(*)
    real(chm_real) ESTORE(*),RSTORE(*),WSTORE(*),DDEV(*),DDF(*)
    INTEGER ISTORE(*)
    LOGICAL LADJU,LSHAK,ERROR,QOK
    real(chm_real) EMINX,CBOLT,CGAUS

    INTEGER NSOL
    PARAMETER (NSOL=3)
    real(chm_real) ASOL(NSOL),AWORK(NSOL,NSOL+1)
    INTEGER IWORK(NSOL,2)
    real(chm_real) FACT
    INTEGER ISTEP,I,J

    EMINX=1.D10
    CBOLT=0.0
    IF(RBOLT.NE.0.0) CBOLT=1.0/(KBOLTZ*RBOLT)
    CGAUS=-RGAUS

    DO ISTEP=1,NSTEP
       IF(NSTEP.GT.1) THEN
          FACT=2*(NSTEP-ISTEP)
          FACT=1.0-FACT/(NSTEP-1)
       ELSE
          FACT=1.0
       ENDIF
       DO J=1,NATOM
          XNEW(J)=X(J)+XNORM(J)*FACT
          YNEW(J)=Y(J)+YNORM(J)*FACT
          ZNEW(J)=Z(J)+ZNORM(J)*FACT
       ENDDO

       IF(LSHAK) THEN
          DO I=1,NATOM
             XTEMP(I)=XNEW(I)
             YTEMP(I)=YNEW(I)
             ZTEMP(I)=ZNEW(I)
          ENDDO
          CALL HOLONOMA(XNEW,YNEW,ZNEW,XTEMP,YTEMP,ZTEMP, &
               .TRUE.,.FALSE.,QOK)
       ENDIF

       CALL ENERGY(XNEW,YNEW,ZNEW,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_PARALLEL==1
       ! No forces needed, but get them anyway for a consistent printout.
       !!#!! MH 2024: Do we need this ?? Check for printouts...
       CALL VDGBR(DX,DY,DZ,1)
#endif 

       ESTORE(ISTEP)=EPROP(EPOT)
       RSTORE(ISTEP)=FACT
       WSTORE(ISTEP)=FACT*AFACT
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,35) FACT,WSTORE(ISTEP)
35        FORMAT(' RELATIVE STEP FACTOR:',F12.6,' GLOBAL FACTOR', &
               F12.6)
          CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
               ISTEP, ZERO, WSTORE(ISTEP), .TRUE.)
       ENDIF
       IF(IUNIT.GE.0 .AND. IOLEV.GT.0) WRITE(IUNIT,48) WSTORE(ISTEP), &
            ESTORE(ISTEP)
48     FORMAT(2F20.6)
       IF(EPROP(EPOT).LT.EMINX) EMINX=EPROP(EPOT)
    ENDDO

    IF(LADJU) THEN
       DO ISTEP=1,NSTEP
          WSTORE(ISTEP)=EXP(CBOLT*(EMINX-ESTORE(ISTEP)))
          WSTORE(ISTEP)=WSTORE(ISTEP)*EXP(CGAUS*RSTORE(ISTEP)**2)
          RSTORE(ISTEP)=RSTORE(ISTEP)*AFACT
          ISTORE(ISTEP)=0

          IF(PRNLEV.GE.2) WRITE(OUTU,88) ISTEP,RSTORE(ISTEP), &
               ESTORE(ISTEP),WSTORE(ISTEP)
88        FORMAT(I5,3F10.5)
       ENDDO
       !
       ! NOW DO LEAST SQUARES FITTING
       ! ESTORE - ENERGIES
       ! RSTORE - DISTANCES
       ! WSTORE - LEAST SQUARES WEIGHTINGS
       !
       CALL LSSOLV(RSTORE,ISTORE,ESTORE,WSTORE,NSTEP,ASOL, &
            AWORK,IWORK,NSOL)

       IF(PRNLEV.GE.2) WRITE(OUTU,84)
       IF(PRNLEV.GE.2) WRITE(OUTU,85)  'OLD',DDEV(ISTRT),DDF(ISTRT)
84     FORMAT(' FREQUENCIES ARE ADJUSTED.  EIGENVALUE   FREQUENCY')
85     FORMAT(24X,A4,F12.6,F12.2)
       DDEV(ISTRT)=ASOL(1)*2.0
       DDF(ISTRT)=CNVFRQ*SQRT(ABS(DDEV(ISTRT)))
       IF(DDEV(ISTRT).LT.0.0) DDF(ISTRT)=-DDF(ISTRT)
       IF(PRNLEV.GE.2) WRITE(OUTU,85) 'NEW',DDEV(ISTRT),DDF(ISTRT)
    ENDIF

    IF(IUNIT.GE.0) CALL VCLOSE(IUNIT,'KEEP',ERROR)
    RETURN
  END SUBROUTINE EXPLNM

  SUBROUTINE PED(X,Y,Z,DX,DY,DZ,TOL,WBOND,WTHETA,WPHI,WIMPHI &
#if KEY_CMAP==1
       ,WCRTERM &  
#endif
       )
    !
    ! THIS ROUTINE COMPUTES THE POTENTIAL ENERGY DISTRIBUTION OF
    ! A NORMAL MODE WITHIN THE INTERNAL ENERGY TERMS.
    !
    ! P.E.D FOR ith INTERNAL COORD. S
    !
    !    = Lik*Lik*Fi
    !
    ! WHERE Fi IS THE FORCE CONSTANT
    !       Lik IS THE ikth ELEMENT OF THE MATRIX
    !
    !    S = L * Q (Q=NORMAL COORD)
    !
    ! FOR Q EXPRESSED IN CARTESIAN BASIS, L IS JUST THE B MATRIX
    !
    ! REF:  Y.Morino and K.Kuchitsu, J.Chem.Phys. 20:1809
    !
    ! WRITTEN BY B. BROOKS AND A. LEE  5/83
    !
  use chm_kinds
  use dimens_fcm
  use psf
  use number
  use param
  use code
  use stream
  use exfunc
  use intcor2,only:geticd
    implicit none

    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real) TOL
    real(chm_real) WBOND(*),WTHETA(*),WPHI(*),WIMPHI(*)
#if KEY_CMAP==1
    real(chm_real) WCRTERM(*)
#endif 

    ! local storage
    INTEGER IBOND,ITHETA,IPHI,IIMPHI,I,J,K,L,IPER
#if KEY_CMAP==1
    INTEGER ICRTERM
    INTEGER I1,J1,K1,L1,I2,J2,K2,L2
#endif 
    real(chm_real) WBTOT,WTTOT,WPTOT,WITOT,WTOT,WBTOL,WTTOL, &
         WPTOL,WITOL
#if KEY_CMAP==1
    real(chm_real) WCTOT
#endif 
    real(chm_real) PIRAD,DERIV,XX,FACTOR

    PIRAD=(ACOS(-1.D0)/180.0)**2
    !
    ! BOND SECTION
    !
    WBTOT=0.0
    DO IBOND=1,NBOND
       I=IB(IBOND)
       J=JB(IBOND)
       CALL GETICD(I,J,0,0,.FALSE.,DERIV,XX,XX,XX,XX,X,Y,Z,DX,DY,DZ)
       WBOND(IBOND)=DERIV*DERIV*CBC(ICB(IBOND))
       WBTOT=WBTOT+WBOND(IBOND)
    ENDDO
    !
    ! ANGLE SECTION
    !
    WTTOT=0.0
    DO ITHETA=1,NTHETA
       I=IT(ITHETA)
       J=JT(ITHETA)
       K=KT(ITHETA)
       CALL GETICD(I,J,K,0,.FALSE.,XX,DERIV,XX,XX,XX,X,Y,Z,DX,DY,DZ)
       WTHETA(ITHETA)=DERIV*DERIV*CTC(ICT(ITHETA))*PIRAD
       WTTOT=WTTOT+WTHETA(ITHETA)
    ENDDO
    !
    ! DIHEDRAL SECTION
    !
    WPTOT=0.0
    DO IPHI=1,NPHI
       I=IP(IPHI)
       J=JP(IPHI)
       K=KP(IPHI)
       L=LP(IPHI)
       CALL GETICD(I,J,K,L,.FALSE.,XX,XX,DERIV,XX,XX,X,Y,Z,DX,DY,DZ)
       IPER=CPD(ICP(IPHI))
       IF(IPER.EQ.0) IPER=1
       WPHI(IPHI)=DERIV*DERIV*CPC(ICP(IPHI))*PIRAD*IPER**2
       WPTOT=WPTOT+WPHI(IPHI)
    ENDDO
    !
    ! IMPROPER DIHEDRAL SECTION
    !
    WITOT=0.0
    DO IIMPHI=1,NIMPHI
       I=IM(IIMPHI)
       J=JM(IIMPHI)
       K=KM(IIMPHI)
       L=LM(IIMPHI)
       CALL GETICD(I,J,K,L,.FALSE.,XX,XX,DERIV,XX,XX,X,Y,Z,DX,DY,DZ)
       IPER=CID(ICI(IIMPHI))
       IF(IPER.EQ.0) IPER=1
       WIMPHI(IIMPHI)=DERIV*DERIV*CIC(ICI(IIMPHI))*PIRAD*IPER**2
       WITOT=WITOT+WIMPHI(IIMPHI)
    ENDDO

#if KEY_CMAP==1
    !
    ! CROSS-TERM DIHEDRAL SECTION
    !
    WCTOT=0.0
    DO ICRTERM=1,NCRTERM
       I1=I1CT(ICRTERM)
       J1=J1CT(ICRTERM)
       K1=K1CT(ICRTERM)
       L1=L1CT(ICRTERM)
       I2=I1CT(ICRTERM)
       J2=J1CT(ICRTERM)
       K2=K1CT(ICRTERM)
       L2=L1CT(ICRTERM)

       ! CALL GETICD(I1,J1,K1,L1,.FALSE.,XX,XX,DERIV,XX,XX,X,Y,Z,DX,DY,DZ)
       ! WCRTERM(ICRTERM)=DERIV*DERIV*CIC(ICI(ICRTERM))*PIRAD*IPER**2
       ! mf?
       WCTOT=WCTOT+WCRTERM(ICRTERM)
    ENDDO
#endif 

    !
    ! COMPUTE TOTALS AND REWEIGHT
    !
    WTOT=WBTOT+WTTOT+WPTOT+WITOT
#if KEY_CMAP==1
    WTOT=WTOT+WCTOT
#endif 
    FACTOR=100.0/WTOT
    !
    WBTOL=0.0
    WTTOL=0.0
    WPTOL=0.0
    WITOL=0.0

    ! PRINT OUT RESULTS GREATER THAN TOL.
    IF(PRNLEV.LT.2) RETURN

    WRITE(OUTU,87) TOL
87  FORMAT(' THE FOLLOWING ICS HAD A CONTRIBUTION GREATER THAN', &
         ' TOL=',F12.6)

    WRITE(OUTU,100)
100 FORMAT(/' CONTRIBUTIONS FROM THE BONDS'/)
    DO IBOND=1,NBOND
       IF(WBOND(IBOND).GE.TOL) THEN
          WBTOL=WBTOL+WBOND(IBOND)
          WRITE(OUTU,200) IBOND,ATYPE(IB(IBOND))(1:idleng), &
               ATYPE(JB(IBOND))(1:idleng), &
               WBOND(IBOND),FACTOR*WBOND(IBOND)
200       FORMAT(1X,I5,4X,A,'-',A,T32,F12.5,' KCAL',F8.2,' %')
       ENDIF
    ENDDO
    WRITE(OUTU,205) WBTOL,WBTOL*FACTOR
205 FORMAT(' TOTAL BOND WITHIN TOL',T32,F12.5,' KCAL',F8.2,' %')
    WRITE(OUTU,206) WBTOT,WBTOT*FACTOR
206 FORMAT(' TOTAL BOND CONTRIBUTION',T32,F12.5,' KCAL',F8.2,' %')

    WRITE(OUTU,300)
300 FORMAT(/' CONTRIBUTIONS FROM THE ANGLES'/)
    DO ITHETA=1,NTHETA
       IF(WTHETA(ITHETA).GE.TOL) THEN
          WTTOL=WTTOL+WTHETA(ITHETA)
          WRITE(OUTU,400) ITHETA,ATYPE(IT(ITHETA))(1:idleng), &
               ATYPE(JT(ITHETA))(1:idleng),ATYPE(KT(ITHETA))(1:idleng), &
               WTHETA(ITHETA),FACTOR*WTHETA(ITHETA)
400       FORMAT(I6,4X,A,'-',A,'-',A,T32,F12.5,' KCAL',F8.2,' %')
       ENDIF
    ENDDO
    WRITE(OUTU,405) WTTOL,WTTOL*FACTOR
405 FORMAT(' TOTAL ANGLE WITHIN TOL',T32,F12.5,' KCAL',F8.2,' %')
    WRITE(OUTU,406) WTTOT,WTTOT*FACTOR
406 FORMAT(' TOTAL ANGLE CONTRIBUTION',T32,F12.5,' KCAL',F8.2,' %')

    WRITE(OUTU,500)
500 FORMAT(/' CONTRIBUTIONS FROM THE DIHEDRALS'/)
    DO IPHI=1,NPHI
       IF(WPHI(IPHI).GE.TOL) THEN
          WPTOL=WPTOL+WPHI(IPHI)
          if (idleng<=4) &
          WRITE(OUTU,600) IPHI,ATYPE(IP(IPHI))(1:idleng), &
               ATYPE(JP(IPHI))(1:idleng),ATYPE(KP(IPHI))(1:idleng), &
               ATYPE(LP(IPHI))(1:idleng),WPHI(IPHI),FACTOR*WPHI(IPHI)
          if (idleng>4) &
          WRITE(OUTU,601) IPHI,ATYPE(IP(IPHI))(1:idleng), &
               ATYPE(JP(IPHI))(1:idleng),ATYPE(KP(IPHI))(1:idleng), &
               ATYPE(LP(IPHI))(1:idleng),WPHI(IPHI),FACTOR*WPHI(IPHI)
600       FORMAT(1X,I5,4X,A,'-',A,'-',A,'-',A,T32,F12.5, &
               ' KCAL',F8.2,' %')
601       FORMAT(1X,I5,4X,A,'-',A,'-',A,'-',A,T48,F12.5, &
               ' KCAL',F8.2,' %')
       ENDIF
    ENDDO
    WRITE(OUTU,605) WPTOL,WPTOL*FACTOR
605 FORMAT(' TOTAL DIHEDRAL WITHIN TOL',T32,F12.5,' KCAL',F8.2,' %')
    WRITE(OUTU,606) WPTOT,WPTOT*FACTOR
606 FORMAT(' TOTAL DIHEDRAL CONTRIBUTION',T32,F12.5,' KCAL',F8.2,' %')

    WRITE(OUTU,700)
700 FORMAT(/' CONTRIBUTIONS FROM THE IMPROPERS'/)
    DO IIMPHI=1,NIMPHI
       IF(WIMPHI(IIMPHI).GE.TOL) THEN
          WITOL=WITOL+WIMPHI(IIMPHI)
          if (idleng<=4) &
          WRITE(OUTU,600) IIMPHI,ATYPE(IM(IIMPHI))(1:idleng), &
               ATYPE(JM(IIMPHI))(1:idleng),ATYPE(KM(IIMPHI))(1:idleng), &
               ATYPE(LM(IIMPHI))(1:idleng), &
               WIMPHI(IIMPHI),FACTOR*WIMPHI(IIMPHI)
          if (idleng>4) &
          WRITE(OUTU,601) IIMPHI,ATYPE(IM(IIMPHI))(1:idleng), &
               ATYPE(JM(IIMPHI))(1:idleng),ATYPE(KM(IIMPHI))(1:idleng), &
               ATYPE(LM(IIMPHI))(1:idleng), &
               WIMPHI(IIMPHI),FACTOR*WIMPHI(IIMPHI)
       ENDIF
    ENDDO
    WRITE(OUTU,705) WITOL,WITOL*FACTOR
705 FORMAT(' TOTAL IMPROPER WITHIN TOL',T32,F12.5,' KCAL',F8.2,' %')
    WRITE(OUTU,706) WITOT,WITOT*FACTOR
706 FORMAT(' TOTAL IMPROPER CONTRIBUTION',T32,F12.5,' KCAL',F8.2,' %')

#if KEY_CMAP==1
    WRITE(OUTU,800)
800 FORMAT(/' CONTRIBUTIONS FROM THE CROSSTERMS'/)
    DO ICRTERM=1,NCRTERM
       IF(WCRTERM(ICRTERM).GE.TOL) THEN
          WITOL=WITOL+WCRTERM(ICRTERM)
          WRITE(OUTU,801) ICRTERM,ATYPE(I1CT(ICRTERM))(1:idleng), &
               ATYPE(J1CT(ICRTERM))(1:idleng), &
               ATYPE(K1CT(ICRTERM))(1:idleng), &
               ATYPE(L1CT(ICRTERM))(1:idleng), &
               ATYPE(I2CT(ICRTERM))(1:idleng), &
               ATYPE(J2CT(ICRTERM))(1:idleng), &
               ATYPE(K2CT(ICRTERM))(1:idleng), &
               ATYPE(L2CT(ICRTERM))(1:idleng), &
               WCRTERM(ICRTERM),FACTOR*WCRTERM(ICRTERM)
801       FORMAT(1X,I5,4X,A,'-',A,'-',A,'-',A,/, &
               A,'-',A,'-',A,'-',A,T32,F12.5, &
               ' KCAL',F8.2,' %')

       ENDIF
    ENDDO
    WRITE(OUTU,805) WITOL,WITOL*FACTOR
805 FORMAT(' TOTAL CROSSTERM WITHIN TOL',T32,F12.5, &
         ' KCAL',F8.2,' %')
    WRITE(OUTU,806) WCTOT,WCTOT*FACTOR
806 FORMAT(' TOTAL CROSSTERM CONTRIBUTION',T32,F12.5, &
         ' KCAL',F8.2,' %')
#endif 

    RETURN
  END SUBROUTINE PED

  SUBROUTINE INTEMD(NATIC,IATIC,X,Y,Z,XNEW,YNEW,ZNEW, &
       NAT3,DDSCR,DDM,LNOMA,OK)
    !
    ! THIS ROUTINE SETS UP A LOCAL INTERNAL MODE FOR A BOND
    ! ANGLE OR DIHEDRAL
    !
    ! By Bernard R. Brooks   1982
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use memory

    implicit none

    INTEGER NATIC,IATIC(*)
    real(chm_real) X(*),Y(*),Z(*),XNEW(*),YNEW(*),ZNEW(*)
    INTEGER NAT3
    real(chm_real) DDSCR(*),DDM(*)
    LOGICAL LNOMA,OK

    integer,allocatable,dimension(:) :: ICON,IBON
    INTEGER I

    call chmalloc('vibsub.src','INTEMD','ICON',NATOM,intg=ICON)
    call chmalloc('vibsub.src','INTEMD','IBON',NBOND,intg=IBON)
    CALL INTEM2(NATIC,IATIC,NATOM,X,Y,Z,XNEW,YNEW,ZNEW,NBOND,IB,JB, &
         IBON,ICON,OK)

    IF(OK) THEN
       ! now remove rotation/translation from this mode
       CALL REMVTR(X,Y,Z,XNEW,YNEW,ZNEW,ICON,NAT3,DDSCR, &
            DDM,LNOMA,AMASS)
    ELSE
       ! if there was an error, we return a zero vector
       XNEW(1:NATOM)=0.0
       YNEW(1:NATOM)=0.0
       ZNEW(1:NATOM)=0.0
    ENDIF

    call chmdealloc('vibsub.src','INTEMD','IBON',NBOND,intg=IBON)
    call chmdealloc('vibsub.src','INTEMD','ICON',NATOM,intg=ICON)
    RETURN
  END SUBROUTINE INTEMD

  SUBROUTINE INTEM2(NAT,IAT,NATOM,X,Y,Z,DX,DY,DZ,NBOND,IB,JB, &
       IBON,ICON,OK)
    !
    ! THIS ROUTINE SETS UP A LOCAL INTERNAL MODE FOR A BOND
    ! ANGLE OR DIHEDRAL
    !
    ! By Bernard R. Brooks   1982
    !
  use chm_kinds
  use stream
  use machutil,only:die
    implicit none

    INTEGER NAT,IAT(*),NATOM
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    INTEGER NBOND
    INTEGER IB(*),JB(*)
    INTEGER ICON(*)
    INTEGER IBON(*)
    LOGICAL OK

    real(chm_real) AX,AY,AZ,BX,BY,BZ,RX,RY,RZ
    INTEGER I,J,K,INC,KK,II,CHANGE,IBB,JBB
    INTEGER ICN,JCN
    LOGICAL RETRY
    real(chm_real) TOL
    DATA TOL/1.0D-6/

    OK=.FALSE.
    ! find-connectivity
    DO I=1,NBOND
       IBON(I)=0
       IF(IB(I).LE.0 .OR. JB(I).LE.0) IBON(I)=1
       ! remove bonds connecting specified atoms
       IBB=0
       JBB=0
       DO II=1,NAT
          IF(IB(I).EQ.IAT(II)) IBB=1
          IF(JB(I).EQ.IAT(II)) JBB=1
       ENDDO
       IF(IBB.EQ.1 .AND. JBB.EQ.1) IBON(I)=1
    ENDDO
    ICON(1:NATOM)=0
    ICON(IAT(1))=-1
    ICON(IAT(NAT))=1
    IF(NAT.EQ.3) THEN
       ICON(IAT(2))=1
    ELSE IF(NAT.EQ.4) THEN
       ICON(IAT(2))=-1
       ICON(IAT(3))=1
    ENDIF

    INC=1
    KK=0
    CHANGE=2
    RETRY=.TRUE.
    DO WHILE (CHANGE.GT.0 .AND. RETRY)
       CHANGE=CHANGE-1
       RETRY=.FALSE.
       iiloop: DO II=1,NBOND
          KK=KK+INC
          IF(IBON(KK).NE.0) CYCLE iiloop

          I=IB(KK)
          J=JB(KK)
          ICN=ICON(I)
          JCN=ICON(J)
          IF(ICN.EQ.0 .AND. JCN.EQ.0) THEN
             RETRY=.TRUE.
             CYCLE iiloop
          ENDIF
          IF(ICN*JCN.GT.0) THEN
             IBON(KK)=1
             CYCLE iiloop
          ENDIF
          IF(ICN*JCN.LT.0) THEN
             ! check for possible loop conflict
             ! ok, now we have a loop conflict...
             IBON(KK)=1
             IF(WRNLEV.GE.2) WRITE(OUTU,33) I,J
33           FORMAT(' Loop found which makes this computation ', &
                  'impossible.',2I5)
             RETURN
          ENDIF

          ! ok, find which is zero and increment connectivity pointer
          K=1
          IF(ICN+JCN.LT.0) K=-1
          IF(ICN.EQ.0) THEN
             ICON(I)=JCN+K
          ELSE
             ICON(J)=ICN+K
          ENDIF
          IBON(KK)=1
          CHANGE=2

       ENDDO iiloop
       KK=KK+INC
       INC=-INC
    ENDDO

    IF(RETRY .AND. PRNLEV.GE.2) WRITE(OUTU,36)
36  FORMAT(' IMTEMD: Note, some atoms in the PSF are not connected', &
         ' to the specified atoms')

    ! find-axis-of-rotation
    IF(NAT.EQ.2) THEN
       ! bond
       RX=X(IAT(2))-X(IAT(1))
       RY=Y(IAT(2))-Y(IAT(1))
       RZ=Z(IAT(2))-Z(IAT(1))
    ELSE IF(NAT.EQ.3) THEN
       ! angle
       AX=X(IAT(2))-X(IAT(1))
       AY=Y(IAT(2))-Y(IAT(1))
       AZ=Z(IAT(2))-Z(IAT(1))
       BX=X(IAT(2))-X(IAT(3))
       BY=Y(IAT(2))-Y(IAT(3))
       BZ=Z(IAT(2))-Z(IAT(3))
       RX=AY*BZ-AZ*BY
       RY=AZ*BX-AX*BZ
       RZ=AX*BY-AY*BX
       BX=X(IAT(2))
       BY=Y(IAT(2))
       BZ=Z(IAT(2))
    ELSE IF(NAT.EQ.4) THEN
       ! dihedral
       RX=X(IAT(3))-X(IAT(2))
       RY=Y(IAT(3))-Y(IAT(2))
       RZ=Z(IAT(3))-Z(IAT(2))
       BX=X(IAT(3))
       BY=Y(IAT(3))
       BZ=Z(IAT(3))
    ELSE
       CALL DIE
    ENDIF
    AX=RX*RX+RY*RY+RZ*RZ
    IF (AX.LT.TOL) THEN
       CALL WRNDIE(-1,'<INTEMD>','Zero norm vector specified')
       RETURN
    ENDIF
    AX=SQRT(AX)
    RX=RX/AX
    RY=RY/AX
    RZ=RZ/AX
    ! generate-new-mode
    DO I=1,NATOM
       IF(ICON(I).LE.0) THEN
          DX(I)=0.0
          DY(I)=0.0
          DZ(I)=0.0
       ELSE
          IF(NAT.EQ.2) THEN
             ! translation...
             DX(I)=RX
             DY(I)=RY
             DZ(I)=RZ
          ELSE
             ! rotation...
             AX=X(I)-BX
             AY=Y(I)-BY
             AZ=Z(I)-BZ
             DX(I)=AY*RZ-AZ*RY
             DY(I)=AZ*RX-AX*RZ
             DZ(I)=AX*RY-AY*RX
          ENDIF
       ENDIF
    ENDDO
    OK=.TRUE.
    RETURN
  END SUBROUTINE INTEM2


  SUBROUTINE CONSINTEMD(NATIC,IATIC,X,Y,Z,XNEW,YNEW,ZNEW, &
       NAT3,DDSCR,DDM,NNMDS,NFREQ,DDEV,NFJAC,FJAC,ANGA,OK)
    !
    ! THIS ROUTINE SETS UP A LOCAL INTERNAL MODE FOR A BOND
    ! OR ANGLE - ONLY THE LAST ATOM IS FREE TO MOVE - THE 
    ! REST IS ASSUMED TO BE CONSTRAINED
    ! (NOTE: THIS BASIS WILL CHANGE THE CENTER OF MASS)
    ! By Gerhard Koenig 2014 based on INTEMD by Bernard R. Brooks   1982
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use memory
  use vector
  use code 
  use param

    implicit none

    INTEGER NATIC            ! Number of atoms in atom selection
    INTEGER NAT3
    INTEGER NFJAC ! Number of entries in FJAC  (OPTI command)
    INTEGER IATIC(NATIC) ! Array with selected atom numbers
    real(chm_real) X(NAT3/3)      ! X coordinates
    real(chm_real) Y(NAT3/3)      ! Y coordinates
    real(chm_real) Z(NAT3/3)      ! Z coordinates
    real(chm_real) XNEW(NAT3/3)  ! Array for the x-components of the new basis 
    real(chm_real) YNEW(NAT3/3)  ! Array for the x-components of the new basis 
    real(chm_real) ZNEW(NAT3/3)  ! Array for the x-components of the new basis 
    real(chm_real) DDSCR(NAT3) ! scratch
    real(chm_real) DDM(NAT3/3)    ! masses  
    INTEGER NNMDS                    ! Number of max normal modes
    INTEGER NFREQ                     ! Actual number of normal modes so far
    real(chm_real) DDEV(NNMDS) ! Eigenvalues
    real(chm_real) FJAC(3,NNMDS) ! Array for Jacobian factors in OPTI plus eigenvalue
    INTEGER ANGA(3,NNMDS) ! Array for indices of atoms in angle CANG

    INTEGER I ! Local variable for loop

    LOGICAL OK        ! Logical Did everything work?
    

    OK = .FALSE. 

    ! Make sure that DDSCR does not contain anything nasty
    DDSCR(1:NAT3) = 0.0E+0

    ! Treat bonds
    IF (NATIC .EQ. 2) THEN 

       ! Calculate bond vector between selected atoms
       DDSCR(1) =   X(IATIC(2))-X(IATIC(1))
       DDSCR(2) =   Y(IATIC(2))-Y(IATIC(1))
       DDSCR(3) =   Z(IATIC(2))-Z(IATIC(1))

       ! Normalize vector and save the Jacobian factor (bond length)^2 
       NFJAC=NFJAC+1 
       FJAC(1,NFJAC) = SUM(DDSCR(1:3)**2)
       DDSCR(1:3) = DDSCR(1:3) / SQRT(FJAC(1,NFJAC))
 
       ! Make sure that this is actually a proper bond 
       IF (FJAC(1,NFJAC).LT.1.0E-8) CALL WRNDIE(-1,'<CONSINTEMD>', &
       'BOND IS NOT WELL DEFINED (TOO SHORT < 0.0001 ANGSTROM )')
       IF (FJAC(1,NFJAC).GT.9999.98) CALL WRNDIE(-1,'<CONSINTEMD>', &
       'BOND IS NOT WELL DEFINED (TOO LONG > 99.9999 ANGSTROM)')      

       ! Store vector in last selected atom
       XNEW(IATIC(2)) = DDSCR(1)
       YNEW(IATIC(2)) = DDSCR(2)
       ZNEW(IATIC(2)) = DDSCR(3)

       ! Save the atom indices of the bond for Jacobian calculation in OPTI      
       ANGA(1,NFJAC) = IATIC(1)
       ANGA(2,NFJAC) = IATIC(2)
       ANGA(3,NFJAC) = 0

       ! Obtain force constant from parameter file and use it as an initial guess for the eigenvalue
       ! Here, we search for the bond in the psf
       DO I = 1,NBOND
          IF(IB(I) .EQ.  IATIC(1)  .OR. JB(I) .EQ.  IATIC(1)) THEN
             IF(IB(I) .EQ.  IATIC(2)  .OR. JB(I) .EQ.  IATIC(2)) THEN
                 ! Save eigenvalue (force constant/mass) to FJAC for PARA option in OPTI              
                 FJAC(3,NFJAC) = 2.0E+0*CBC(ICB(IB(I)))*DDM(IATIC(2))*DDM(IATIC(2))
                 OK=.TRUE.
                 EXIT
             ENDIF   
          ENDIF
       ENDDO

       ! If bond was not found in PSF, die
       IF(.NOT. OK) THEN
          CALL WRNDIE(0,'<CONSINTEMD>','BOND DOES NOT EXIST IN PSF')
       ENDIF
      
       OK=.TRUE.

    ! Treat bond angles
    ELSEIF (NATIC .EQ. 3) THEN  

       ! Calculate normal vector for plane defined by the three atoms
       ! Vector for first bond between atom 2 and 1  
       DDSCR(1) =   X(IATIC(2))-X(IATIC(1))
       DDSCR(2) =   Y(IATIC(2))-Y(IATIC(1))
       DDSCR(3) =   Z(IATIC(2))-Z(IATIC(1))
       ! Vector for second bond between atom 2 and 3
       DDSCR(4) =   X(IATIC(2))-X(IATIC(3))
       DDSCR(5) =   Y(IATIC(2))-Y(IATIC(3))
       DDSCR(6) =   Z(IATIC(2))-Z(IATIC(3))

       ! Use cross product to calculate vector representing plane 
       ! The Jacobian factor for an unbranched chain = sin(\theta_123) 
       ! We can obtain this from the cross product |a x b| = |a| |b| sin(\theta)
       CALL CROSS3(DDSCR(1:3),DDSCR(4:6),DDSCR(7:9))

       ! The Jacobian factor = sin(\theta)         
       FJAC(1,NFJAC+1) = SQRT(SUM(DDSCR(7:9)**2) / SUM(DDSCR(1:3)**2) / SUM(DDSCR(4:6)**2))

       ! Make sure that the angle is not linear      
       IF (FJAC(1,NFJAC+1).LT.0.001) CALL WRNDIE(-1,'<CONSINTEMD>', &
       'ANGLE IS ALMOST LINEAR (< 0.001 RAD)')
       NFJAC=NFJAC+1    

       ! Save the atom indices of the angle for Jacobian calculation in OPTI      
       ANGA(1,NFJAC) = IATIC(1)
       ANGA(2,NFJAC) = IATIC(2)
       ANGA(3,NFJAC) = IATIC(3) 

       ! Save bond length for later use
       FJAC(2,NFJAC) = SQRT(SUM(DDSCR(4:6)**2))
      

       ! The tangential vector can be obtained from the crossproduct 
       ! between the bond_23 and the normal vector of the plane of 123
       CALL CROSS3(DDSCR(4:6),DDSCR(7:9),DDSCR(1:3))

       ! Normalize vector
       CALL NORMALL(DDSCR(7:9),3) 

       ! Store vector in last selected atom
       XNEW(IATIC(3)) = DDSCR(1)
       YNEW(IATIC(3)) = DDSCR(2)
       ZNEW(IATIC(3)) = DDSCR(3) 

       ! Obtain force constant from parameter file and use it as an initial guess for the eigenvalue
       ! Here, we search for the angle in the psf
       DO I = 1,NTHETA
          IF(IT(I) .EQ.  IATIC(1)  .OR. KT(I) .EQ.  IATIC(1)) THEN
             IF(IT(I) .EQ.  IATIC(3)  .OR. KT(I) .EQ.  IATIC(3)) THEN
                 IF(JT(I) .EQ.  IATIC(2)) THEN
                     ! Save eigenvalue (force constant/mass) to FJAC for PARA option in OPTI              
                     FJAC(3,NFJAC) = 2.0E+0*CTC(ICT(IT(I)))*DDM(IATIC(3))*DDM(IATIC(3))
                     ! (the eigenvalue is a force constant along the tangential vector
                     ! i.e,  a displacement in Ang. Due to the converstion from angle
                     ! to displacement force constant, it becomes dependent on 
                     ! the bond length between atoms J and K.... which is stored in FJAC:
                     FJAC(3,NFJAC) =FJAC(3,NFJAC)/(FJAC(2,NFJAC)*FJAC(2,NFJAC))
                     OK=.TRUE.
                     EXIT
                 ENDIF
             ENDIF   
          ENDIF
       ENDDO

       ! If angle was not found in PSF, die
       IF(.NOT. OK) THEN
          CALL WRNDIE(0,'<CONSINTEMD>','ANGLE DOES NOT EXIST IN PSF')
       ENDIF

       ! If everything worked, increase number of Jacobian factors 
       OK=.TRUE.

    ENDIF  

    IF(.NOT.OK) THEN
       ! if there was an error, we return a zero vector
       XNEW(1:NATOM)=0.0
       YNEW(1:NATOM)=0.0
       ZNEW(1:NATOM)=0.0
       NFJAC = NFJAC - 1 
    ENDIF

    ! Make sure that DDSCR does not contain anything nasty
    DDSCR(1:NAT3) = 0.0E+0

    RETURN
  END SUBROUTINE CONSINTEMD


  SUBROUTINE REMVTR(X,Y,Z,DX,DY,DZ,ICON,NAT3,DDSCR,DDM, &
       LNOMA,AMASS)
    !
    ! DX COMES IN AS COORD DISPACEMENTS, AND GOES OUT A NORMALIZED
    ! INTERNAL SUBMODE
    ! Note: this routine is inefficient
    !
    ! By Bernard R. Brooks   1982
    !
    use chm_kinds
    use memory
    use vector
    implicit none

    INTEGER NAT3,NTROT
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) AMASS(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    INTEGER ICON(*)
    LOGICAL LNOMA

    real(chm_real) DDSCR(NAT3),DDM(*)

    real(chm_real),dimension(NAT3,6) :: DDTROT
    INTEGER NATOM,I,J,IPT

    NATOM=NAT3/3

    CALL GETTR(X,Y,Z,ICON,NAT3,DDSCR,DDM,NTROT,DDTROT, &
         LNOMA,AMASS)
    !
    ! remove trans-rot modes from internal motion
    !
    IPT=-2
    DO J=1,NATOM
       IPT=IPT+3
       IF(ICON(J).NE.0) THEN
          DDSCR(IPT)=DX(J)/DDM(J)
          DDSCR(IPT+1)=DY(J)/DDM(J)
          DDSCR(IPT+2)=DZ(J)/DDM(J)
       ELSE
          DDSCR(IPT)=0.0
          DDSCR(IPT+1)=0.0
          DDSCR(IPT+2)=0.0
       ENDIF
    ENDDO

    DO I=1,NTROT
       CALL ORTHOG(DDSCR,DDTROT(1,I),NAT3)
    ENDDO
    CALL NORMALL(DDSCR,NAT3)

    IPT=-2
    DO J=1,NATOM
       IPT=IPT+3
       IF(ICON(J).NE.0) THEN
          DX(J)=DDSCR(IPT)*DDM(J)
          DY(J)=DDSCR(IPT+1)*DDM(J)
          DZ(J)=DDSCR(IPT+2)*DDM(J)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE REMVTR

  SUBROUTINE GETTR(X,Y,Z,ICON,NAT3,DDSCR,DDM,NTROT,DDTROT, &
       LNOMA,AMASS)
    !
    ! This routine generates translation/rotation modes for
    ! selected atoms.
    !
    ! By Bernard R. Brooks   1984
    !
    use chm_kinds
    use vector
    implicit none
    INTEGER NAT3,NTROT
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) AMASS(*)
    INTEGER ICON(*)
    LOGICAL LNOMA

    real(chm_real) DDSCR(NAT3),DDM(*),DDTROT(NAT3,6)
    INTEGER NATOM,I,K,IPT
    real(chm_real) XCM,YCM,ZCM,AMASST,AM,Q,TOL

    DATA TOL/1.0D-10/

    NATOM=NAT3/3
    !
    ! COMPUTE CENTER OF MASS
    !
    XCM=0.0
    YCM=0.0
    ZCM=0.0
    AMASST=0.0
    DO I=1,NATOM
       IF(ICON(I).NE.0) THEN
          AM=AMASS(I)
          IF(LNOMA) AM=1.0
          XCM=XCM+X(I)*AM
          YCM=YCM+Y(I)*AM
          ZCM=ZCM+Z(I)*AM
          AMASST=AMASST+AM
       ENDIF
    ENDDO
    XCM=XCM/AMASST
    YCM=YCM/AMASST
    ZCM=ZCM/AMASST
    !
    ! COMPUTE TRANS-ROT MODES FOR THIS SUBSPACE
    !
    NTROT=0
    DO K=1,6
       IPT=-2
       DO I=1,NATOM
          IPT=IPT+3
          DDSCR(IPT)=0.0
          DDSCR(IPT+1)=0.0
          DDSCR(IPT+2)=0.0
          IF(ICON(I).NE.0) THEN
             IF(K.EQ.1) THEN
                DDSCR(IPT)=1.0
             ELSE IF(K.EQ.2) THEN
                DDSCR(IPT+1)=1.0
             ELSE IF(K.EQ.3) THEN
                DDSCR(IPT+2)=1.0
             ELSE IF(K.EQ.4) THEN
                DDSCR(IPT+2)=(Y(I)-YCM)
                DDSCR(IPT+1)=-(Z(I)-ZCM)
             ELSE IF(K.EQ.5) THEN
                DDSCR(IPT)=(Z(I)-ZCM)
                DDSCR(IPT+2)=-(X(I)-XCM)
             ELSE
                DDSCR(IPT+1)=(X(I)-XCM)
                DDSCR(IPT)=-(Y(I)-YCM)
             ENDIF
             DDSCR(IPT)=DDSCR(IPT)/DDM(I)
             DDSCR(IPT+1)=DDSCR(IPT+1)/DDM(I)
             DDSCR(IPT+2)=DDSCR(IPT+2)/DDM(I)
          ENDIF
       ENDDO
       DO I=1,NTROT
          CALL ORTHOG(DDSCR,DDTROT(1,I),NAT3)
       ENDDO
       CALL DOTPR(DDSCR,DDSCR,NAT3,Q)
       IF(Q.GT.TOL) THEN
          NTROT=NTROT+1
          CALL NORMALL(DDSCR,NAT3)
          ddtrot(1:nat3,ntrot) = DDSCR(1:NAT3)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE GETTR

end module vibsub

