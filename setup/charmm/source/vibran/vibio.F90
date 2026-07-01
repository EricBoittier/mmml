module vibio
  implicit none

contains

  SUBROUTINE PRTNMD(ISTRT,ISTOP,IUNIT,NAT3,ISLCT, &
       DDV,DDM,DDF,DDEV,DDSCR, &
       LNOMA,LNONO,ITYPE,RTYPE,TFREQ, &
       LINTD,LFINIT,LSTAT,LVECT,LDOTP,X,Y,Z,XNEW,YNEW,ZNEW, &
       XNORM,YNORM,ZNORM,CG,LDIPO, &
       AMASS,IBASE,ATYPE,RES,NRES &
       )
    !
    ! PRINTS INFORMATION ABOUT NORMAL MODES
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use intcor_module
  use intcor2,only:writic,diffic,intder
  use dimens_fcm
  use consta    ! Polarizability (G. Lamoureux)
  use deriv
  use bases_fcm
  use memory
  use stream
  use chutil,only: getres
  use vector

    implicit none

    INTEGER ISTRT,ISTOP,IUNIT,NAT3,ITYPE,NRES
    INTEGER IBASE(*),ISLCT(*)
    character(len=*) ATYPE(*),RES(*)
    real(chm_real) AMASS(*),CG(*)

    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)

    LOGICAL LNOMA,LNONO,LINTD,LFINIT,LVECT,LDOTP,LTFREQ,LDIPO,LSTAT
    INTEGER NATOM,I,J,K,IPT,KM,IIRES
    INTEGER J2
    real(chm_real) RTYPE,TFREQ

    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) X(*),Y(*),Z(*),XNEW(*),YNEW(*),ZNEW(*)

    real(chm_real),allocatable,dimension(:,:) :: DDTROT
    real(chm_real) DE,DRT,PDX,XCM,YCM,ZCM
    real(chm_real) CGT,AMASST,UX,UY,UZ,UR,UCG
    real(chm_real) DELTA
    real(chm_real) AFACT

    real(chm_real) ALXX,ALXY,ALXZ,ALYY,ALYZ,ALZZ         ! Polarizability
    real(chm_real) SALXX,SALXY,SALXZ,SALYY,SALYZ,SALZZ   ! Polarizability

    CHARACTER(len=4) :: STYPE(5),SNONO,SNOMA,SNO,SBLANK
    DATA STYPE/'TEMP','KCAL','RMS ','FACT','MRMS'/
    DATA SNO/'NO  '/,SBLANK/'    '/

    NATOM=NAT3/3

    SNOMA=SBLANK
    IF(LNOMA) SNOMA=SNO
    SNONO=SBLANK
    IF(LNONO) SNONO=SNO

    WRITE(IUNIT,625)
625 FORMAT(/15X,'NORMAL MODES'/)
    WRITE(IUNIT,628) (I,DDF(I),I=ISTRT,ISTOP)
628 FORMAT(5(I6,F10.2))

    XCM=0.0
    YCM=0.0
    ZCM=0.0
    AMASST=0.0
    CGT=0.0
    IF(LNOMA) THEN
       DO J=1,NATOM
          XCM=XCM+X(J)
          YCM=YCM+Y(J)
          ZCM=ZCM+Z(J)
          AMASST=AMASST+1.0
          CGT=CGT+CG(J)
       ENDDO
    ELSE
       DO J=1,NATOM
          XCM=XCM+X(J)*AMASS(J)
          YCM=YCM+Y(J)*AMASS(J)
          ZCM=ZCM+Z(J)*AMASS(J)
          AMASST=AMASST+AMASS(J)
          CGT=CGT+CG(J)
       ENDDO
    ENDIF
    XCM=XCM/AMASST
    YCM=YCM/AMASST
    ZCM=ZCM/AMASST

    call chmalloc('vibio.src','PRTNMD','DDTROT',NAT3,6,crl=DDTROT)
    DO K=1,6
       IPT=1
       DO J=1,NATOM
          DDSCR(IPT)=0.0
          DDSCR(IPT+1)=0.0
          DDSCR(IPT+2)=0.0
          IF(K.EQ.1) THEN
             DDSCR(IPT)=1.0
          ELSE IF(K.EQ.2) THEN
             DDSCR(IPT+1)=1.0
          ELSE IF(K.EQ.3) THEN
             DDSCR(IPT+2)=1.0
          ELSE IF(K.EQ.4) THEN
             DDSCR(IPT+2)=(Y(J)-YCM)
             DDSCR(IPT+1)=-(Z(J)-ZCM)
          ELSE IF(K.EQ.5) THEN
             DDSCR(IPT)=(Z(J)-ZCM)
             DDSCR(IPT+2)=-(X(J)-XCM)
          ELSE
             DDSCR(IPT+1)=(X(J)-XCM)
             DDSCR(IPT)=-(Y(J)-YCM)
          ENDIF
          DDSCR(IPT)=DDSCR(IPT)/DDM(J)
          DDSCR(IPT+1)=DDSCR(IPT+1)/DDM(J)
          DDSCR(IPT+2)=DDSCR(IPT+2)/DDM(J)
          IPT=IPT+3
       ENDDO
       KM=K-1
       DO I=1,KM
          CALL ORTHOG(DDSCR,DDTROT(1,I),NAT3)
       ENDDO
       CALL NORMALL(DDSCR,NAT3)
       ddtrot(1:nat3,k) = DDSCR(1:NAT3)
    ENDDO

    ! BEGIN Polarizability (G. Lamoureux)
    IF(LDIPO) THEN
       SALXX=0.0
       SALXY=0.0
       SALXZ=0.0
       SALYY=0.0
       SALYZ=0.0
       SALZZ=0.0
    ENDIF
    ! END Polarizability (G. Lamoureux)

    DO I=ISTRT,ISTOP

       CALL MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV(1,I),DDM,DDSCR, &
            DDEV(I),.TRUE.,ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT,LTFREQ)

       DE=0.0
       DO J=1,NATOM
          DE=DE+(DX(J)*XNORM(J)+DY(J)*YNORM(J)+DZ(J)*ZNORM(J))
       ENDDO

       DRT=0.0
       DO K=1,6
          CALL DOTPR(DDV(1,I),DDTROT(1,K),NAT3,PDX)
          DRT=DRT+PDX*PDX
       ENDDO
       DRT=DRT*100.0

       IF(LDIPO) THEN
          ! Compute dipole derivatives
          UX=0.0
          UY=0.0
          UZ=0.0
          DO J=1,NATOM
             UCG=CG(J)-CGT/NATOM
             UX=UX+XNORM(J)*UCG
             UY=UY+YNORM(J)*UCG
             UZ=UZ+ZNORM(J)*UCG
          ENDDO
          UR=SQRT(UX*UX+UY*UY+UZ*UZ)
       ENDIF

       WRITE(IUNIT,629) I,DDF(I),DRT,DDEV(I),DE
629    FORMAT(/2X,'VIBRATION MODE',I4,'  FREQUENCY=',F12.6, &
            '  TRANS ROT %=',F12.6/, &
            '  EIGENVALUE=',F12.6,'  ENERGY DERIVATIVE=',F12.6)
       IF(LDIPO) WRITE(IUNIT,634) UX,UY,UZ,UR
634    FORMAT('  DIPOLE DERIVATIVES',3F12.6,' TOTAL',F12.6)
       IF(LTFREQ) WRITE(IUNIT,624) TFREQ
624    FORMAT('  THIS MODE IS BELOW TFREQ (',F9.2,')')
       WRITE(IUNIT,623) STYPE(ITYPE),RTYPE,SNOMA,SNONO,AFACT
623    FORMAT('  TYPE  ''',A4,'''',F10.4,1X,A2,'MASS', &
            1X,A2,'NORM  STEP=',F12.6/)

631    FORMAT(10X,2I5,1X,A,1X,A,3F10.5)
633    FORMAT('   EIGENVECTOR:')

       ! BEGIN Polarizability (G. Lamoureux)
       IF(LDIPO) THEN
          ! Compute and print polarizabilities
          ! [UX,UY,UZ] = D/A
          ! [DDEV] = kcal/mol/A
          ! [ALXX,...] = A^3
          ! IF(DDEV(I).GT.0.00001) THEN
          IF(DRT.LT.0.001) THEN
             ALXX=CCELEC*UX*UX/DDEV(I)
             ALXY=CCELEC*UX*UY/DDEV(I)
             ALXZ=CCELEC*UX*UZ/DDEV(I)
             ALYY=CCELEC*UY*UY/DDEV(I)
             ALYZ=CCELEC*UY*UZ/DDEV(I)
             ALZZ=CCELEC*UZ*UZ/DDEV(I)
             WRITE(IUNIT,'(a)') ' POLARIZABILITY (IN ANGST**3):'
             WRITE(IUNIT,'(2X,3F12.6)') ALXX,ALXY,ALXZ
             WRITE(IUNIT,'(2X,3F12.6)') ALXY,ALYY,ALYZ
             WRITE(IUNIT,'(2X,3F12.6)') ALXZ,ALYZ,ALZZ
             SALXX=SALXX+ALXX
             SALXY=SALXY+ALXY
             SALXZ=SALXZ+ALXZ
             SALYY=SALYY+ALYY
             SALYZ=SALYZ+ALYZ
             SALZZ=SALZZ+ALZZ
          ELSE
             WRITE(IUNIT,'(a)') ' UNDEFINED POLARIZABILITY'
             WRITE(IUNIT,'(a)') ' (TRANSLATION-ROTATION MODE)'
          ENDIF
       ENDIF
       ! END Polarizability (G. Lamoureux)

       IF(LVECT) THEN
          WRITE(IUNIT,633)
          DO J=1,NATOM
                IF(ISLCT(J).GT.0) THEN
                   J2=J
                   IIRES=GETRES(J2,IBASE,NRES)
                   WRITE(IUNIT,631) J,IIRES,RES(IIRES)(1:idleng), &
                        ATYPE(J)(1:idleng),XNORM(J),YNORM(J),ZNORM(J)
                ENDIF
          ENDDO
       ENDIF

       IF(LINTD) THEN
          ! NOW TRANSFORM TO INTERNAL COORDINATES AND PRINT THEM
          DELTA=1.0
          IF(LFINIT) THEN
             DO J=1,NATOM
                XNEW(J)=X(J)+XNORM(J)
                YNEW(J)=Y(J)+YNORM(J)
                ZNEW(J)=Z(J)+ZNORM(J)
             ENDDO

             CALL DIFFIC(DELTA,XNEW,YNEW,ZNEW,X,Y,Z,&
                  icr_struct%lenic,.FALSE., &
                  icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR)
          ELSE
             CALL INTDER(DELTA,XNORM,YNORM,ZNORM,X,Y,Z,&
                  icr_struct%lenic,.FALSE., &
                  icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR)
          ENDIF

          CALL WRITIC(1,icr_struct%lenic,-1,0,OUTU, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       ENDIF

       IF(LDOTP) THEN
          WRITE(IUNIT,639)
639       FORMAT('  DOT PRODUCTS:')
          DO J=ISTRT,ISTOP
             CALL DOTPR(DDV(1,I),DDV(1,J),NAT3,DDSCR(J))
          ENDDO
          WRITE(IUNIT,638) (J,DDSCR(J),J=ISTRT,ISTOP)
638       FORMAT(5(I5,F11.5))
       ENDIF

       IF(LSTAT) THEN
          DDSCR(1:4)=0.0
          DO K=1,NAT3
             DO J=1,4
                DDSCR(J)=DDSCR(J)+ABS(DDV(K,I)**J)
             ENDDO
          ENDDO
          WRITE(IUNIT,643) (DDSCR(J),J=1,4)
643       FORMAT(' STATISTICS FOR EIGENVECTOR:',/, &
               ' <C>=',F10.4,' <C**2>=',F10.4, &
               ' <C**3>=',F10.4,' <C**4>=',F10.4)
       ENDIF

    ENDDO
    call chmdealloc('vibio.src','PRTNMD','DDTROT',NAT3,6,crl=DDTROT)

    ! BEGIN Polarizability (G. Lamoureux)
    IF(LDIPO) THEN
       WRITE(IUNIT,'(a)') ' TOTAL POLARIZABILITY (IN ANGST**3):'
       WRITE(IUNIT,'(2X,3F12.6)') SALXX,SALXY,SALXZ
       WRITE(IUNIT,'(2X,3F12.6)') SALXY,SALYY,SALYZ
       WRITE(IUNIT,'(2X,3F12.6)') SALXZ,SALYZ,SALZZ
       WRITE(IUNIT,'(A,F12.6)') ' TRACE / 3 = ',(SALXX+SALYY+SALZZ)/3
    ENDIF
    ! END Polarizability (G. Lamoureux)

    RETURN
  END SUBROUTINE PRTNMD

  SUBROUTINE RDNMDX(LCARD,NFREQ,NNMDS,NAT3,NDIM, &
       DDV,DDSCR,DDF,DDEV,IUNNMD,LAPPE,ISTR,ISTP)
    !
    ! READS IN A NORMAL MODE FILE THAT IS PRESENT
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use ctitla
  use consta
  use stream
#if KEY_DIMB==1
  use dimb
#endif 
  use machutil,only:die
  use parallel,only: psnd4,psnd8
    implicit none
    LOGICAL LCARD
    INTEGER NFREQ,NNMDS,NAT3,NDIM,IUNNMD,ISTR,ISTP
    real(chm_real) DDV(NAT3,*),DDSCR(*),DDF(*),DDEV(*)
    LOGICAL LAPPE
    INTEGER ISTRT

    INTEGER ICNTRL(20)
    INTEGER NATOM,ISTOP,I,J
    CHARACTER(len=4) :: HDRN,HDR
    DATA HDRN/'NMDS'/

    NATOM=NAT3/3
    ISTRT=1
    IF(LAPPE) ISTRT=NFREQ+1

    IF(IOLEV.GT.0) THEN

       IF(LCARD) THEN
          CALL RDTITL(TITLEB,NTITLB,IUNNMD,0)
          READ(IUNNMD,35) ICNTRL
35        FORMAT(20I5)
       ELSE
          READ(IUNNMD) HDR,ICNTRL
          IF(HDR.NE.HDRN) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,45) HDR,HDRN
45           FORMAT(/' **** ERROR **** HEADERS DONT MATCH  ',A4,3X,A4)
             CALL DIE
          ENDIF
          CALL RDTITL(TITLEB,NTITLB,IUNNMD,-1)
       ENDIF

       CALL WRTITL(TITLEB,NTITLB,OUTU,1)

#if KEY_DIMB==1
       ITER=ICNTRL(4)
       IPAR1=ICNTRL(5)
       IPAR2=ICNTRL(6)
#endif 

       IF(NATOM.NE.ICNTRL(3)) THEN
          IF(WRNLEV.GE.2) WRITE(OUTU,49) ICNTRL(3),NATOM
49        FORMAT(/' **** ERROR **** NUMBER OF ATOMS',2I6, &
               ' DO NOT MATCH')
          CALL DIE
       ENDIF

       NDIM=ICNTRL(1)
       IF(ICNTRL(1).GT.ISTP) ICNTRL(1)=ISTP
       IF(ISTP.GT.ICNTRL(1)) ISTP=ICNTRL(1)
       NFREQ=ISTP-ISTR+ISTRT
       IF(NFREQ.GT.NNMDS) THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,55) NFREQ,NNMDS
55        FORMAT(/' ***** WARNING ***** NOT ENOUGH SPACE AVAILABLE', &
               ' TO CONTAIN',I5,' MODES. ONLY',I5,' WILL BE KEPT.'/)
          NFREQ=NNMDS
       ENDIF
       ISTOP=NFREQ

       IF(LCARD) THEN
          READ(IUNNMD,36) (DDSCR(J),J=1,NATOM)
36        FORMAT(6E20.12)
          IF(ISTR.EQ.1) THEN
             READ(IUNNMD,36) (DDEV(J),J=ISTRT,ISTOP)
          ELSE
             READ(IUNNMD,36) (DDSCR(J),J=1,ISTR-1), &
                  (DDEV(J),J=ISTRT,ISTOP)
             DO I=1,ISTR-1
                READ(IUNNMD,36)
             ENDDO
          ENDIF
          DO I=ISTRT,ISTOP
             READ(IUNNMD,36) (DDV(J,I),J=1,NAT3)
          ENDDO
       ELSE
          READ(IUNNMD) (DDSCR(J),J=1,NATOM)
          IF(ISTR.EQ.1) THEN
             READ(IUNNMD) (DDEV(J),J=ISTRT,ISTOP)
          ELSE
             READ(IUNNMD) (DDSCR(J),J=1,ISTR-1), &
                  (DDEV(J),J=ISTRT,ISTOP)
             DO I=1,ISTR-1
                READ(IUNNMD)
             ENDDO
          ENDIF
          DO I=ISTRT,ISTOP
             READ(IUNNMD) (DDV(J,I),J=1,NAT3)
          ENDDO
       ENDIF

    ENDIF

#if KEY_PARALLEL==1
    CALL PSND4(NFREQ,1)
    CALL PSND8(DDEV(ISTRT),NFREQ-ISTRT+1)
    DO I=ISTRT,ISTOP
       CALL PSND8(DDV(1,I),NAT3)
    ENDDO
#endif 

    DO I=ISTRT,ISTOP
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    RETURN
  END SUBROUTINE RDNMDX

  SUBROUTINE RDNMD(LCARD,NFREQ,NNMDS,NAT3,NDIM, &
       DDV,DDSCR,DDF,DDEV,IUNNMD,LAPPE,ISTR,ISTP)
    !
    ! WRAPPER FOR GETTING NORMAL MODE DATA FROM FILE
    ! File reading itself is now done by RDNMD0 -- C H Robert 2008
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use ctitla
  use consta
  use stream
  use parallel,only: psnd4,psnd8
    !...##IF DIMB
    !...  use dimb
    !...##ENDIF
    implicit none
    LOGICAL LCARD
    INTEGER NFREQ,NNMDS,NAT3,NDIM,IUNNMD,ISTR,ISTP
    real(chm_real) DDV(NAT3,*),DDSCR(*),DDF(*),DDEV(*)
    LOGICAL LAPPE
    INTEGER ISTRT

    INTEGER NATOM,ISTOP,I,J

    NATOM=NAT3/3
    ISTRT=1
    IF(LAPPE) ISTRT=NFREQ+1

    IF(IOLEV.GT.0) THEN
       CALL RDNMD0(LCARD,NFREQ,NNMDS,NAT3,NDIM, &
            DDV,DDSCR,DDF,DDEV, &
            IUNNMD,ISTR,ISTP,ISTRT,ISTOP)
    ENDIF

#if KEY_PARALLEL==1
    CALL PSND4(NFREQ,1)
    CALL PSND8(DDEV(ISTRT),NFREQ-ISTRT+1)
    DO I=ISTRT,ISTOP
       CALL PSND8(DDV(1,I),NAT3)
    ENDDO
#endif 

    DO I=ISTRT,ISTOP
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    RETURN
  END SUBROUTINE RDNMD

  SUBROUTINE RDNMD0(LCARD,NFREQ,NNMDS,NAT3,NDIM, &
       DDV,DDSCR,DDF,DDEV,IUNNMD,ISTR,ISTP,ISTRT,ISTOP)
    !
    ! READ NORMAL MODE DATAFILE
    ! Code extracted from RDNMD by CH Robert 2008
    ! Normally called by RDNMD, this subroutine can now
    ! be called by a single node of a parallel run
    ! if desired (e.g., as in VMOD).
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use ctitla
  use consta
  use stream
#if KEY_DIMB==1
  use dimb
#endif 
  use machutil,only:die
    implicit none
    LOGICAL LCARD
    INTEGER NFREQ,NNMDS,NAT3,NDIM,IUNNMD,ISTR,ISTP
    real(chm_real) DDV(NAT3,*),DDSCR(*),DDF(*),DDEV(*)
    INTEGER ISTRT,NMFILE

    INTEGER ICNTRL(20)
    INTEGER NATOM,ISTOP,I,J
    CHARACTER(len=4) :: HDRN,HDR
    DATA HDRN/'NMDS'/

    NATOM=NAT3/3

    IF(LCARD) THEN
       CALL RDTITL(TITLEB,NTITLB,IUNNMD,0)
       READ(IUNNMD,35) ICNTRL
35     FORMAT(20I5)
    ELSE
       READ(IUNNMD) HDR,ICNTRL
       IF(HDR.NE.HDRN) THEN
          IF(WRNLEV.GE.2) WRITE(OUTU,45) HDR,HDRN
45        FORMAT(/' **** ERROR **** HEADERS DONT MATCH  ',A4,3X,A4)
          CALL DIE
       ENDIF
       CALL RDTITL(TITLEB,NTITLB,IUNNMD,-1)
    ENDIF

    CALL WRTITL(TITLEB,NTITLB,OUTU,1)

    NMFILE=ICNTRL(1)
#if KEY_DIMB==1
    ITER=ICNTRL(4)
    IPAR1=ICNTRL(5)
    IPAR2=ICNTRL(6)
#endif 

    IF(NATOM.NE.ICNTRL(3)) THEN
       IF(WRNLEV.GE.2) WRITE(OUTU,49) ICNTRL(3),NATOM
49     FORMAT(/' **** ERROR **** NUMBER OF ATOMS',2I6, &
            ' DO NOT MATCH')
       CALL DIE
    ENDIF

    NDIM=ICNTRL(1)
    IF(ICNTRL(1).GT.ISTP) ICNTRL(1)=ISTP
    IF(ISTP.GT.ICNTRL(1)) ISTP=ICNTRL(1)
    NFREQ=ISTP-ISTR+ISTRT
    IF(NFREQ.GT.NNMDS) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,55) NFREQ,NNMDS
55     FORMAT(/' ***** WARNING ***** NOT ENOUGH SPACE AVAILABLE', &
            ' TO CONTAIN',I5,' MODES. ONLY',I5,' WILL BE KEPT.'/)
       NFREQ=NNMDS
    ENDIF
    ISTOP=NFREQ

    IF(LCARD) THEN
       READ(IUNNMD,36) (DDSCR(J),J=1,NATOM)
36     FORMAT(6E20.12)
       IF(ISTR.EQ.1) THEN
          READ(IUNNMD,36) (DDEV(J),J=ISTRT,ISTOP)
       ELSE
          READ(IUNNMD,36) (DDSCR(J),J=1,ISTR-1), &
               (DDEV(J),J=ISTRT,ISTOP), &
               (DDSCR(J),J=1,NMFILE-ISTP)
          ! Read undesired ddv data
          DO I=1,ISTR-1
             READ(IUNNMD,36) (DDSCR(J),J=1,NAT3)
          ENDDO
       ENDIF
       ! Read desired ddv data
       DO I=ISTRT,ISTOP
          READ(IUNNMD,36) (DDV(J,I),J=1,NAT3)
       ENDDO
    ELSE
       READ(IUNNMD) (DDSCR(J),J=1,NATOM)
       IF(ISTR.EQ.1) THEN
          READ(IUNNMD) (DDEV(J),J=ISTRT,ISTOP)
       ELSE
          READ(IUNNMD) (DDSCR(J),J=1,ISTR-1), &
               (DDEV(J),J=ISTRT,ISTOP)
          DO I=1,ISTR-1
             READ(IUNNMD)
          ENDDO
       ENDIF
       DO I=ISTRT,ISTOP
          READ(IUNNMD) (DDV(J,I),J=1,NAT3)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RDNMD0

  SUBROUTINE WRTNMD(LCARD,ISTRT,ISTOP,NAT3, &
       DDV,DDSCR,DDEV,IUNNMD,AMASS)
    !
    ! WRITES OUT A NORMAL MODE FILE THAT IS PRESENT
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use ctitla
  use stream
#if KEY_DIMB==1
  use dimb
#endif
    implicit none
    LOGICAL LCARD
    INTEGER ISTRT,ISTOP,NAT3,IUNNMD
    real(chm_real) DDV(NAT3,*),DDSCR(*),DDEV(*)
    real(chm_real) AMASS(*)
    INTEGER ICNTRL(20)
    INTEGER NATOM,I,J
    CHARACTER(len=4) :: HDRN
    DATA HDRN/'NMDS'/

    IF(IOLEV.LT.0) RETURN

    NATOM=NAT3/3
    DDSCR(1:NATOM)=AMASS(1:NATOM)
    ICNTRL(1)=ISTOP-ISTRT+1
    ICNTRL(2)=NAT3
    ICNTRL(3)=NATOM
    ICNTRL(4:20)=0
#if KEY_DIMB==1
    ICNTRL(4)=ITER
    ICNTRL(5)=IPAR1
    ICNTRL(6)=IPAR2
#endif 

    IF(LCARD) THEN
       CALL WRTITL(TITLEA,NTITLA,IUNNMD,0)
       WRITE(IUNNMD,35) ICNTRL
35     FORMAT(20I5)
       WRITE(IUNNMD,36) (DDSCR(J),J=1,NATOM)
       WRITE(IUNNMD,36) (DDEV(J),J=ISTRT,ISTOP)
       DO I=ISTRT,ISTOP
          WRITE(IUNNMD,36) (DDV(J,I),J=1,NAT3)
       ENDDO
36     FORMAT(6E20.12)
    ELSE
       WRITE(IUNNMD) HDRN,ICNTRL
       CALL WRTITL(TITLEA,NTITLA,IUNNMD,-1)
       WRITE(IUNNMD) (DDSCR(J),J=1,NATOM)
       WRITE(IUNNMD) (DDEV(J),J=ISTRT,ISTOP)
       DO I=ISTRT,ISTOP
          WRITE(IUNNMD) (DDV(J,I),J=1,NAT3)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE WRTNMD

  SUBROUTINE VIBTRJ(ISTRT,ISTOP,NAT3,DDM,IUNIT,ITYPE,RTYPE, &
       LNOMA,LNONO,LSEQU,X,Y,Z,DDV,DDSCR,XNEW,YNEW,ZNEW, &
       XNORM,YNORM,ZNORM,PHAS,NCYC,STEP,DDEV,DDF,TFREQ, &
       LSHAK,XTEMP,YTEMP,ZTEMP,AMASS,IMOVE,ISKP)
    !
    ! THIS ROUTINE OUTPUTS A SERIES OF TRAJECTORY FILES
    !
    ! By Bernard R. Brooks    1982
    !
  use chm_kinds
  use ctitla
  use consta
  use number
  use cvio
  use stream
  use holonom,only:holonoma
    implicit none
    INTEGER ISTRT,ISTOP,NAT3,IUNIT,ITYPE
    real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDEV(*),DDF(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) RTYPE,TFREQ,AFACT,PHAS,STEP
    LOGICAL LNOMA,LNONO,LTFREQ,LSEQU,LSHAK
    real(chm_real) XTEMP(*),YTEMP(*),ZTEMP(*)
    real(chm_real) AMASS(*)
    INTEGER IMOVE(*),ISKP(*)

    INTEGER NATOM,NSTEP,NSTEPT,NCYC,IUNCRD,ISTEP,ISTEPT,ICYC,I,J
    real(chm_real) PHA,PHASE,PHASE2,FRQ,FACT
    CHARACTER(len=4) :: STYPE(5),SNONO,SNOMA,SNO,SBLANK
    LOGICAL QOK
    DATA STYPE/'TEMP','KCAL','RMS ','FACT','MRMS'/
    DATA SNO/'NO'/,SBLANK/'  '/

    NATOM=NAT3/3
    SNOMA=SBLANK
    IF(LNOMA) SNOMA=SNO
    SNONO=SBLANK
    IF(LNONO) SNONO=SNO
    PHASE=PHAS
    IF(PHASE.LT.0.01) PHASE=30.0
    NSTEP=(360.0/PHASE+0.5)
    PHASE=2.0*PI
    PHASE=PHASE/NSTEP
    PHASE2=360.0
    PHASE2=PHASE2/NSTEP
    IF(LSEQU) THEN
       NSTEPT=NSTEP*NCYC
    ELSE
       NSTEPT=NSTEP*(ISTOP-ISTRT+1)*NCYC
    ENDIF

    IF(PRNLEV.GE.2) THEN
       IF(STEP.EQ.0.0) THEN
          WRITE(OUTU,224) ISTRT,ISTOP,PHASE2
224       FORMAT(/' TRAJECTORIES FOR MODES',I4,' TO',I4, &
               ' USING A PHASE ANGLE OF',F8.2)
       ELSE
          WRITE(OUTU,223) ISTRT,ISTOP,STEP
223       FORMAT(/' TRAJECTORIES FOR MODES',I4,' TO',I4, &
               ' USING A STEP SIZE OF',F10.4,' PICOSECONDS')
       ENDIF
       IF(LSEQU) THEN
          WRITE(OUTU,226) IUNIT
226       FORMAT(' WILL BE WRITTEN SEQUENTIALLY STARTING WITH UNIT',I4)
       ELSE
          WRITE(OUTU,227) IUNIT
227       FORMAT(' WILL ALL BE WRITTEN TO UNIT',I4)
       ENDIF
       IF(STEP.EQ.0.0) THEN
          IF(NCYC.NE.1) WRITE(OUTU,228) NCYC
228       FORMAT(I5,' CYCLES OF EACH MODE WILL BE WRITTEN')
       ELSE
          PHA=NCYC*NSTEP*STEP
          WRITE(OUTU,229) PHA
229       FORMAT(1X,F10.4,' PICOSECONDS OF EACH MODE WILL BE WRITTEN')
       ENDIF
    ENDIF

    IF(LSEQU) THEN
       IUNCRD=IUNIT-1
    ELSE
       IUNCRD=IUNIT
    ENDIF
    ISTEPT=0

    DO I=ISTRT,ISTOP
       CALL MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV(1,I),DDM,DDSCR, &
            DDEV(I),.TRUE.,ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT,LTFREQ)
       IF(PRNLEV.GE.2) WRITE(OUTU,629) I,DDF(I),DDEV(I)
629    FORMAT(/2X,'VIBRATION MODE',I4,'  FREQUENCY=',F12.6, &
            '  EIGENVALUE=',F12.6)
       IF(LTFREQ .AND. PRNLEV.GE.2) WRITE(OUTU,624) TFREQ
624    FORMAT('  THIS MODE IS BELOW TFREQ (',F9.2,')')
       IF(PRNLEV.GE.2) WRITE(OUTU,623) STYPE(ITYPE),RTYPE,SNOMA, &
            SNONO,AFACT
623    FORMAT('  TYPE  ''',A4,'''',F10.4,1X,A2,'MASS', &
            1X,A2,'NORM  STEP=',F12.6)

       IF(LSEQU) THEN
          ISTEPT=0
          IUNCRD=IUNCRD+1
       ENDIF
       IF(STEP.GT.0.0) THEN
          FRQ=ABS(DDF(I))
          IF(FRQ.LT.TFREQ) FRQ=TFREQ
          PHASE=360.0*STEP*(FRQ*SPEEDL)
          IF(PRNLEV.GE.2) WRITE(OUTU,255) PHASE,I
255       FORMAT(' PHASE OF',F12.2,' FOUND FOR MODE',I5)
          PHASE=PHASE*PI/180.0
       ENDIF

       PHA=0.0
       DO ICYC=1,NCYC
          DO ISTEP=1,NSTEP
             ISTEPT=ISTEPT+1
             FACT=SIN(PHA)
             PHA=PHA+PHASE
             DO J=1,NATOM
                XNEW(J)=X(J)+XNORM(J)*FACT
                YNEW(J)=Y(J)+YNORM(J)*FACT
                ZNEW(J)=Z(J)+ZNORM(J)*FACT
             ENDDO

             IF(LSHAK) THEN
                DO J=1,NATOM
                   XTEMP(J)=XNEW(J)
                   YTEMP(J)=YNEW(J)
                   ZTEMP(J)=ZNEW(J)
                ENDDO
                CALL HOLONOMA(XNEW,YNEW,ZNEW,XTEMP,YTEMP,ZTEMP, &
                     .TRUE.,.FALSE.,QOK)
             ENDIF

             CALL WRITCV(XNEW,YNEW,ZNEW, &
#if KEY_CHEQ==1
                  (/ ZERO /), .FALSE.,                  & 
#endif
                  NATOM, (/ 0 /), NATOM,1,ISTEPT, &
                  NAT3,STEP,1,NSTEPT, &
                  TITLEA,NTITLA,IUNCRD,.FALSE., &
                  .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
          ENDDO
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE VIBTRJ

  SUBROUTINE VIBSUP(ISTRT,ISTOP,NAT3,DDM,IUNIT,ITYPE,RTYPE, &
       LNOMA,LNONO,LSEQU,X,Y,Z,DDV,DDSCR,XNEW,YNEW,ZNEW, &
       XNORM,YNORM,ZNORM,PHAS,NCYC,STEP,DDEV,DDF,TFREQ, &
       LSHAK,XTEMP,YTEMP,ZTEMP,AMASS,IMOVE,ISKP,LRAND,ISEED)
    !-----------------------------------------------------------------------
    ! 18-Sep-1996 Herman van Vlijmen
    !
    ! This routine superposes a set of normal modes with
    ! random initial phases and writes a trajectory file
    !-----------------------------------------------------------------------
  use chm_kinds
  use ctitla
  use consta
  use number
  use cvio
  use stream
  use exfunc
  use clcg_mod,only: random
  use holonom,only:holonoma
    implicit none
    ! Passed variables
    INTEGER ISTRT,ISTOP,NAT3,IUNIT,ITYPE
    real(chm_real) DDV(NAT3,*),DDM(*),DDSCR(*),DDEV(*),DDF(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) XNORM(*),YNORM(*),ZNORM(*)
    real(chm_real) RTYPE,TFREQ,AFACT,PHAS,STEP
    LOGICAL LNOMA,LNONO,LTFREQ,LSEQU,LSHAK,LRAND
    real(chm_real) XTEMP(*),YTEMP(*),ZTEMP(*)
    real(chm_real) AMASS(*)
    INTEGER IMOVE(*),ISKP(*),ISEED
    ! Local variables
    INTEGER NATOM,NSTEP,NSTEPT,NCYC,IUNCRD,ISTEP,ISTEPT,ICYC,I,J
    real(chm_real) PHA,PHASE,PHASE2,FRQ,FACT,TFREQQ,PHASER(1000)
    CHARACTER(len=4) :: STYPE(5),SNONO,SNOMA,SNO,SBLANK
    LOGICAL QOK
    DATA STYPE/'TEMP','KCAL','RMS ','FACT','MRMS'/
    DATA SNO/'NO'/,SBLANK/'  '/

    ! Begin
    TFREQQ=TFREQ
    TFREQ=0.5
    NATOM=NAT3/3
    SNOMA=SBLANK
    IF(LNOMA) SNOMA=SNO
    SNONO=SBLANK
    IF(LNONO) SNONO=SNO
    PHASE=PHAS
    IF(PHASE.LT.0.01) PHASE=30.0
    NSTEP=(360.0/PHASE+0.5)
    PHASE=2.0*PI
    PHASE=PHASE/NSTEP
    PHASE2=360.0
    PHASE2=PHASE2/NSTEP
    NSTEPT=NSTEP*NCYC

    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,223) ISTRT,ISTOP,STEP
223    FORMAT(/' SUPERPOSED TRAJECTORIES FOR MODES',I4,' TO',I4, &
            ' USING A STEP SIZE OF',F10.4,' PICOSECONDS')
       IF(LSEQU) THEN
          WRITE(OUTU,226) IUNIT
226       FORMAT(' WILL BE WRITTEN SEQUENTIALLY STARTING WITH UNIT',I4)
       ELSE
          WRITE(OUTU,227) IUNIT
227       FORMAT(' WILL ALL BE WRITTEN TO UNIT',I4)
       ENDIF
       PHA=NCYC*NSTEP*STEP
       IF(LRAND) THEN
          WRITE(OUTU,229) PHA
       ELSE
          WRITE(OUTU,230) PHA
       ENDIF
229    FORMAT(1X,F8.3,' PS OF SUPERPOSED MODES WILL', &
            ' BE WRITTEN WITH RANDOM INITIAL PHASES')
230    FORMAT(1X,F8.3,' PS OF SUPERPOSED MODES WILL', &
            ' BE WRITTEN WITH ZERO INITIAL PHASES')
    ENDIF

    IF(LSEQU) THEN
       IUNCRD=IUNIT-1
    ELSE
       IUNCRD=IUNIT
    ENDIF
    ISTEPT=0

    DO I=ISTRT,ISTOP
       IF(LRAND) THEN
          PHASER(I)=RANDOM(ISEED)*PI
       ELSE
          PHASER(I)=0.0
       ENDIF
    ENDDO
    DO ICYC=1,NCYC
       DO ISTEP=1,NSTEP
          ISTEPT=ISTEPT+1
          XNEW(1:NATOM)=0.0
          YNEW(1:NATOM)=0.0
          ZNEW(1:NATOM)=0.0
          DO I=ISTRT,ISTOP
             CALL MAGFAC(NAT3,XNORM,YNORM,ZNORM,DDV(1,I),DDM, &
                  DDSCR,DDEV(I),.TRUE.,ITYPE,RTYPE,LNOMA,LNONO,TFREQ, &
                  AFACT,LTFREQ)
             IF(STEP.GT.0.0) THEN
                FRQ=ABS(DDF(I))
                IF(FRQ.LT.TFREQ) FRQ=TFREQ
                PHASE=360.0*STEP*(FRQ*SPEEDL)
                PHASE=PHASE*PI/180.0
             ENDIF

             FACT=SIN(PHASER(I))
             PHASER(I)=PHASER(I)+PHASE
             DO J=1,NATOM
                XNEW(J)=XNEW(J)+XNORM(J)*FACT
                YNEW(J)=YNEW(J)+YNORM(J)*FACT
                ZNEW(J)=ZNEW(J)+ZNORM(J)*FACT
             ENDDO
          ENDDO

          DO J=1,NATOM
             XNEW(J)=XNEW(J)+X(J)
             YNEW(J)=YNEW(J)+Y(J)
             ZNEW(J)=ZNEW(J)+Z(J)
          ENDDO
          IF(LSHAK) THEN
             DO J=1,NATOM
                XTEMP(J)=XNEW(J)
                YTEMP(J)=YNEW(J)
                ZTEMP(J)=ZNEW(J)
             ENDDO
             CALL HOLONOMA(XNEW,YNEW,ZNEW,XTEMP,YTEMP,ZTEMP, &
                  .TRUE.,.FALSE.,QOK)
          ENDIF

          CALL WRITCV(XNEW,YNEW,ZNEW, &
#if KEY_CHEQ==1
               (/ ZERO /), .FALSE.,  & 
#endif
               NATOM, (/ 0 /), NATOM,1,ISTEPT, &
               NAT3,STEP,1,NSTEPT,TITLEA,NTITLA,IUNCRD, &
               .FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
       ENDDO
    ENDDO
    TFREQ=TFREQQ

    RETURN
  END SUBROUTINE VIBSUP

end module vibio

