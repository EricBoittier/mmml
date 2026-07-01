module surfacmod

#if KEY_MPI==1
  use parallel
  use mpi
#else
  INTEGER, PRIVATE :: MYNOD=0, NUMNOD=1
#endif

contains
#if KEY_NOMISC==0
  SUBROUTINE SURFAC(NATOMX,X,Y,Z,WMAIN,LWEIG,ISLCT,NSLCT,SX, &
       SY,SZ,SRAD,SMAIN, &
       MODE,RPRO,NP)
    !-----------------------------------------------------------------------
    !     Richards accessible surface program
    !
    !     For syntax see main parsing loop "HELP"
    !
    !     Program was obtained from Mark Wagman. The use of knset has been
    !     eliminated. The comments have been redone, and the continuation
    !     statements have been reformatted for greater legibility.
    !     The common block has been eliminated, and the data is passed by
    !     subroutine call.
    !
    !     Keyword ACCEss means accessible surface, whereas CONTact means
    !     contact area.
    !
    !     routine originally included by Robert Bruccoleri
    !     calling routine overhauled by Axel Brunger, 14-JAN-84
    !
    !     MPI parallelization by James E. Gonzales II and Wonmuk Hwang,
    !     05-APR-2024
    !
  use chm_kinds
  use dimens_fcm
  use psf
  use param
  use comand
  use stream
  use string
  use machutil,only:die
  use memory
  use param_store, only: set_param

    implicit none

    logical,allocatable,dimension(:) :: ISKIP
    integer,allocatable,dimension(:) :: INTAG1,INTAG,ITAG,IDER &
         ,SIGN_YDER,LT,TAG,TAG1,INZ,KN,KENT,KOUT
    real(chm_real),allocatable,dimension(:) :: XC1, YC1, ZC1, BG &
         , THER, RI, RISQ, B1, DSQ1, BSQ1, GR, XC, YC, ZC, UX, UY &
         , UZ, DSQ, BSQ, B, ARCI1, ARCF1, EX, ZLB, ZUB, ZR1, YR1, XR1 &
         , RAD1, RSEC2, AREA, RSEC
    real(chm_real),allocatable,dimension(:,:) :: arci,arcf
    INTEGER NATOMX
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    LOGICAL LWEIG
    INTEGER ISLCT(*)
    INTEGER NSLCT,MODE
    real(chm_real) SX(*),SY(*),SZ(*)
    real(chm_real) SRAD(*),SMAIN(*)
    real(chm_real) RPRO, NP
    !
    INTEGER I, J, IAT
    !  INTEGER TAG, TAG1, INZ, KN, ZLB, ZUB, ZR1, YR1, XR1, RAD1
    !  INTEGER RSEC2, AREA, RSEC
    !

    integer,PARAMETER :: MARC=2002,MOV=2800,MPT=3000

    !  INTEGER ISKIP,INTAG1,INTAG,ITAG,IDER,SIGN_YDER,XC1,YC1,ZC1, &
    !       BG,THER,RI,RISQ,B1,DSQ1,BSQ1,GR,XC,YC,ZC,UX,UY,UZ, &
    !       DSQ,BSQ,B,KENT,KOUT,ARCI,ARCF,EX,LT
    !
    !     ICT gives the maximum number of contacts for each sphere.
    INTEGER,PARAMETER :: ICT=35

    ! mpi related
    INTEGER CHUNKSIZE(NUMNOD), OFFSET(NUMNOD), IERR
    INTEGER IDUM1, IDUM2

    !     begin

    IF (MYNOD .EQ. 0) THEN
       IF(MODE.LT.0)THEN
          IF(INDXA(COMLYN,COMLEN,'CONT').GT.0) MODE=2
          IF(INDXA(COMLYN,COMLEN,'ACCE').GT.0) MODE=1
          RPRO=1.6
          RPRO=GTRMF(COMLYN,COMLEN,'RPRO',RPRO)
          NP=0.0
          NP=GTRMF(COMLYN,COMLEN,'ACCU',NP)
       ENDIF
    ENDIF ! IF (MYNOD .EQ. 0) THEN
    !
    IF (NSLCT.LE.0) RETURN
    IF(PRNLEV.GE.3) THEN
       IF (LWEIG) THEN
          IF (MYNOD .EQ. 0) THEN
             WRITE(OUTU,'(A)') &
                  ' SURFAC: The weighting array is used for radii'
          ENDIF
       ELSE
          IF (MYNOD .EQ. 0) THEN
             WRITE(OUTU,'(A)') &
                  ' SURFAC: Lennard-Jones radii values being used'
          ENDIF
          !+ln: add check that parameters have actually been read...
          IF(NATC .LE. 0) &
               CALL WRNDIE(-1,'<SURFAC>','No parameters found: all radii=0')
          !-ln
       ENDIF
    ENDIF

#if KEY_MPI==1
       call mpi_barrier(COMM_CHARMM, IERR)
       call psnd8(NP, 1)
       call psnd8(RPRO, 1)
       call mpi_barrier(COMM_CHARMM, IERR)
#endif

    !
    !=======================================================================
    !     Section for analytic surface areas
    IF(NP.LE.0.0) THEN
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IF (LWEIG) THEN
                SRAD(I)=WMAIN(I)+RPRO
             ELSE
                !              RAD is set to sigma/2
                IF (ITC(IAC(I)).LE.0) THEN
                   SRAD(I)=0.0
                ELSE
                   SRAD(I)=VDWR(ITC(IAC(I)))+RPRO
                ENDIF
             ENDIF
          ENDIF
          WMAIN(I)=0.0
       ENDDO
       !
       IF(PRNLEV.GE.3) THEN
          IF (MYNOD .EQ. 0) THEN
             WRITE(OUTU,'(A,F10.5)') ' SURFAC: RPRObe=',RPRO
             WRITE(OUTU,'(A)') &
                  ' SURFAC: Analytic surface area method used'
          ENDIF
       ENDIF
       !
       ! Uniformly distribute atoms among nodes
       IDUM1=NATOM/NUMNOD
       IDUM2=MOD(NATOM, NUMNOD)
       DO I = 1, NUMNOD
          CHUNKSIZE(I)=IDUM1
          IF (I .LE. IDUM2) CHUNKSIZE(I)=IDUM1+1
       ENDDO ! DO I = 1, NUMNOD
       OFFSET(1)=1

#if KEY_MPI==1
       DO I = 2, NUMNOD
          J = I-1
          OFFSET(I)=OFFSET(J)+CHUNKSIZE(J)
       ENDDO ! DO I = 2, NUMNOD
       call mpi_barrier(COMM_CHARMM, IERR)
#endif
       !
       call chmalloc('surfac.src','SURFAC','ISKIP',MOV,log=ISKIP)
       call chmalloc('surfac.src','SURFAC','INTAG1',MOV,intg=INTAG1)
       call chmalloc('surfac.src','SURFAC','INTAG',MOV,intg=INTAG)
       call chmalloc('surfac.src','SURFAC','ITAG',MOV,intg=ITAG)
       call chmalloc('surfac.src','SURFAC','IDER',MOV,intg=IDER)
       call chmalloc('surfac.src','SURFAC','SIGN_YDER',MOV,intg=SIGN_YDER)
       call chmalloc('surfac.src','SURFAC','XC1',MOV,crl=XC1)
       call chmalloc('surfac.src','SURFAC','YC1',MOV,crl=YC1)
       call chmalloc('surfac.src','SURFAC','ZC1',MOV,crl=ZC1)
       call chmalloc('surfac.src','SURFAC','BG',MOV,crl=BG)
       call chmalloc('surfac.src','SURFAC','THER',MOV,crl=THER)
       call chmalloc('surfac.src','SURFAC','RI',MOV,crl=RI)
       call chmalloc('surfac.src','SURFAC','RISQ',MOV,crl=RISQ)
       call chmalloc('surfac.src','SURFAC','B1',MOV,crl=B1)
       call chmalloc('surfac.src','SURFAC','DSQ1',MOV,crl=DSQ1)
       call chmalloc('surfac.src','SURFAC','BSQ1',MOV,crl=BSQ1)
       call chmalloc('surfac.src','SURFAC','GR',MOV,crl=GR)
       call chmalloc('surfac.src','SURFAC','XC',MOV,crl=XC)
       call chmalloc('surfac.src','SURFAC','YC',MOV,crl=YC)
       call chmalloc('surfac.src','SURFAC','ZC',MOV,crl=ZC)
       call chmalloc('surfac.src','SURFAC','UX',MOV,crl=UX)
       call chmalloc('surfac.src','SURFAC','UY',MOV,crl=UY)
       call chmalloc('surfac.src','SURFAC','UZ',MOV,crl=UZ)
       call chmalloc('surfac.src','SURFAC','DSQ',MOV,crl=DSQ)
       call chmalloc('surfac.src','SURFAC','BSQ',MOV,crl=BSQ)
       call chmalloc('surfac.src','SURFAC','B',MOV,crl=B)
       call chmalloc('surfac.src','SURFAC','KENT',MARC,intg=KENT)
       call chmalloc('surfac.src','SURFAC','KOUT',MARC,intg=KOUT)
       call chmalloc('surfac.src','SURFAC','ARCI1',MPT,crl=ARCI1)
       call chmalloc('surfac.src','SURFAC','ARCF1',MPT,crl=ARCF1)
       call chmalloc('surfac.src','SURFAC','EX',MPT,crl=EX)
       call chmalloc('surfac.src','SURFAC','LT',MPT,intg=LT)

       CALL ANAREA(NATOMX,ISLCT,X,Y,Z,SRAD,WMAIN, &
            MARC,MOV,MPT,ISKIP,INTAG1,INTAG, &
            ITAG,IDER,SIGN_YDER, &
            XC1,YC1,ZC1,BG, &
            THER,RI,RISQ, &
            B1,DSQ1,BSQ1,GR, &
            XC,YC,ZC, &
            UX,UY,UZ,DSQ,BSQ, &
            B,KENT,KOUT,ARCI1, &
            ARCF1,EX,LT, &
            CHUNKSIZE,OFFSET)

       !
       call chmdealloc('surfac.src','SURFAC','ISKIP',MOV,log=ISKIP)
       call chmdealloc('surfac.src','SURFAC','INTAG1',MOV,intg=INTAG1)
       call chmdealloc('surfac.src','SURFAC','INTAG',MOV,intg=INTAG)
       call chmdealloc('surfac.src','SURFAC','ITAG',MOV,intg=ITAG)
       call chmdealloc('surfac.src','SURFAC','IDER',MOV,intg=IDER)
       call chmdealloc('surfac.src','SURFAC','SIGN_YDER',MOV,intg=SIGN_YDER)
       call chmdealloc('surfac.src','SURFAC','XC1',MOV,crl=XC1)
       call chmdealloc('surfac.src','SURFAC','YC1',MOV,crl=YC1)
       call chmdealloc('surfac.src','SURFAC','ZC1',MOV,crl=ZC1)
       call chmdealloc('surfac.src','SURFAC','BG',MOV,crl=BG)
       call chmdealloc('surfac.src','SURFAC','THER',MOV,crl=THER)
       call chmdealloc('surfac.src','SURFAC','RI',MOV,crl=RI)
       call chmdealloc('surfac.src','SURFAC','RISQ',MOV,crl=RISQ)
       call chmdealloc('surfac.src','SURFAC','B1',MOV,crl=B1)
       call chmdealloc('surfac.src','SURFAC','DSQ1',MOV,crl=DSQ1)
       call chmdealloc('surfac.src','SURFAC','BSQ1',MOV,crl=BSQ1)
       call chmdealloc('surfac.src','SURFAC','GR',MOV,crl=GR)
       call chmdealloc('surfac.src','SURFAC','XC',MOV,crl=XC)
       call chmdealloc('surfac.src','SURFAC','YC',MOV,crl=YC)
       call chmdealloc('surfac.src','SURFAC','ZC',MOV,crl=ZC)
       call chmdealloc('surfac.src','SURFAC','UX',MOV,crl=UX)
       call chmdealloc('surfac.src','SURFAC','UY',MOV,crl=UY)
       call chmdealloc('surfac.src','SURFAC','UZ',MOV,crl=UZ)
       call chmdealloc('surfac.src','SURFAC','DSQ',MOV,crl=DSQ)
       call chmdealloc('surfac.src','SURFAC','BSQ',MOV,crl=BSQ)
       call chmdealloc('surfac.src','SURFAC','B',MOV,crl=B)
       call chmdealloc('surfac.src','SURFAC','KENT',MARC,intg=KENT)
       call chmdealloc('surfac.src','SURFAC','KOUT',MARC,intg=KOUT)
       call chmdealloc('surfac.src','SURFAC','ARCI1',MPT,crl=ARCI1)
       call chmdealloc('surfac.src','SURFAC','ARCF1',MPT,crl=ARCF1)
       call chmdealloc('surfac.src','SURFAC','EX',MPT,crl=EX)
       call chmdealloc('surfac.src','SURFAC','LT',MPT,intg=LT)


       RETURN
    ENDIF
    !=======================================================================
    !     Section for grid method
    !
    !     map all coordinates according to selected subset
    IF (NUMNOD .GT. 1) THEN
       CALL WRNDIE(1, '<SURFAC>', 'Grid method is not parallelized')
       ! IF (MYNOD .NE. 0) RETURN ! Only root node does the rest
    ENDIF

    IAT=0
    DO I=1,NATOMX
       IF (ISLCT(I).EQ.1) THEN
          IAT=IAT+1
          SX(IAT)=X(I)
          SY(IAT)=Y(I)
          SZ(IAT)=Z(I)
          !
          IF (LWEIG) THEN
             SRAD(IAT)=WMAIN(I)
          ELSE
             !     RAD is set to sigma/2
             IF (ITC(IAC(I)).LE.0) THEN
                SRAD(IAT)=0.0
             ELSE
                SRAD(IAT)=VDWR(ITC(IAC(I)))
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    IF(IAT.NE.NSLCT) CALL DIE
    IF(IAT.LT.ICT) IAT=ICT
    call chmalloc('surfac.src','SURFAC','TAG',IAT,intg=TAG)
    call chmalloc('surfac.src','SURFAC','TAG1',IAT,intg=TAG1)
    call chmalloc('surfac.src','SURFAC','INZ',IAT,intg=INZ)
    call chmalloc('surfac.src','SURFAC','KN',IAT,intg=KN)
    call chmalloc('surfac.src','SURFAC','ZLB',IAT,crl=ZLB)
    call chmalloc('surfac.src','SURFAC','ZUB',IAT,crl=ZUB)
    call chmalloc('surfac.src','SURFAC','ZR1',IAT,crl=ZR1)
    call chmalloc('surfac.src','SURFAC','YR1',IAT,crl=YR1)
    call chmalloc('surfac.src','SURFAC','XR1',IAT,crl=XR1)
    call chmalloc('surfac.src','SURFAC','RAD1',IAT,crl=RAD1)
    call chmalloc('surfac.src','SURFAC','RSEC2',IAT,crl=RSEC2)
    call chmalloc('surfac.src','SURFAC','AREA',IAT,crl=AREA)
    call chmalloc('surfac.src','SURFAC','RSEC',IAT,crl=RSEC)
    call chmalloc('surfac.src','SURFAC','ARCF',ICT,IAT,crl=ARCF)
    call chmalloc('surfac.src','SURFAC','ARCi',ICT,IAT,crl=ARCi)

    CALL SURFA2(NSLCT,ICT,TAG,TAG1,INZ, &
         KN,ZLB,ZUB, &
         ZR1,YR1,XR1,RAD1, &
         RSEC2,AREA,RSEC, &
         ARCI,ARCF,SX,SY,SZ,SRAD, &
         NP,RPRO,SMAIN,MODE)
    !
    !     finally map the WMAIN (=SMAIN) array back.
    J=0
    RPRO=0.0
    DO I=1,NATOMX
       IF (ISLCT(I).EQ.1) THEN
          J=J+1
          WMAIN(I)=SMAIN(J)
          RPRO=RPRO+WMAIN(I)
       ELSE
          WMAIN(I)=0.0
       ENDIF
    ENDDO
    if (prnlev >= 2) WRITE(OUTU,'(A,F15.5)') ' SURFAC: TOTAL = ',RPRO
    call set_param('AREA',RPRO)
    !
    call chmdealloc('surfac.src','SURFAC','TAG',IAT,intg=TAG)
    call chmdealloc('surfac.src','SURFAC','TAG1',IAT,intg=TAG1)
    call chmdealloc('surfac.src','SURFAC','INZ',IAT,intg=INZ)
    call chmdealloc('surfac.src','SURFAC','KN',IAT,intg=KN)
    call chmdealloc('surfac.src','SURFAC','ZLB',IAT,crl=ZLB)
    call chmdealloc('surfac.src','SURFAC','ZUB',IAT,crl=ZUB)
    call chmdealloc('surfac.src','SURFAC','ZR1',IAT,crl=ZR1)
    call chmdealloc('surfac.src','SURFAC','YR1',IAT,crl=YR1)
    call chmdealloc('surfac.src','SURFAC','XR1',IAT,crl=XR1)
    call chmdealloc('surfac.src','SURFAC','RAD1',IAT,crl=RAD1)
    call chmdealloc('surfac.src','SURFAC','RSEC2',IAT,crl=RSEC2)
    call chmdealloc('surfac.src','SURFAC','AREA',IAT,crl=AREA)
    call chmdealloc('surfac.src','SURFAC','RSEC',IAT,crl=RSEC)
    call chmdealloc('surfac.src','SURFAC','ARCF',ICT,IAT,crl=ARCF)
    call chmdealloc('surfac.src','SURFAC','ARCi',ICT,IAT,crl=ARCi)
    !
    RETURN
  END SUBROUTINE SURFAC

  SUBROUTINE SURFA2(NATOM,ICT,TAG,TAG1,INZ,KN,ZLB, &
       ZUB,ZR1,YR1,XR1,RAD1,RSEC2,AREA,RSEC, &
       ARCI,ARCF,XR,YR,ZR,RAD,P,RH2O,ACCESS,MODE)
    !-----------------------------------------------------------------------
    !     ACCESS: CALCULATE ACCESSIBLE CONTACT SURFACE AREA FOR A GROUP OF
    !     ATOMS. THE ACCESSIBLE AREA FOR A GIVEN ATOM IS ¯CALCULATED BY THE
    !     FORMULA,
    !
    !      (ARCSUM) X (ATOM RADIUS+PROBE RADIUS) X (DELTAZ)
    !
    !     NUMERICAL INTEGRATION IS CARRIED OUT OVER Z. IN EACH Z-SECTION,
    !     THE ARCSUM FOR A GIVEN ATOM IS THE ARCLENGTH OF THE CIRCLE
    !     (INTERSECTION OF THE ATOM SPHERE WITH THE Z-SECTION) THAT IS NOT
    !     INTERIOR TO ANY OTHER ATOM CIRCLES IN THE SAME Z-SECTION.
    !
  use chm_kinds
  use number
  use stream
    implicit none
    !
    INTEGER NATOM, ICT, TAG(*), TAG1(*), INZ(*), KN(*)
    real(chm_real)  ZLB(*), ZUB(*), XR1(*), YR1(*), ZR1(*), RAD1(*)
    real(chm_real)  RSEC2(*), AREA(*), RSEC(*), ARCI(ICT,*),  &
         ARCF(ICT,*)
    real(chm_real)  XR(*), YR(*), ZR(*)
    real(chm_real)  RAD(*), P, RH2O, ACCESS(*)
    INTEGER MODE
    !     local
    INTEGER ANO, I, IDUM, N, K, J, J1, J2, J3, M, IEND, ITAB, ICT1
    INTEGER IANO, ICNT1, ICNT2, IFLAG
    real(chm_real) PIE, RMIN, PIEX2, ZUBMAX, RINC, ZOR, HZRES,  &
         ZRES, CUTOFF
    real(chm_real) ZNEXT, ZGRID, A, DX, DY, D, B, Q, D2
    !
    !     begin
    PIE=ACOS(MINONE)
    PIEX2=TWO*PIE
    ICT1=ICT-1
    ANO=NATOM
    RMIN=10000.D0
    !
    !
    !     CALCULATE LOWEST ZBOUND FOR EACH ATOM AND SORT ATOMS FROM LOW TO
    !     HIGH ON ZLB
    DO I=1,ANO
       ZLB(I)=ZR(I)-RAD(I)
    ENDDO
    CALL SORTAG(ZLB,ANO,TAG)
    DO I=1,ANO
       J=TAG(I)
       TAG1(J)=I
       XR1(I)=XR(J)
       YR1(I)=YR(J)
       ZR1(I)=ZR(J)
       RAD1(I)=RAD(J)
       IF(RAD1(I).LT.RMIN)RMIN=RAD1(I)
    ENDDO
    ZUBMAX=0.0
    !
    !     THE RADIUS OF AN ATOM SPHERE = ATOM RADIUS + PROBE RADIUS
    !
    RINC=RH2O
    ZOR=ZLB(1)-RINC
    DO I=1,ANO
       RAD1(I)=RAD1(I)+RINC
       ZR1(I)=ZR1(I)-ZOR
       ZUB(I)=ZR1(I)+RAD1(I)
       ZLB(I)=ZR1(I)-RAD1(I)
       IF(ZUB(I).GT.ZUBMAX)ZUBMAX=ZUB(I)
       AREA(I)=0.0
    ENDDO
    !
    !     Z RESOLUTION DETERMINED
    !
    HZRES=(RMIN+RH2O)*P
    ZRES=2.0*HZRES
    CUTOFF=ZRES/100.
    IANO=1
    J1=1
    IFLAG=0
    ICNT1=0
    ICNT2=0
    ZNEXT=ZRES
    IEND=ZUBMAX/ZRES
    !
    !     SECTION ATOM SPHERES PERPENDICULAR TO THE Z AXIS
    !
    loop900: DO I=1,IEND
       ITAB=0
       ZGRID=ZNEXT
       ZNEXT=ZGRID+ZRES
       loop100: DO N=IANO,ANO
          !
          !     THE UPPER AND LOWER Z BOUNDS OF AN ATOM ARE USED TO DETERMINE
          !     WHETHER THIS Z-SECTION CUTS THE ATOM SPHERE
          !
          IF(ZUB(N).LE.ZGRID) GOTO 21
          IF(ZLB(N).GE.ZGRID) GOTO 30
          KN(N)=0
          DO K=1,ICT
             ARCI(K,N)=0.0
          ENDDO
          !
          !     COUNT AND RECORD SPHERES CUT BY SECTION
          !
          ITAB=ITAB+1
          INZ(ITAB)=N
          !
          !     FIND RADIUS OF CIRCLE LOCUS
          !
          A=RAD1(N)**2-(ZGRID-ZR1(N))**2
          RSEC2(N)=A
          RSEC(N)=SQRT(A)
21        CONTINUE
          IF(IFLAG /= 1) then         !  .EQ.1) GOTO 10
             !
             !     FIND 1ST SPHERE CUT BY SECTION I,SKIP ATOMS PREDEEDING IT IN LIST
             !     FOR NEXT SECTION
             !
             IF(ZUB(N).GT.ZNEXT) then !  GOTO 32
                IFLAG=1               !(line 32)
             else
                J1=J1+1
             endif
          endif                       !  10      CONTINUE
       enddo loop100 !  100     CONTINUE
30     IANO=J1
       IFLAG=0
       !
       !     ZERO, ONE, OR MORE CIRCLES ON SECTION REQUIRE DIFFERENT PROCESSING
       !
       IF(ITAB > 1 ) then   ! .LT.1) GOTO 890
          IF(ITAB /= 1) then   ! .EQ.1) GOTO 880
             J3=ITAB-1
             !
             !     FIND INTERSECTIONS OF CIRCLES IN SECTION CALARC CALLED TO FIND
             !     INITIAL AND FINAL ANGLES OF INTERSECTION OF CIRCLES. IF ARCI AND
             !     ARCF ARRAYS ARE FILLED, REDUCE IS CALLED. THE CURRENT 1ST INDEX OF
             !     THESE ARRAYS FOR THE ATOM INDICATED BY 2ND INDEX IS STORED IN THE
             !     ARRAY KN, IF KN FOR ANY ATOM IS 10000, THEN NO AREA REMAINS
             !     ACCESSIBLE FOR THAT ATOM ON THIS SECTION OR THE ATOM IS NOT OF
             !     INTEREST(KNSET=1).
             !
             loop600: DO K=1,J3
                N=INZ(K)
                J2=K+1
                loop570: DO J=J2,ITAB
                   M=INZ(J)
                   A=RSEC(M)+RSEC(N)
                   DX=XR1(M)-XR1(N)
                   IF(ABS(DX).GE.A) cycle loop570
                   DY=YR1(M)-YR1(N)
                   IF(ABS(DY).GE.A) cycle loop570
                   D2=DY**2+DX**2
                   D=SQRT(D2)
                   IF(D.GE.A) cycle loop570
                   IF(KN(N) >= ICT1) then  !GOTO 4
                      IF(KN(N).LE.ICT) THEN
                         CALL REDUCE(N,0,ICT,ARCI,ARCF,TAG,TAG1,ZR1,RAD1,AREA, &
                              INZ,KN,ICNT1,ICNT2, &
                              PIE,PIEX2,ICT1,ZRES,RH2O,ZGRID,HZRES,CUTOFF)
                      ENDIF
                   endif
                   KN(N)=KN(N)+1
                   IF(KN(M) >= ICT1) then
                      IF(KN(M).LE.ICT) THEN
                         CALL REDUCE(M,0,ICT,ARCI,ARCF,TAG,TAG1,ZR1,RAD1,AREA, &
                              INZ,KN,ICNT1,ICNT2, &
                              PIE,PIEX2,ICT1,ZRES,RH2O,ZGRID,HZRES,CUTOFF)
                      ENDIF
                   endif
                   KN(M)=KN(M)+1
                   !
                   !     DO THE CIRCLES INTERSECT, OR IS ONE COMPLETELY INSIDE THE OTHER?
                   !
                   B=RSEC(M)-RSEC(N)
                   IF(D <= ABS(B)) then
                      IF(B <= 0.0) then
                         KN(M)=10000
                         KN(N)=KN(N)-1
                         cycle loop570
                      endif
                      KN(N)=10000
                      KN(M)=KN(M)-1
                      cycle loop570
                      !
                      !     IF THE CIRCLES INTERSECT, FIND THE POINTS OF INTERSECTION
                      !
                   endif
                   Q=RSEC2(M)-RSEC2(N)
                   D=2.0*D
                   IF(KN(M) <= ICT) &
                        CALL CALARC(M,ONE,ICT,D2,Q,D,RSEC,ARCI,ARCF,KN,DY,DX,PIE, &
                        PIEX2)
                   IF(KN(N) <= ICT)  &
                        CALL CALARC(N,MINONE,ICT,D2,Q,D,RSEC,ARCI,ARCF,KN,DY,DX, &
                        PIE,PIEX2)
                enddo loop570
             enddo loop600
             !
             !     FIND THE ACCESSIBLE CONTACT SURFACE AREA FOR ALL THE SPHERES
             !     INTERSECTING THIS SECTION
             !
          endif   ! 880     CONTINUE
          CALL REDUCE(IDUM,ITAB,ICT,ARCI,ARCF,TAG,TAG1,ZR1,RAD1, &
               AREA,INZ,KN,ICNT1,ICNT2, &
               PIE,PIEX2,ICT1,ZRES,RH2O,ZGRID,HZRES,CUTOFF)
       endif   !   890     CONTINUE
    enddo loop900
    !
    !     OUTPUT OPERATION PARAMETERS
    !
    IF(PRNLEV.GE.2) THEN
       IF (MODE.EQ.1) THEN
          WRITE(OUTU,'(A)') ' SURFAC: ACCEssible area'
       ELSE
          WRITE(OUTU,'(A)') ' SURFAC: CONTact area'
       ENDIF
       WRITE(OUTU,9000) &
            ' SURFAC: ACCUracy=',P,' RPRObe=',RH2O, &
            '         Z-grid=',ZRES,' number-of-Z-sections=',IEND, &
            '         measures-of-arc=',ICNT1,' and',ICNT2
    ENDIF
9000 FORMAT(A,F10.5,A,F10.5,/,A,F10.5,A,I5,/,A,I6,A,I6)
    !
    DO J=1,ANO
       I=TAG1(J)
       !
       !     SCALE AREA TO VDW SHELL IF NECESSARY
       !
       IF (MODE.EQ.2) THEN
          ACCESS(J)=AREA(I)*((RAD1(I)-RH2O)/RAD1(I))**2
       ELSE
          ACCESS(J)=AREA(I)
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE SURFA2

  SUBROUTINE CALARC(M,SIGN,ICT,D2,Q,D,RSEC,ARCI,ARCF,KN,DY,DX, &
       PIE,PIEX2)
    !-----------------------------------------------------------------------
    !     INITIAL AND FINAL ARC ENDPOINTS ARE FOUND FOR A REFERENCE CIRCLE
    !     INTERSECTED BY ANOTHER CIRCLE CONTAINED IN THE SAME PLANE. THE
    !     INITIAL ENDPOINT OF THE ENCLOSED ARC IS STORED IN ARCI, AND THE
    !     FINAL ARC IN ARCF
    !
  use chm_kinds
    implicit none
    INTEGER M
    real(chm_real) SIGN
    INTEGER ICT
    real(chm_real) D2, Q, D, RSEC(*), ARCI(ICT,*), ARCF(ICT,*)
    INTEGER KN(*)
    real(chm_real) DY, DX, PIE, PIEX2
    !     local
    INTEGER K1
    real(chm_real) ARG, ALPHA1, BETA1, TI, TF
    !
    !     begin
    K1=KN(M)
    !
    !     LAW OF COSINES
    !
    ARG=(D2+Q*SIGN)/(D*RSEC(M))
    IF (ARG.GT.1.0) ARG=1.0
    IF (ARG.LT.-1.0) ARG=-1.0
    ALPHA1=ACOS(ARG)
    !
    !     ALPHA1 IS THE ANGLE BETWEEN A LINE CONTAINING A POINT OF
    !     INTERSECTION AND THE REFERENCE CIRCLE CENTER AND THE LINE
    !     CONTAINING BOTH CIRCLE CENTERS
    !
    IF (DY.EQ.0.0 .AND. DX.EQ.0.0) DX=1.0E-30
    BETA1=ATAN2(DY,DX)
    !
    !     BETA1 IS THE ANGLE BETWEEN THE LINE CONTAINING BOTH CIRCLE CENTERS
    !     AND THE X-AXIS
    !
    IF(SIGN.EQ.1.0)BETA1=BETA1+PIE
    TI=BETA1-ALPHA1
    TF=BETA1+ALPHA1
    IF(TI.LT.0.0)TI=TI+PIEX2
    IF(TF.GT.PIEX2)TF=TF-PIEX2
    IF(TF.LT.0.0)TF=TF+PIEX2
    IF(TF.GE.TI) GOTO 3
    !
    !     IF THE ARC CROSSES ZERO, THEN BREAK IT INTO TWO SEGMENTS. THE
    !     FIRST ENDS AT 2XPI AND THE SECOND BEGINS AT ZERO
    !
    ARCF(K1+1,M)=TF
    ARCF(K1,M)=PIEX2
    KN(M)=KN(M)+1
    GOTO 2
3   ARCF(K1,M)=TF
2   ARCI(K1,M)=TI
    RETURN
  END SUBROUTINE CALARC

  SUBROUTINE REDUCE(N,ITAB,ICT,ARCI,ARCF,TAG,TAG1,ZR1,RAD1, &
       AREA,INZ,KN,ICNT1,ICNT2, &
       PIE,PIEX2,ICT1,ZRES,RH2O,ZGRID,HZRES,CUTOFF)
    !-----------------------------------------------------------------------
    !     1) ITAB=0, REMOVE DEGENERACIES IN ARCI AND ARCF FOR A SINGLE ATOM
    !     SINCE THESE ARRAYS ARE FILLED, OR 2) ITAB>0, CALCULATE THE
    !     ACCESSIBLE SURFACE AREA FOR THE N ATOMS IN THIS SECTION
    !
  use chm_kinds
  use stream
  use machutil,only:die
    implicit none
    !
    INTEGER N, ITAB, ICT
    real(chm_real)    ARCI(ICT,*),ARCF(ICT,*)
    INTEGER TAG(*),TAG1(*)
    real(chm_real)    ZR1(*),RAD1(*),AREA(*)
    INTEGER INZ(*), KN(*), ICNT1, ICNT2
    real(chm_real)    PIE, PIEX2
    INTEGER ICT1
    real(chm_real)    ZRES, RH2O, ZGRID, HZRES, CUTOFF
    !     local
    INTEGER II, K1, JJ, K, M, I, I1, J
    real(chm_real)    ARCSUM, T, A, PAREA
    !
    !     begin
    loop400: DO II=1,MAX(1,ITAB)
       !
       !     N IS PASSED IN THE SUBROUTINE ARGUMENT LIST FOR 1), OR IN INZ FOR 2)
       !
       IF(ITAB.NE.0)N=INZ(II)
       K1=KN(N)
       IF(K1.GT.ICT) cycle loop400    ! GOTO 390
       !
       !     IF K1>ICT, THIS ATOM IS NOT OF INTEREST OR IT HAS NO AREA
       !     REMAINING IS THIS CIRCLE INTERSECTED BY OTHERS?
       !
       IF(K1.NE.0) then        !GOTO 190
          !
          !     THE ARC ENDPOINTS ARE SORTED ON THE VALUE OF THE INITIAL ARC
          !     ENDPOINT
          !
          !  190     CONTINUE
          CALL SORTAG(ARCI(1,N),K1,TAG)
          !
          !     CALCULATE THE ACCESSIBLE AREA
          !
          ARCSUM=ARCI(1,N)
          JJ=TAG(1)
          T=ARCF(JJ,N)
          IF(K1 /= 1) then ! GOTO 320
             loop300: DO K=2,MAX(2,K1)
                IF(T < ARCI(K,N)) then     !  GOTO 250
                   ARCSUM=ARCSUM+ARCI(K,N)-T
                   JJ=TAG(K)
                   T=ARCF(JJ,N)
                   ! GOTO 290
                else
                   M=TAG(K)
                   IF(ARCF(M,N).GT.T)T=ARCF(M,N)
                endif
             enddo loop300
          endif
          ARCSUM=ARCSUM+PIEX2-T
       else
          !
          !     THERE IS ONLY ONE CIRCLE IN THIS SECTION
          !
          ARCSUM=PIEX2
       endif        !  GOTO 350
       A=ZR1(N)-ZGRID    !(350)
       A=RAD1(N)-ABS(A)
       !
       !     THE AREA IS EQUAL TO THE ACCESSIBLE ARC LENGTH X THE SECTION
       !     THICKNESS, CORRECTED IF IT IS THE FIRST OR LAST SECTION, X THE
       !     RADIUS OF THE SPHERE
       !
       PAREA=ARCSUM*(HZRES+MIN(A,HZRES))*RAD1(N)
       !
       !     ADD THE ACCESSIBLE AREA FOR THIS ATOM IN THIS SECTION TO THE AREA
       !     FOR THIS ATOM FOR ALL THE SECTION ENCOUNTERED THUS FAR
       !
       AREA(N)=AREA(N)+PAREA
    enddo loop400
    !
    !     IF THIS WAS THE FINAL ATOM FOR THIS Z SECTION RETURN TO ACCESS
    !
    IF(ITAB.GT.0) GOTO 990
    !
    !     THE ARCI AND ARCF ARRAYS WERE FILLED, DOES SIGNIFICANT AREA
    !     REMAIN?
    !
    IF(PAREA.GT.CUTOFF) GOTO 470
    !
    !     NO, THIS ATOM IS USED ONLY IN CALCS FOR OTHER ATOMS, UNTIL NEXT Z
    !     SECTION
    !
    KN(N)=10000
    ICNT1=ICNT1+1
    GOTO 990
    !
    !     YES, REMOVE DEGENERACIES FROM ARCI AND ARCF (COMBINE OVERLAPPING
    !     ARCS)
    !
470 JJ=TAG(1)
    T=ARCF(JJ,N)
    I=1
    !
    loop110: DO K=2,MAX(2,K1)
       IF(T.GE.ARCI(K,N)) GOTO 30
       I=I+1
       IF(I.EQ.ICT1) GOTO 40
       I1=I-1
       DO J=K,MAX(K,K1)
          IF(TAG(J).EQ.I1) GOTO 60
       ENDDO
       GOTO 90
60     ARCF(JJ,N)=ARCF(I1,N)
       TAG(J)=JJ
90     ARCF(I1,N)=T
       ARCI(I,N)=ARCI(K,N)
       JJ=TAG(K)
       T=ARCF(JJ,N)
       GOTO 20
30     M=TAG(K)
       IF(ARCF(M,N).GT.T)T=ARCF(M,N)
20     CONTINUE
    enddo loop110
    ARCF(I,N)=T
    KN(N)=I
    I=I+1
    DO K=I,MAX(K1,I)
       ARCI(K,N)=0.0
    ENDDO
    !
    !     REMOVE THE PARTIAL AREA ADDED, THIS CIRCLE MAY HAVE MORE
    !     INTERSECTIONS
    !
    AREA(N)=AREA(N)-PAREA
    ICNT2=ICNT2+1
    GOTO 990
    !
    !      NO DEGENERACIES WERE FOUND IN THE OVERLAPS, START OVER WITH LARGER
    !      ICT
    !
40  CONTINUE
    IF(WRNLEV.GE.2) WRITE(OUTU,15) TAG1(N),ICT
15  FORMAT('SURFAC>  ERROR: ATOM ',I5, &
         ' HAS MORE THAN ',I4,' CONTACTS')
    CALL DIE
990 CONTINUE
    RETURN
  END SUBROUTINE REDUCE

  SUBROUTINE ANAREA(NATOM,ISLCT,X,Y,Z,RX,AREA, &
       MARC,MOV,MPT,ISKIP,INTAG1,INTAG,ITAG,IDER, &
       SIGN_YDER,XC1,YC1,ZC1,BG,THER,RI,RISQ,B1,DSQ1,BSQ1,GR,XC, &
       YC,ZC,UX,UY,UZ,DSQ,BSQ,B,KENT,KOUT,ARCI,ARCF,EX,LT, &
       CHUNKSIZE,OFFSET)
    !-----------------------------------------------------------------------
    !  This routine contains the actual adaptation of the Richmond
    !  routine to calculate solvent accessible surface areas.
    !       See: J. Mol Biol 178: (1) 63-89 1984.
    !
    !  ANALYTICAL ACCESSIBLE SURFACE AREA AND GRADIENT CALCULATION
    !      T.J.RICHMOND
    !      Modified 9/86 by Morgan Wesson
    !**
    !      for fixed atoms:  the routine calculates derivatives for atom pairs
    !      if one of the atoms is not fixed.
    !      The derivatives of fixed atoms are set to 0 at the end of the routine.
    !      The solvation energy of fixed atoms doesn't contribute to the total
    !      energy that is output.
    !
    !      Arguments passed in:
    !     X,Y,Z - Coordinates for energy evaluation
    !** Indices:
    !**      MARC = max. no. of partial arcs for IR sphere (ith sphere)
    !**      MAXA = max. no. of atoms
    !**      MOV  = max. no. of IN overlapping spheres (j & k spheres for the ith)
    !**      MPT  = max. no. of overlap end pts. on a circle of intersection
    !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use stream
  use param_store, only: set_param
  use memory

    implicit none

    ! definitions of passed arguments...
    INTEGER NATOM
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER ISLCT(*)
    real(chm_real) RX(*), AREA(*)
    !
    INTEGER MARC,MOV,MPT
    LOGICAL ISKIP(MOV)
    INTEGER INTAG1(MOV),INTAG(MOV),ITAG(MOV),IDER(MOV),SIGN_YDER(MOV)
    real(chm_real)   XC1(MOV),YC1(MOV),ZC1(MOV), &
         BG(MOV),THER(MOV),RI(MOV),RISQ(MOV), &
         B1(MOV),DSQ1(MOV),BSQ1(MOV),GR(MOV), &
         XC(MOV),YC(MOV),ZC(MOV), &
         UX(MOV),UY(MOV),UZ(MOV), &
         DSQ(MOV),BSQ(MOV),B(MOV)
    INTEGER   KENT(MARC),KOUT(MARC)
    real(chm_real)    ARCI(MPT),ARCF(MPT),EX(MPT)
    INTEGER LT(MPT)
    !
    INTEGER IERR, CHUNKSIZE(NUMNOD), OFFSET(NUMNOD), N1, N2
    !
    ! local logical variables...
    logical QLONE,LTOP,ISKIPS
    !
    ! local integer variables...
    integer IB_LOCAL, JB_LOCAL, I, IR, IO, IN, K, L, IO1, K1, &
         KARC, MI, N, J, M, II, IFAIL
    !
    ! local real variables...
    real(chm_real) SIG, SIGSQ, PIX2, PIX4, PID2, EE, ARCLEN, EXANG, &
         XR, YR, ZR, RR, RRX2, RRSQ, RPLUS, TX, TY, TZ, XYSQ, &
         CCSQ, CC, RMINUS, TXK, TYK, TZK, BSQK, BK, GL, THERK, TD, &
         DK, GK, RISQK, RIK, T1, AXX, AXY, AXZ, AYX, AYY, AZX, AZY, &
         AZZ, TXL, TYL, TZL, UXL, UYL, UZL, DSQL, TB, TXB, TYB, TR, &
         TXR, TYR, TK1, TK2, THE, TF, ARCSUM, T, TT, RCN, BGL, BSQL, &
         RISQL, WXLSQ, WXL, P, V, DEAL, DECL, DTKAL, DTKCL, S, &
         T2, DTLAL, DTLCL, GACA, GACB, FACA, FACB, FACC, DAX, DAY, DAZ, &
         TI, ACOS_INPUT, SQRT_INPUT
    real(chm_real), allocatable, dimension(:) :: AREA_TMP
    !
    !CC##IF INTEL
    !CC      real(chm_real) DINMOD
    !CC      EXTERNAL DINMOD
    !CC##ENDIF
    !
#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM, IERR)
#endif

    call chmalloc('surfac.src', 'ANAREA', 'AREA_TMP', NATOM, cr8=AREA_TMP)

#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM, IERR)
#endif
    !
    IF (MYNOD .EQ. 0) THEN
       IF(PRNLEV.GT.8) THEN
          WRITE(OUTU,'(A)')  'Entering ASPEN1'
          WRITE(OUTU,'(2A)') ' Atom', &
               '     X         Y         Z     VDW_SURF   ISLCT'
          do I = 1, NATOM
             WRITE (OUTU,'(I5,4F10.5,I5)') I, &
                  X(I),Y(I),Z(I),RX(I),ISLCT(I)
          ENDDO
       ENDIF
    ENDIF ! IF (MYNOD .EQ. 0) THEN
    !
    !** Overlap significance (also used to test if spheres colinear)
    SIG=.01
    SIGSQ=SIG**2
    PIX2=2.*PI
    PIX4=4.*PI
    PID2=PI/2.
    !
    DO I=1,MOV
       IDER(I)=0
       SIGN_YDER(I)=0
    ENDDO
    !
    !** Set start and stop indexes for mpi
    N1=OFFSET(MYNOD+1)
    N2=N1+CHUNKSIZE(MYNOD+1)-1
#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM, IERR)
#endif
    !
    !** Process each atom
    !** Find the IN spheres which overlap the IR sphere
    DO IR=N1,N2
       IF(PRNLEV.GT.8) THEN
          WRITE(OUTU,*)
          WRITE(OUTU,'(I5,4F10.5,I5)') IR, &
               X(IR),Y(IR),Z(IR),RX(IR), ISLCT(IR)
       ENDIF
       ! otherwise, skip this atom.
       IF(ISLCT(IR).EQ.1) THEN
          QLONE=.FALSE.
          IO=1
          JB_LOCAL=0
          IB_LOCAL=0
          ARCLEN=ZERO
          EXANG=ZERO
          AREA (IR ) = ZERO
          AREA_TMP(IR) = ZERO
          XR=X(IR)
          YR=Y(IR)
          ZR=Z(IR)
          RR=RX(IR)
          RRX2=RR*2.
          RRSQ=RR**2
          loop12: do IN = 1, NATOM
             IF(ISLCT(IN).EQ.0) cycle loop12
             ! Is the IN sphere next to the IR sphere
             RPLUS=RR+RX(IN)
             !
             TX=X(IN)-XR
             ! exit IN loop
             IF(ABS(TX).GE.RPLUS)cycle loop12
             !
             TY=Y(IN)-YR
             ! exit IN loop
             IF(ABS(TY).GE.RPLUS)cycle loop12
             !
             TZ=Z(IN)-ZR
             ! exit IN loop
             IF(ABS(TZ).GE.RPLUS)cycle loop12
             !
             !            exclude in = ir
             ! exit IN loop
             if (IN.eq.IR) cycle loop12
             !
             !**             Check for overlap of spheres by testing center to center distance
             !**             Against sum and difference of radii
             XYSQ=TX**2+TY**2
             IF (XYSQ.lt.SIGSQ) then
                TX=SIG
                TY=ZERO
                XYSQ=SIGSQ
             ENDIF
             !
             CCSQ=XYSQ+TZ**2
             CC=SQRT(CCSQ)
             ! exit IN loop
             IF(RPLUS-CC.LE.SIG) cycle loop12
             RMINUS=RR-RX(IN)
             if (CC-ABS(RMINUS).le.SIG) then
                IF(RMINUS.LE.ZERO) then
                   ! IR atom is buried, go to next atom in ir loop
                   go to 4
                else
                   ! IN atom is buried, exit IN loop
                   cycle loop12
                ENDIF
             ENDIF
             !**             Calc. overlap parameters
             IF(IO.GT.MOV) CALL WRNDIE(-3,'<ANAREA>', &
                  'Maximum overlapping spheres limit exceeded.')
             XC1(IO)=TX
             YC1(IO)=TY
             ZC1(IO)=TZ
             !
             DSQ1(IO)=XYSQ
             BSQ1(IO)=CCSQ
             B1(IO)=CC
             ! this is the distance from the IR sphere to the plane containing its
             ! intersection with the IN sphere, divided by the radius of the IR sphere.
             GR(IO)=(CCSQ+RPLUS*RMINUS)/(RRX2*B1(IO))
             INTAG1(IO)=IN
             IO=IO+1
          enddo loop12
          IO=IO-1
          !
          if (IO.eq.0) then
             AREA_TMP(IR) =PIX4
             GO TO 16
          ENDIF
          !
          if (IO.eq.1) then
             K=1
             QLONE=.TRUE.
             TXK=XC1(1)
             TYK=YC1(1)
             TZK=ZC1(1)
             BSQK=BSQ1(1)
             BK=B1(1)
             INTAG(1)=INTAG1(1)
             ARCSUM=PIX2
             IB_LOCAL=IB_LOCAL+1
             !mbm..27-Dec-94 Fujitsu Port / Bugfix
             !mbm..      GO TO 19
             ARCLEN=ARCLEN+GR(K)*ARCSUM
             IN=INTAG(K)
             T1=ARCSUM*RRSQ*(BSQK-RRSQ+RX(IN)**2)/(RRX2*BSQK*BK)
             goto 56
             !mbm..
          ENDIF
          !**           Sort IN spheres by degree of overlap with IR sphere
          IFAIL=0
          CALL SORTAG(GR,IO,ITAG)
          do L=1,IO
             K=ITAG(L)
             IN=INTAG1(K)
             INTAG(L)=IN
             !
             XC(L)=XC1(K)
             YC(L)=YC1(K)
             ZC(L)=ZC1(K)
             !
             DSQ(L)=DSQ1(K)
             B(L)=B1(K)
             BSQ(L)=BSQ1(K)
             ISKIP(L)=.FALSE.
          ENDDO
          !
          do L=1,IO
             GL=GR(L)*RR
             BG(L)=B(L)*GL
             RISQ(L)=RRSQ-GL**2
             RI(L)=SQRT(RISQ(L))
             !**             Radius of the IN circle on the surface of the sphere
             THER(L)=PID2-ASIN(GR(L))
          ENDDO
          !
          !** Find boundary of inaccessible area on IR sphere
          IO1=IO-1
          DO K=1,IO1
             if (.not.ISKIP(K)) then
                TXK=XC(K)
                TYK=YC(K)
                TZK=ZC(K)
                BK=B(K)
                THERK=THER(K)
                K1=K+1
                loop31: DO L=K1,IO
                   if (.not.ISKIP(L)) then
                      !** Is L circle intersecting K circle?
                      !** Distance between circle centers and sum of radii
                      ACOS_INPUT = (TXK*XC(L)+TYK*YC(L)+TZK*ZC(L))/(BK*B(L))
                      ACOS_INPUT = MAX( ACOS_INPUT,MINONE )
                      ACOS_INPUT = MIN( ACOS_INPUT, ONE )
                      CC = acos( ACOS_INPUT )
                      TD=THERK+THER(L)
                      !** Circles enclose separate regions?
                      IF(CC.GE.TD)cycle loop31
                      !** Circle L completely inside circle K?
                      if (CC+THER(L).lt.THER(K)) then
                         ISKIP(L)=.TRUE.
                      else
                         !** Circles essentially parallel?
                         if (CC.gt.SIG) then
                            !** IR sphere completely buried?
                            ! IR atom is buried, go to next atom in IR loop.
                            if (PIX2-CC.le.TD) go to 4
                         else
                            ISKIP(L)=.TRUE.
                         ENDIF
                      ENDIF
                   ENDIF
                enddo loop31
             ENDIF
          ENDDO
          !** Find T value of circle intersections
          do K=1,IO
             if (.not.ISKIP(K)) then
                ISKIPS=ISKIP(K)
                ISKIP(K)=.TRUE.
                KARC=0
                LTOP=.FALSE.
                TXK=XC(K)
                TYK=YC(K)
                TZK=ZC(K)
                DK=SQRT(DSQ(K))
                BSQK=BSQ(K)
                BK=B(K)
                GK=GR(K)*RR
                RISQK=RISQ(K)
                RIK=RI(K)
                THERK=THER(K)
                !** Rotation matrix elements
                T1=TZK/(BK*DK)
                AXX=TXK*T1
                AXY=TYK*T1
                AXZ=DK/BK
                AYX=TYK/DK
                AYY=TXK/DK
                AZX=TXK/BK
                AZY=TYK/BK
                AZZ=TZK/BK
                DO L=1,IO
                   if (.not.ISKIP(L))then
                      TXL=XC(L)
                      TYL=YC(L)
                      TZL=ZC(L)
                      !** Rotate spheres so K vector colinear with z-axis
                      UXL=TXL*AXX+TYL*AXY-TZL*AXZ
                      UYL=TYL*AYY-TXL*AYX
                      UZL=TXL*AZX+TYL*AZY+TZL*AZZ
                      ACOS_INPUT = UZL/B(L)
                      ACOS_INPUT = MAX( ACOS_INPUT, MINONE )
                      ACOS_INPUT = MIN( ACOS_INPUT, ONE )
                      if (acos( ACOS_INPUT ).lt.THERK+THER(L)) then
                         GL=GR(L)*RR
                         DSQL=UXL**2+UYL**2
                         TB=UZL*GK-BG(L)
                         TXB=UXL*TB
                         TYB=UYL*TB
                         TD=RIK*DSQL
                         SQRT_INPUT = RISQK * DSQL - TB**2
                         SQRT_INPUT = MAX( SQRT_INPUT, ZERO )
                         TR = SQRT( SQRT_INPUT )
                         TXR=UXL*TR
                         TYR=UYL*TR
                         !** T values of intersection for K circle
                         TB=(TXB+TYR)/TD
                         if (ABS(TB).gt.ONE) TB=SIGN(ONE,TB)
                         TK1=ACOS(TB)
                         if (TYB-TXR.lt.ZERO) TK1=PIX2-TK1
                         TB=(TXB-TYR)/TD
                         if (ABS(TB).gt.ONE) TB=SIGN(ONE,TB)
                         TK2=ACOS(TB)
                         if (TYB+TXR.lt.ZERO) TK2=PIX2-TK2
                         ACOS_INPUT = (RRSQ*UZL-GK*BG(L))/(RIK*RI(L)*B(L))
                         ACOS_INPUT = MAX( ACOS_INPUT, MINONE )
                         ACOS_INPUT = MIN( ACOS_INPUT, ONE )
                         THE = -acos( ACOS_INPUT )
                         !** Is TK1 entry or exit point?  check T=0 point.
                         !** TI IS EXIT PT., TF IS ENTRY PT.
                         ACOS_INPUT = (UZL*GK-UXL*RIK)/(B(L)*RR)
                         IF(WRNLEV.GE.2) THEN
                            if (ACOS_INPUT.gt.ONE) then
                               WRITE(OUTU,*)'ASPENER: acos input is ',ACOS_INPUT
                            elseif (ACOS_INPUT.lt.MINONE) then
                               WRITE(OUTU,*)'ASPENER: acos input is ',ACOS_INPUT
                            endif
                         ENDIF
                         ACOS_INPUT = MAX( ACOS_INPUT, MINONE )
                         ACOS_INPUT = MIN( ACOS_INPUT, ONE )
                         if ((ACOS(ACOS_INPUT)-THER(L))*(TK2-TK1).le.0) then
                            TI=TK2
                            TF=TK1
                         else
                            TI=TK1
                            TF=TK2
                         endif
                         KARC=KARC+1
                         IF(KARC.GT.MPT) CALL WRNDIE(-3,'<ANAREA>', &
                              'Maximum overlap end point limit exceeded')
                         if (TF.le.TI) then
                            ARCF(KARC)=TF
                            ARCI(KARC)=ZERO
                            TF=PIX2
                            LT(KARC)=L
                            EX(KARC)=THE
                            LTOP=.TRUE.
                            KARC=KARC+1
                         ENDIF
                         ARCF(KARC)=TF
                         ARCI(KARC)=TI
                         LT(KARC)=L
                         EX(KARC)=THE
                         UX(L)=UXL
                         UY(L)=UYL
                         UZ(L)=UZL
                      ENDIF
                   ENDIF
                ENDDO
                ISKIP(K)=ISKIPS
                !** Special case: K circle without intersections?
                if (KARC.LE.0) then
                   ARCSUM=PIX2
                   IB_LOCAL=IB_LOCAL+1
                   go to 19
                ENDIF
                !** General case: sum up arclength and set connectivity code
                IFAIL=0
                CALL SORTAG(ARCI,KARC,ITAG)
                ARCSUM=ARCI(1)
                MI=ITAG(1)
                T=ARCF(MI)
                N=MI
                if (KARC.ne.1) then
                   do J=2,KARC
                      M=ITAG(J)
                      if (T.lt.ARCI(J)) then
                         ARCSUM=ARCSUM+ARCI(J)-T
                         EXANG=EXANG+EX(N)
                         JB_LOCAL=JB_LOCAL+1
                         IF(JB_LOCAL.GT.MARC) CALL WRNDIE(-3, &
                              '<ANAREA>','Maximum arc limit exceeded')
                         L=LT(N)
                         IDER(L)=IDER(L)+1
                         SIGN_YDER(L) = SIGN_YDER(L) + 1
                         KENT(JB_LOCAL)=L*1024+K
                         L=LT(M)
                         IDER(L)=IDER(L)+1
                         SIGN_YDER(L) = SIGN_YDER(L) - 1
                         KOUT(JB_LOCAL)=K*1024+L
                      ENDIF
                      TT=ARCF(M)
                      if (TT.ge.T) then
                         T=TT
                         N=M
                      ENDIF
                   ENDDO
                ENDIF
                ARCSUM=ARCSUM+PIX2-T
                if (.not.LTOP) then
                   EXANG=EXANG+EX(N)
                   JB_LOCAL=JB_LOCAL+1
                   L=LT(N)
                   IDER(L)=IDER(L)+1
                   SIGN_YDER(L) = SIGN_YDER(L) + 1
                   KENT(JB_LOCAL)=L*1024+K
                   L=LT(MI)
                   IDER(L)=IDER(L)+1
                   SIGN_YDER(L) = SIGN_YDER(L) - 1
                   KOUT(JB_LOCAL)=K*1024+L
                ENDIF
                !
19              ARCLEN=ARCLEN+GR(K)*ARCSUM
                IN=INTAG(K)
                T1=ARCSUM*RRSQ*(BSQK-RRSQ+RX(IN)**2)/(RRX2*BSQK*BK)
                !
                if (QLONE) goto 56
             ENDIF
          ENDDO
          ! end of K loop
          if (ARCLEN.eq.ZERO) go to 4
          if (JB_LOCAL.eq.0) go to 56
          !** Find number of independent boundaries
          J=0
          do K=1,JB_LOCAL
             if (KOUT(K).ne.0) then
                I=K
62              N=KOUT(I)
                KOUT(I)=0
                J=J+1
                do II=1,JB_LOCAL
                   if (N.eq.KENT(II)) then
                      if (II.eq.K) then
                         IB_LOCAL=IB_LOCAL+1
                         if (J.eq.JB_LOCAL) goto 56
                         goto 60
                      ENDIF
                      I=II
                      go to 62
                   ENDIF
                ENDDO
             ENDIF
60           CONTINUE
          ENDDO
          IB_LOCAL=IB_LOCAL+1
          IF(WRNLEV.GE.2) WRITE(OUTU,*) &
               'CONNECTIVITY ERROR ON SPHERE NO. ',IR
56        AREA_TMP(IR)=IB_LOCAL*PIX2+EXANG+ARCLEN
          !CC##IF INTEL
          !CC          AREA(IR)=DINMOD(AREA(IR),PIX4)
          !CC##ELSE
          AREA_TMP(IR)=MOD(AREA_TMP(IR),PIX4)
          !CC##ENDIF
16        AREA_TMP(IR)=AREA_TMP(IR)*RRSQ
          IF(PRNLEV.GT.8) write (OUTU,'(I10,3(A,F10.5))') IR, &
               ' AREA=',AREA_TMP(IR)
       ENDIF
4      CONTINUE
    ENDDO
    !
#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM, IERR)
    call mpi_allreduce(AREA_TMP, AREA, NATOM, mpi_double_precision, &
         MPI_SUM, COMM_CHARMM, IERR)
    call mpi_barrier(COMM_CHARMM, IERR)
#else
    DO I=1,NATOM
       AREA(I)=AREA_TMP(I)
    ENDDO ! DO I=1,NATOM
#endif
    !
    IF (MYNOD .EQ. 0) THEN
       ! Sum all areas to get total area
       EE = ZERO
       DO I=1,NATOM
          IF (ISLCT(I) .EQ. 1) EE=EE+AREA(I)
       ENDDO ! DO I=1,NATOM
    ELSE
       PRNLEV=0
    ENDIF ! IF (MYNOD .EQ. 0) THEN
    !
    IF (MYNOD .EQ. 0) THEN
       IF(PRNLEV.GT.6) THEN
          WRITE(OUTU,'(A,F10.5)') 'Total surface area:  ',EE
       ENDIF
    ENDIF ! IF (MYNOD .EQ. 0) THEN
    call set_param('AREA',EE)
    !
#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM, IERR)
#endif

    call chmdealloc('surfac.src', 'ANAREA', 'AREA_TMP', NATOM, cr8=AREA_TMP)
    !
    return
  end SUBROUTINE ANAREA

#endif
  SUBROUTINE NULL_SF
    RETURN
  END SUBROUTINE NULL_SF

end module surfacmod
