#if KEY_NOGRAPHICS==1 /*nograph*/
SUBROUTINE NULL_POV
  RETURN
END SUBROUTINE NULL_POV
#else /* (nograph)*/
SUBROUTINE POVDFN(IUNIT,ISTRT,ISTOP,XYZS,IACOL, &
     ISTER,INDRW,FINDRW,NPOVRTX,TRNVRTX,XYZCS)
  !
  ! GENERATES A POVRAY DEFINITION FILE FROM THE SORTED ATOM LIST
  ! THE SORTING IS NOT NEEDED EXPLICITLY, BUT USED FOR Z-CLIPPING
  !
  !     IUNIT - FORTRAN UNIT NO. FOR POV DEFN FILE
  !     ISTRT - START POINTER INTO INDEXS
  !     ISTOP - STOP    "       "     "
  !     XYZS  - Z SORTED ATOM XYZ IN ROWS 1-3, RADII IN ROW 4
  !     IACOL - ATOM COLOR REGISTER VALUES
  !     ISTER - IMAGE TYPE FLAG; -1=STEREO LEFT, 0=MONO, 1=STEREO RIGHT
  !     INDRW - INITIAL CALL; BEGIN A NEW PICTURE
  !     FINDRW- FINAL CALL; PICTURE FINISHED
  !
  !     NPOVRTX - NO. OF POV OBJECT VERTEX 3D POINTS
  !     TRNVRTX - VERTEX ARRAY AFTER GRAPHICS TRANSFORM
  !
  use chm_kinds
  use dimens_fcm
  use number
  use graph
  use stream
  use hbondm
  implicit none

  INTEGER IUNIT,ISTRT,ISTOP,ISTER
  INTEGER IACOL(*), NPOVRTX
  real(chm_real) TRNVRTX(4,NPOVRTX), OPHI,OTHE,OTAU,A,B
  LOGICAL INDRW,FINDRW
  real(chm_real) XYZS(4,*), X2,Y2,Z2, CMN(3),CMX(3)
  real(chm_real) RD2DG, XYZCS(4,*)

  INTEGER IRAD,CCOL,IBD,KUNIT
  INTEGER HBWIDTH,ITEMP,JTEMP
  INTEGER IER,J,I,II,IAT,JAT,IX,IY
  INTEGER IZH,IZL,ILEV,NLEV,KFONT,ICMUSED(0:191)
  CHARACTER(len=80) TMP
  CHARACTER(len=40) SC
  CHARACTER(len=26) S,T
  CHARACTER(len=9) CHMCLR
  CHARACTER(len=32) SHBTXTR(0:7)

  SAVE ICMUSED

  IF (PRNLEV.GT.5) WRITE(OUTU,34) 1+ISTOP-ISTRT
34 FORMAT(' ',I8,' atoms will be displayed')

  ! stereo mode check; iunit+1 used for right-eye image
  IF (ISTER.EQ.-1) THEN
     ! left eye image
     KUNIT = IUNIT
     WRITE (KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// LEFT EYE STEREO IMAGE'
  ELSEIF (ISTER.EQ.0) THEN
     ! monoscopic image
     KUNIT = IUNIT
     WRITE (KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// MONOSCOPIC IMAGE'
  ELSEIF (ISTER.EQ.1) THEN
     ! right eye image
     KUNIT = IUNIT + 1
     INQUIRE(KUNIT, ACCESS=TMP)
     IF (TMP(1:7).EQ.'UNKNOWN') THEN
        CALL WRNDIE(-1,'POVDFN>','STEREO ERROR, IUNIT+1 NOT OPEN')
        RETURN
     ENDIF
     WRITE (KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// RIGHT EYE STEREO IMAGE'
  ENDIF

  RD2DG = 180.e0 / ASIN(-1.e0)
  ! setup text array of Hbond textures; arbitrary subset of POV textures
  SHBTXTR(0) = 'Yellow_Glass'
  SHBTXTR(1) = 'Green_Glass'
  SHBTXTR(2) = 'pigment { Sapphire_Agate }'
  SHBTXTR(3) = 'pigment { Red_Marble }'
  SHBTXTR(4) = 'pigment { DMFLightOak }'
  SHBTXTR(5) = 'Chrome_Metal'
  SHBTXTR(6) = 'New_Brass'
  SHBTXTR(7) = 'pigment { Candy_Cane }'

  ! traverse atom and label color arrays and declare the colors used
  IF (INDRW.OR.QERASE) THEN
     WRITE(KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// COLOR DECLARATIONS; ATOMS, BONDS, ...'
     WRITE(KUNIT,'(A)') '#declare RV_Shiny = finish { '
     WRITE(KUNIT,'(A)') '  ambient 0.2 diffuse 0.7'
     WRITE(KUNIT,'(A)') '  specular 0.8 roughness 0.001 }'
     WRITE(KUNIT,'(A)') '#declare RV_Dull = finish { '
     WRITE(KUNIT,'(A)') '  ambient 0.2 diffuse 0.6 }'
     DO J=0,15
        ICMUSED(J) = 1
     ENDDO
     DO J=16,191
        ICMUSED(J) = 0
     ENDDO
     DO II=ISTRT,ISTOP
        J = IACOL(II)
        ICMUSED(J) = 1
        IAT=INDEXP(INDEXS(II))
        J=IGRLBL(IAT)/256
        ICMUSED(J) = 1
     ENDDO
     TMP = '// ALT FINISHES: Shiny Mirror Metal Glossy Dull'
     WRITE(KUNIT,'(A)') TMP
     DO J=0,191
        IF (0.NE.MOD(J,16)) THEN
           IF (ICMUSED(J).EQ.1) THEN
              CALL SPOVCLR(J,COLOR_MAP,SC,I)
              IF (I.LT.0) CALL WRNDIE(0,'POVDFN>','colormap error')
              TMP='#declare CHMtxr    = texture { pigment {'//SC
              WRITE(TMP(16:18),'(I3.3)') J
              WRITE(KUNIT,'(A)') TMP
              I = MOD(J,16)
              IF ((I.EQ.2).OR.(I.EQ.5).OR.(I.EQ.9)) THEN
                 TMP=' } finish { RV_Shiny }}'
              ELSE
                 TMP=' } finish { RV_Dull }}'
              ENDIF
              WRITE(KUNIT,'(A)') TMP
           ENDIF
        ENDIF
     ENDDO
  ELSE
     ! additional pass, e.g. for COMP atom display
     DO II=ISTRT,ISTOP
        J = IACOL(II)
        IF (ICMUSED(J).NE.1) ICMUSED(J) = 2
     ENDDO
     DO J=0,191
        IF (0.NE.MOD(J,16)) THEN
           IF (ICMUSED(J).EQ.2) THEN
              CALL SPOVCLR(J,COLOR_MAP,SC,I)
              IF (I.LT.0) CALL WRNDIE(0,'POVDFN>','colormap error')
              TMP='#declare CHMtxr    = texture { pigment {'//SC
              WRITE(TMP(16:18),'(I3.3)') J
              WRITE(KUNIT,'(A)') TMP
              I = MOD(J,16)
              IF ((I.EQ.2).OR.(I.EQ.5).OR.(I.EQ.9)) THEN
                 TMP=' } finish { RV_Shiny }}'
              ELSE
                 TMP=' } finish { RV_Dull }}'
              ENDIF
              WRITE(KUNIT,'(A)') TMP
           ENDIF
        ENDIF
     ENDDO
  ENDIF

  ! main drawing section

  ! first include any high-level objects
  IF (INDRW.AND.IPOVOBJ.GT.0) THEN
     WRITE (KUNIT,'(A)') ''
     WRITE (KUNIT,'(A)') '// BEGIN INCLUDED OBJECT SECTION'
     ! process the vertex array using the control array as a guide
     ! declare coord identifiers using graphics transformed coords
     WRITE (KUNIT,'(A)') ''
     IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
        CALL WRNDIE(0,'POVDFN>','Vertex limit exceeded')
        WRITE (KUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECTS'
        GOTO 125
     ENDIF
     WRITE (KUNIT,'(A)') '// OBJECT COORDINATE DECLARATIONS'
     DO IX = 1,NPVOBJ
        IY = KPVOBJ(1,IX)
        II = KPVOBJ(2,IX)
        J  = KPVOBJ(3,IX)
        ! process by object type code, 1..5
        IF (IY.EQ.1) THEN
           ! sphere object
           X2 = TRNVRTX(1,II+1)
           Y2 = TRNVRTX(2,II+1)
           Z2 = TRNVRTX(3,II+1)
           CALL POVVEC(X2,Y2,Z2,S)
           WRITE(KUNIT,601) IX,S
601        FORMAT('#declare Vec_Sphr',I3.3,' = ',A)
        ELSE IF (IY.EQ.2) THEN
           ! cylinder object
           X2 = TRNVRTX(1,II+1)
           Y2 = TRNVRTX(2,II+1)
           Z2 = TRNVRTX(3,II+1)
           CALL POVVEC(X2,Y2,Z2,S)
           WRITE(KUNIT,602) IX,S
602        FORMAT('#declare Vec_Cyln',I3.3,' = ',A)
           X2 = TRNVRTX(1,II+2)
           Y2 = TRNVRTX(2,II+2)
           Z2 = TRNVRTX(3,II+2)
           A=SQRT( X2**2 + Y2**2 + Z2**2 )
           OPHI  =  RD2DG*ACOS(X2/A)
           WRITE(KUNIT,603) IX,OPHI
603        FORMAT('#declare Phi_Cyln',I3.3,' = ',F8.2)
           IF (Y2.NE.ZERO) THEN
              OTHE = RD2DG*ATAN(Z2/Y2)
           ELSE
              OTHE = SIGN(ONE, Z2) * 90.e0
           ENDIF
           WRITE(KUNIT,604) IX,OTHE
604        FORMAT('#declare The_Cyln',I3.3,' = ',F8.2)
        ELSE IF (IY.EQ.3) THEN
           ! slab object
           X2 = TRNVRTX(1,II+1)
           Y2 = TRNVRTX(2,II+1)
           Z2 = TRNVRTX(3,II+1)
           CALL POVVEC(X2,Y2,Z2,S)
           WRITE(KUNIT,605) IX,S
605        FORMAT('#declare Vec_Slab',I3.3,' = ',A)
           X2 = TRNVRTX(1,II+2)
           Y2 = TRNVRTX(2,II+2)
           Z2 = TRNVRTX(3,II+2)
           A=SQRT( X2**2 + Y2**2 + Z2**2 )
           OPHI  =  RD2DG*ACOS(X2/A)
           WRITE(KUNIT,606) IX,OPHI
606        FORMAT('#declare Phi_Slab',I3.3,' = ',F8.2)
           IF (Y2.NE.ZERO) THEN
              OTHE = RD2DG*ATAN(Z2/Y2)
           ELSE
              OTHE = SIGN(ONE, Z2) * 90.e0
           ENDIF
           WRITE(KUNIT,607) IX,OTHE
607        FORMAT('#declare The_Slab',I3.3,' = ',F8.2)
           X2 = TRNVRTX(1,II+3)
           Y2 = TRNVRTX(2,II+3)
           Z2 = TRNVRTX(3,II+3)
           A=SQRT( X2**2 + Y2**2 + Z2**2 )
           IF (Y2.NE.ZERO) THEN
              OTHE = RD2DG*ATAN(Z2/Y2)
           ELSE
              OTHE = SIGN(ONE, Z2) * 90.e0
           ENDIF
           WRITE(KUNIT,608) IX,OTHE
608        FORMAT('#declare Tau_Slab',I3.3,' = ',F8.2)
        ELSE IF (IY.EQ.4) THEN
           ! mesh object, triangle; declare atom coords
           DO I=II+1,II+J
              CALL POVVEC(TRNVRTX(1,I), TRNVRTX(2,I), &
                   TRNVRTX(3,I), S)
              WRITE(KUNIT,609) I-II,IX,S
609           FORMAT('#declare Atm',I3.3,'_Mesh',I3.3,' = ',A)
           ENDDO
        ELSE IF (IY.EQ.5) THEN
           ! ribbon object, smoothed triangle mesh; atom coords and normal
           DO I=II+1,II+J,2
              CALL POVVEC(TRNVRTX(1,I), TRNVRTX(2,I), &
                   TRNVRTX(3,I), S)
              WRITE(KUNIT,610) I-II,IX,S
610           FORMAT('#declare Atm',I3.3,'_Ribn',I3.3,' = ',A)
              CALL POVVEC(TRNVRTX(1,I+1), TRNVRTX(2,I+1), &
                   TRNVRTX(3,I+1), S)
              WRITE(KUNIT,611) 1+(I-II),IX,S
611           FORMAT('#declare Nrm',I3.3,'_Ribn',I3.3,' = ',A)
           ENDDO
        ENDIF
     ENDDO
     ! done with vertex array; now read the object and texture defns
     WRITE (KUNIT,'(A)') ''
     WRITE (KUNIT,'(A)') '// OBJECT DEFINITIONS AND TEXTURES'
25   READ(IPOVOBJ,'(A)',END=30) TMP
     WRITE (KUNIT,'(A)') TMP
     GOTO 25
30   CONTINUE
     WRITE (IUNIT,'(A)') '// END INCLUDED OBJECT SECTION'
  ENDIF

  ! bad object list skip point
125 CONTINUE

  ! show the axes; a nice shiny gradient
  IF (QDAXE) THEN

     ! x axis
     WRITE(KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// BEGIN CARTESIAN AXES'
     CALL POVVEC(AXETRN(1,1), AXETRN(2,1), AXETRN(3,1),S)
     CALL POVVEC(AXETRN(1,2), AXETRN(2,2), AXETRN(3,2),T)
     WRITE (KUNIT,811) S,T, TWO*AXETRN(1,1)
811  FORMAT('cylinder { ',A,',',A,', 0.35',/, &
          ' pigment { gradient x color_map { [0 color Yellow] ', &
          '[1 color Cyan] }',/,' translate -0.5*x scale', &
          F7.2,' } finish { Metal } }')
     ! label positive x axis
     WRITE (KUNIT,813) AXETRN(1,1) + 0.5
813  FORMAT('text { ttf "timrom.ttf","X",1,0 texture { pigment ',/, &
          '{ color Cyan } finish { Metal }}', &
          /,'scale 3 translate ',F7.2,'*x }')

     ! y axis
     CALL POVVEC(AXETRN(1,3), AXETRN(2,3), AXETRN(3,3),S)
     CALL POVVEC(AXETRN(1,4), AXETRN(2,4), AXETRN(3,4),T)
     WRITE (KUNIT,814) S,T, TWO*AXETRN(2,3)
814  FORMAT('cylinder { ',A,',',A,', 0.35',/, &
          ' pigment { gradient y color_map { [0 color Yellow] ', &
          '[1 color Cyan] }',/,' translate -0.5*y scale', &
          F7.2,' } finish { Metal } }')
     ! label positive y axis
     WRITE (KUNIT,816) AXETRN(2,3) + 0.5
816  FORMAT('text { ttf "timrom.ttf","Y",1,0 texture { pigment ',/, &
          '{ color Cyan } finish { Metal }}', &
          /,'scale 3 translate ',F7.2,'*y }')

     ! z axis; colors inverted for mock RH coord system
     CALL POVVEC(AXETRN(1,5), AXETRN(2,5), AXETRN(3,5),S)
     CALL POVVEC(AXETRN(1,6), AXETRN(2,6), AXETRN(3,6),T)
     WRITE (KUNIT,817) S,T, TWO*AXETRN(3,5)
817  FORMAT('cylinder { ',A,',',A,', 0.35',/, &
          ' pigment { gradient z color_map { [0 color Cyan] ', &
          '[1 color Yellow] }',/,' translate -0.5*z scale', &
          F7.2,' } finish { Metal } }')
     ! label positive z axis
     WRITE (KUNIT,819) AXETRN(3,5) + 0.5
819  FORMAT('text { ttf "timrom.ttf","Z",1,0 texture { pigment ',/, &
          '{ color Cyan } finish { Metal }}', &
          /,'scale 3 translate ',F7.2,'*z }')

  ENDIF

  ! display hydrogen bonds
  IF (QDHBON .AND. NHB.GT.0) THEN

     WRITE(KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// BEGIN HYDROGEN BONDS'
     ! map the H-bond dashlength to the POV-Ray texture
     WRITE(KUNIT,820) SHBTXTR(MOD(IGHBTY,8))
820  FORMAT('#declare HBtxtur = texture { ',A,' }')

     ! loop over the potential H-bonds; lots of bounds checking here
     DO II=1,NHB
        IAT=KHB(II)
        JAT=JHB(II)
        IF(IAT.LE.0) IAT=IHB(II)
        IF(IAT.LE.0) GOTO 155
        IF(JAT.LE.0) GOTO 155
        IAT=INDEXR(IAT)
        JAT=INDEXR(JHB(II))
        IF(IAT.LE.0) GOTO 155
        IF(JAT.LE.0) GOTO 155
        IF(INDEXB(IAT).LT.ISTRT) GOTO 155
        IF(INDEXB(IAT).GT.ISTOP) GOTO 155
        IF(INDEXB(JAT).LT.ISTRT) GOTO 155
        IF(INDEXB(JAT).GT.ISTOP) GOTO 155

        CALL POVVEC(XYZS(1,JAT),XYZS(2,JAT),XYZS(3,JAT),S)
        CALL POVVEC(XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT),T)
        WRITE(KUNIT,821) S,T, .1+.1*IGHBWI
821     FORMAT('cylinder { ',A,',',A,', ',F7.2,/, &
             ' texture { HBtxtur } }')

155     CONTINUE
     ENDDO
  ENDIF

  ! display atoms and/or bonds
  CCOL=-1
  WRITE(KUNIT,'(A)') ''
  WRITE(KUNIT,'(A)') '// BEGIN ATOMS AND/OR BONDS'
  DO II=ISTRT,ISTOP
     IAT=INDEXS(II)
     IF (CCOL.NE.IACOL(IAT)) THEN
        CCOL=IACOL(IAT)
        CHMCLR='CHMtxr   '
        WRITE(CHMCLR(7:9),'(I3.3)') CCOL
     ENDIF
     IF (QDATOM) THEN
        IF (XYZS(4,IAT).GT.0.0) THEN
           CALL POVVEC(XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT),S)
           WRITE (KUNIT,830) S, XYZS(4,IAT), CHMCLR
830        FORMAT('sphere { ' ,A,',',F7.2,' texture { ',A,' }}')
        ENDIF
     ENDIF
     IF (QDBOND) THEN
        IBD=0
        DO J=1,NATBON(IAT)
           JAT=IATBON(J,IAT)
           X2 = HALF * (XYZS(1,IAT) + XYZS(1,JAT))
           Y2 = HALF * (XYZS(2,IAT) + XYZS(2,JAT))
           Z2 = HALF * (XYZS(3,IAT) + XYZS(3,JAT))
           CALL POVVEC(X2,Y2,Z2,S)
           CALL POVVEC(XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT),T)
           WRITE(KUNIT,831) S,T, .1+.1*IGRWIDTH,CHMCLR
831        FORMAT('cylinder { ',A,',',A,', ',/,2X,F7.2, &
                ' texture { ',A,' }}')
        ENDDO
     ENDIF
     ! display vectors from COMP if requested
     IF (QDVECT) THEN
        IF (XYZS(4,IAT).GT.0.0) THEN
           CHMCLR='CHMtxr   '
           WRITE(CHMCLR(7:9),'(I3.3)') IVECCO
           X2 = XYZS(1,IAT)-XYZCS(1,IAT)
           Y2 = XYZS(2,IAT)-XYZCS(2,IAT)
           Z2 = XYZS(3,IAT)-XYZCS(3,IAT)
           A = SQRT(X2**2 + Y2**2 + Z2**2)
           IF (QDVEHD.AND.(A.GT.0.3)) THEN
             B = 0.25
             IF(A<0.8) B = 0.5 
             X2 = XYZCS(1,IAT) + B*X2
             Y2 = XYZCS(2,IAT) + B*Y2
             Z2 = XYZCS(3,IAT) + B*Z2
             CALL POVVEC(XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT),S)
             CALL POVVEC(X2,Y2,Z2,T)
             WRITE (KUNIT,831) S, T, 0.075*IVECWI, CHMCLR
             CALL POVVEC(XYZCS(1,IAT),XYZCS(2,IAT),XYZCS(3,IAT),S)
             WRITE(KUNIT,834) S, T, 0.1*IVECWI, CHMCLR
834          FORMAT('cone { ',A,', 0.0 ' ,A,', ',F7.2, &
                ' texture { ',A,' }}')
           ELSE
             CALL POVVEC(XYZS(1,IAT),XYZS(2,IAT),XYZS(3,IAT),S)
             CALL POVVEC(XYZCS(1,IAT),XYZCS(2,IAT),XYZCS(3,IAT),T)
             WRITE (KUNIT,831) S, T, 0.075*IVECWI, CHMCLR
           ENDIF
           CHMCLR='CHMtxr   '
           WRITE(CHMCLR(7:9),'(I3.3)') CCOL
        ENDIF
     ENDIF
  ENDDO


  !  display labels if enabled

  IF (QLABEL) THEN
     KFONT=0
     WRITE(KUNIT,'(A)') ''
     WRITE(KUNIT,'(A)') '// BEGIN ATOM-BASED LABELING'
     DO II=ISTRT,ISTOP
        IAT=INDEXP(INDEXS(II))
        IF (IGRLBL(IAT).NE.0) THEN
           IF (IGRLBL(IAT).NE.KFONT) THEN
              KFONT=IGRLBL(IAT)
              ! color map index, 0 thru 191
              I=IGRLBL(IAT)/256
              CHMCLR='CHMtxr   '
              WRITE(CHMCLR(7:9),'(I3.3)') I
              ! font size, 1 thru 4
              IX=MOD(IGRLBL(IAT),256)
           ENDIF
           JAT=INDEXS(II)
           X2 = XYZS(1,JAT) + HALF
           Y2 = XYZS(2,JAT) + HALF
           Z2 = XYZS(3,JAT) - HALF
           CALL POVVEC(X2,Y2,Z2,S)
           WRITE (KUNIT,832) TGRLBL(IAT)(1:IGRLLN(IAT)), &
                CHMCLR, (0.32*IX)/GRSCAL, S
        ENDIF
     ENDDO
  ENDIF
832 FORMAT('text { ttf "timrom.ttf","',A,'",1,0 texture { ', &
       A,' } scale ',F7.2,' translate ',A,' }')

  !
  ! final stuff before finishing file
  !
  IF (FINDRW.AND.QERASE) THEN
     IF (QTITLE) THEN
        ! determine coordinate limits
        DO J = 1,3
           CMN(J) = ZERO
           !c            CMX(J) = ZERO
        ENDDO
        DO II=ISTRT,ISTOP
           DO J = 1,3
              CMN(J) = MIN(CMN(J), XYZS(J,II))
              !c              CMX(J) = MAX(CMX(J), XYZS(J,II))
           ENDDO
        ENDDO
        I=5
        X2 = CMN(1) - 5.
        Y2 = CMN(2) - 5.
        Z2 = CMN(3) - 5.
        CALL POVVEC(X2,Y2,Z2,S)
        TMP = 'text { ttf "timrom.ttf",'
        WRITE (KUNIT,'(A)') ''
        WRITE (KUNIT,833) TMP,GTITLE(1:IGRTLEN), &
             (HALF*IGRFONT)/GRSCAL, S
     ENDIF
  ENDIF
833 FORMAT(A,/,'"',A,'",1,0',/,' texture { pigment { color Silver }', &
       'finish { Metal }} scale ',F7.2,/,' translate ',A,'}')
  !
  RETURN
END SUBROUTINE POVDFN


SUBROUTINE POVOBJ(IUNIT,COMLYN,COMLEN,NATOM,X,Y,Z,ISLCT, &
     NPOVRTX,POVRTX)
  !
  ! parse object options, create and place POV objects in lab coords
  ! fill the control, real data, and vertex arrays
  !
  use chm_kinds
  use dimens_fcm
  use number
  use graph
  use image
  use corman_mod
  use stream
  use string
  use helix
  use param_store, only: get_param

  implicit none

  ! arguments
  INTEGER IUNIT, COMLEN, NATOM, ISLCT(*),NSLCT, NPOVRTX
  CHARACTER(len=*) COMLYN
  real(chm_real) X(*),Y(*),Z(*), POVRTX(4,NPOVRTX)

  ! internals
  INTEGER I,J,K, NDIG, NTX, NCLR, II,JJ,KK, NPTS, IPUNIT
  real(chm_real) CRAD, CLEN, CHLF, PT1(3), PT2(3), THIC, AX(3), R0(3)
  real(chm_real) AXISXL,AXISYL,AXISZL, AXISRL, XT, YT, ZT, PHI, XYZIGR
  real(chm_real) CMN(3), CMX(3), XC,YC,ZC, SX,SY,SZ
  REAL*4 UO(4,4),US(4,4), AM,A,B,C, UL(4,4),UR(4,4), OPHI,OTHET
  REAL*4, parameter :: RZER=0.0, RONE=1.0
  CHARACTER(len=20) OFN
  CHARACTER(len=26) S1,S2,S3,S4,S5,S6,S7
  CHARACTER(len=35) CLRNAM,CPOV(3)
  CHARACTER(len=90) TXR(5)
  LOGICAL QDBG,QSCL

  QDBG = PRNLEV.GT.4

  NSLCT=0
  DO I=1,NATOM
     NSLCT=NSLCT+ISLCT(I)
  ENDDO

  IF (NPVOBJ.EQ.MAXPVO) THEN
     CALL WRNDIE(0,'POVOBJ>','Object limit reached, nothing done')
     RETURN
  ENDIF

  ! find and remove any texture options; finishes first
  OFN = ' finish { RV_Shiny }'
  IF (INDXA(COMLYN,COMLEN,'SHIN').GT.0) THEN
     OFN=' finish { RV_Shiny }'
  ELSEIF (INDXA(COMLYN,COMLEN,'DULL').GT.0) THEN
     OFN=' finish { RV_Dull }'
  ELSEIF (INDXA(COMLYN,COMLEN,'META').GT.0) THEN
     OFN=' finish { Metal }'
  ELSEIF (INDXA(COMLYN,COMLEN,'GLOS').GT.0) THEN
     OFN=' finish { Glossy }'
  ELSEIF (INDXA(COMLYN,COMLEN,'MIRR').GT.0) THEN
     OFN=' finish { Mirror }'
  ELSEIF (INDXA(COMLYN,COMLEN,'LUMI').GT.0) THEN
     OFN=' finish { Luminous }'
  ENDIF

  ! now find and remove any colors
  NCLR = 0
21 I=INDX(COMLYN,COMLEN,'COLO',4)
  IF (I.LE.0) GOTO 50
  NCLR = NCLR+1
  IF (NCLR.GT.3) GOTO 50
  CALL GTRMWA(COMLYN,COMLEN,'COLO',4,CLRNAM,35,J)
  ! leading quote means POV color name
  IF (CLRNAM(1:1).EQ.'"') THEN
     IF (CLRNAM(J:J).EQ.'"') CLRNAM(J:J) = ' '
     CPOV(NCLR)='color '//CLRNAM(2:)
  ELSE
     ! must be a charmm graphics color
     I=INDX(COMLYN,COMLEN,'COLO',4)
     J=INDX(COMLYN,COMLEN,'LEV',3)
     IF (J.GT.0.AND.(J.LT.I.OR.I.EQ.0)) THEN
        II=GTRMI(COMLYN,COMLEN,'LEV',3)
     ELSE
        II=0
     ENDIF
     CALL GETCOLOR(J,CLRNAM(1:4),7)
     CALL SPOVCLR(J+16*II,COLOR_MAP,CPOV(NCLR),I)
  ENDIF
  GOTO 21
50 CONTINUE

  ! now look for texture keywords, and build the POV statements
  NTX=1
  QSCL = .FALSE.
  TXR(1)=' texture { Gold_Metal }'
  IF (INDXA(COMLYN,COMLEN,'GLAS').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { Yellow_Glass }'
     ELSE
        TXR(1)= &
             ' texture { NBglass pigment { <0,0,0,.9> +'//CPOV(1)//'}'
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'GRAD').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { pigment { gradient x color_map {'
        TXR(2)='  [ 0 color Red ] [ 1 color Blue ] } }'
        TXR(3)='  translate -0.5*x '//OFN
        NTX=3
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.2) THEN
        TXR(1)=' texture { pigment { gradient x color_map {'
        TXR(2)='  [ 0 '//CPOV(1)//' ]'
        TXR(3)='  [ 1 '//CPOV(2)//' ] } }'
        TXR(4)='  translate -0.5*x '//OFN
        NTX=4
        QSCL=.TRUE.
     ELSE
        CALL WRNDIE(0,'POVOBJ>','0 or 2 colors for GRADient')
        GOTO 750
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'STRI').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { pigment { gradient x+y color_map {'
        TXR(2)='  [ 0.00, 0.25 color Red color Red]'
        TXR(3)='  [ 0.25, 0.75 color White color White]'
        TXR(4)='  [ 0.75, 1.00 color Red color Red] } }'//OFN
        NTX=4
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.2) THEN
        TXR(1)=' texture { pigment { gradient x+y color_map {'
        TXR(2)='  [.00, .25 '//CPOV(1)//' '//CPOV(1)//']'
        TXR(3)='  [.25, .75 '//CPOV(2)//' '//CPOV(2)//']'
        TXR(4)='  [.75, 1.0 '//CPOV(1)//' '//CPOV(1)//']}}'//OFN
        NTX=4
        QSCL=.TRUE.
     ELSE
        CALL WRNDIE(0,'POVOBJ>','0 or 2 colors for STRIpes')
        GOTO 750
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'CHEC').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { checker Green, White '//OFN
        NTX=1
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.2) THEN
        TXR(1)=' texture { checker '//CPOV(1)//','
        TXR(2)='  '//CPOV(2)//' '//OFN
        NTX=2
        QSCL=.TRUE.
     ELSE
        CALL WRNDIE(0,'POVOBJ>','0 or 2 colors for CHECker')
        GOTO 750
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'HEXA').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { hexagon Green White Red '//OFN
        NTX=1
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.2) THEN
        TXR(1)=' texture { hexagon '//CPOV(1)//','
        TXR(2)='  '//CPOV(2)//','
        TXR(3)='  '//CPOV(3)//OFN
        NTX=3
        QSCL=.TRUE.
     ELSE
        CALL WRNDIE(0,'POVOBJ>','0 or 3 colors for HEXAgon')
        GOTO 750
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'MARB').GT.0) THEN
     IF (NCLR.EQ.0) THEN
        TXR(1)=' texture { pigment { marble turbulence 1 color_map {'
        TXR(2)='  [0.0, 0.8 color Gray90 color Gray50]'
        TXR(3)='  [0.8, 1.0 color Gray50 color Gray20]}}'//OFN
        NTX=3
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.1) THEN
        TXR(1)=' texture { pigment { marble turbulence 1 color_map {'
        TXR(2)='  [0.0, 0.8 color Gray90 color Gray50]'
        TXR(3)='  [0.8, 1.0 color Gray50 '//CPOV(1)//']'
        TXR(4)='  } } '//OFN
        NTX=4
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.2) THEN
        TXR(1)=' texture { pigment { marble turbulence 1 color_map {'
        TXR(2)='  [0.0, 0.8 color Gray90 color Gray50]'
        TXR(3)='  [0.8, 0.9 color Gray50 '//CPOV(1)//']'
        TXR(4)='  [0.9, 1.0 '//CPOV(1)
        TXR(5)='  '//CPOV(2)//']}}'//OFN
        NTX=5
        QSCL=.TRUE.
     ELSEIF (NCLR.EQ.3) THEN
        TXR(1)=' texture { pigment { marble turbulence 1 color_map {'
        TXR(2)='  [0.0, 0.8 '//CPOV(2)//' '//CPOV(3)//']'
        TXR(3)='  [0.8, 0.9 '//CPOV(3)//' '//CPOV(1)//']'
        TXR(4)='  [0.9, 1.0 '//CPOV(1)//' '//CPOV(2)//']'
        TXR(5)='  } } '//OFN
        NTX=5
        QSCL=.TRUE.
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'GRAN').GT.0) THEN
     TXR(1)=' texture { pigment { PinkGranite }'//OFN
     NTX=1
     QSCL=.TRUE.
  ELSEIF (INDXA(COMLYN,COMLEN,'WOOD').GT.0) THEN
     IF (INDXA(COMLYN,COMLEN,'CHER').GT.0) THEN
        TXR(1) = ' texture { pigment { Cherry_Wood }'//OFN
     ELSEIF (INDXA(COMLYN,COMLEN,'PINE').GT.0) THEN
        TXR(1) = ' texture { pigment { Pine_Wood }'//OFN
     ELSEIF (INDXA(COMLYN,COMLEN,'SPRU').GT.0) THEN
        TXR(1) = ' texture { pigment { DMFLightOak }'//OFN
     ELSEIF (INDXA(COMLYN,COMLEN,'OAK').GT.0) THEN
        TXR(1) = ' texture { pigment { DMFDarkOak }'//OFN
     ELSEIF (INDXA(COMLYN,COMLEN,'ROSE').GT.0) THEN
        TXR(1) = ' texture { Rosewood '//OFN
     ELSEIF (INDXA(COMLYN,COMLEN,'MAPL').GT.0) THEN
        TXR(1) = ' texture { Sandalwood '//OFN
     ELSE
        TXR(1) = ' texture { pigment { Tan_Wood }'//OFN
     ENDIF
     NTX=1
     QSCL=.TRUE.
  ELSEIF (INDXA(COMLYN,COMLEN,'AGAT').GT.0) THEN
     TXR(1)=' texture { pigment { Sapphire_Agate }'//OFN
     NTX=1
     QSCL=.TRUE.
  ELSEIF (INDXA(COMLYN,COMLEN,'JADE').GT.0) THEN
     TXR(1)=' texture { pigment { Jade }'//OFN
     NTX=1
     QSCL=.TRUE.
  ELSEIF (INDXA(COMLYN,COMLEN,'ALUM').GT.0) THEN
     TXR(1)=' texture { Aluminum }'
     NTX=1
  ELSEIF (INDXA(COMLYN,COMLEN,'BRAS').GT.0) THEN
     TXR(1)=' texture { New_Brass }'
     NTX=1
  ELSEIF (INDXA(COMLYN,COMLEN,'CHRO').GT.0) THEN
     TXR(1)=' texture { Polished_Chrome }'
     NTX=1
  ELSEIF (INDXA(COMLYN,COMLEN,'COPP').GT.0) THEN
     TXR(1)=' texture { Copper_Metal }'
     NTX=1
  ENDIF

  ! done with textures, on to geometric shapes; cylinder options first
  IF (INDXA(COMLYN,COMLEN,'CYLI').GT.0) THEN
     CRAD = GTRMF(COMLYN,COMLEN,'RAD',MINONE)
     CLEN = GTRMF(COMLYN,COMLEN,'LEN',MINONE)

     ! user specified cylinder endpoints
     IF (INDXA(COMLYN,COMLEN,'PTS').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> cylinder endpoints'
        PT1(1) = NEXTF(COMLYN,COMLEN)
        PT1(2) = NEXTF(COMLYN,COMLEN)
        PT1(3) = NEXTF(COMLYN,COMLEN)
        PT2(1) = NEXTF(COMLYN,COMLEN)
        PT2(2) = NEXTF(COMLYN,COMLEN)
        PT2(3) = NEXTF(COMLYN,COMLEN)
        IF (CRAD.LT.ZERO) CRAD = TWO
        XC = HALF*(PT1(1)+PT2(1))
        YC = HALF*(PT1(2)+PT2(2))
        ZC = HALF*(PT1(3)+PT2(3))
        AXISXL = PT2(1) - PT1(1)
        AXISYL = PT2(2) - PT1(2)
        AXISZL = PT2(3) - PT1(3)
        AXISRL = SQRT(AXISXL*AXISXL+AXISYL*AXISYL+AXISZL*AXISZL)
        IF (CLEN.LT.ZERO) CLEN = AXISRL*TWO
        AXISXL = AXISXL / AXISRL
        AXISYL = AXISYL / AXISRL
        AXISZL = AXISZL / AXISRL

        ! use the last COOR AXIS command to place the cylinder
     ELSEIF (INDXA(COMLYN,COMLEN,'AXIS').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> cylinder COOR AXIS'
        call get_param('XAXI', AXISXL)
        call get_param('YAXI', AXISYL)
        call get_param('ZAXI', AXISZL)
        call get_param('RAXI', AXISRL)
        call get_param('XCEN', XC)
        call get_param('YCEN', YC)
        call get_param('ZCEN', ZC)
        IF (CRAD.LT.ZERO) CRAD = TWO
        IF (CLEN.LT.ZERO) CLEN = AXISRL*TWO

        ! use the axis determined via the Aqvist algorithm
     ELSEIF (INDXA(COMLYN,COMLEN,'HELI').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> cylinder helix'
        CALL HLXSP1(COMLYN,COMLEN,NATOM,ISLCT)
        IF(NHLXAT <= 0) THEN
           IF(WRNLEV.GE.3) WRITE(OUTU,'(A)') &
                '<POVOBJ> No atoms selected to define helix'
        ELSE
           CALL PAXATM(X,Y,Z,ISLCT,NATOM, AXISXL,AXISYL,AXISZL,AXISRL, &
                XC,YC,ZC, SX,SY,SZ, CMN,CMX, QDBG)
           DO I=1,3
              AX(I)=ZERO
              R0(I)=ZERO
           ENDDO
           CALL HELIX1(AX,R0,NDIG)
           IF(NDIG .GT. 0) THEN
              IF(PRNLEV.GE.3) WRITE(OUTU,310) AX,R0,NDIG
310           FORMAT(/'     HELIX AXIS:',3F10.6/ &
                   '     Perp. vector to origin:',3F10.6/ &
                   '     Estimated number of significant digits:',I4/)
           ELSE
              IF(WRNLEV.GE.3) WRITE(OUTU,320) -NDIG,AX,R0
320           FORMAT(/'     Something went wrong....IER=',I5,'???'/ &
                   '     HELIX AXIS (as found):',3F10.6/ &
                   '     Perp. vector to origin:',3F10.6/)
           ENDIF
           AXISXL = AX(1)
           AXISYL = AX(2)
           AXISZL = AX(3)
           XC = R0(1)
           YC = R0(2)
           ZC = R0(3)
           IF (CRAD.LT.ZERO) CRAD = TWO
           IF (CLEN.LT.ZERO) CLEN = AXISRL*TWO
        ENDIF

        ! use the principal axes of the selected atoms
     ELSEIF (INDXA(COMLYN,COMLEN,'PAX').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> cylinder principal axes'
        CALL PAXATM(X,Y,Z,ISLCT,NATOM, AXISXL,AXISYL,AXISZL, AXISRL, &
             XC,YC,ZC, SX,SY,SZ, CMN,CMX, QDBG)
        IF (CRAD.LT.ZERO) THEN
           CRAD = HALF*HALF*((CMX(2)-CMN(2))+(CMX(3)-CMN(3)))
        ENDIF
        IF (CLEN.LT.ZERO) THEN
           CLEN = CMX(1) - CMN(1)
        ENDIF
     ELSE
        CALL WRNDIE(0,'POVOBJ>','invalid POV cylinder spec')
        GOTO 750
     ENDIF

     ! done parsing cylinder options; store data and write defn to file
     NPVOBJ = NPVOBJ+1
     IF (NPVOBJ.GT.MAXPVO) THEN
        CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
        WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! object type
     KPVOBJ(1,NPVOBJ) = 2
     ! index offset into vertex array
     IF (NPVOBJ.EQ.1) THEN
        K = 0
     ELSE
        I = NPVOBJ - 1
        K = KPVOBJ(2,I) + KPVOBJ(3,I)
     ENDIF
     KPVOBJ(2,NPVOBJ) = K
     ! number of vertices for object 
     KPVOBJ(3,NPVOBJ) = 2
     IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
        CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
        WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! set vertex data; center, and orientation vector
     POVRTX(1,K+1) = XC
     POVRTX(2,K+1) = YC
     POVRTX(3,K+1) = ZC
     POVRTX(4,K+1) = RONE
     POVRTX(1,K+2) = AXISXL
     POVRTX(2,K+2) = AXISYL
     POVRTX(3,K+2) = AXISZL
     POVRTX(4,K+2) = RONE

     ! write the cylinder, mono or left-eye stereo view
     ! n.b. the povvec routine requires real*4 args

     A = CLEN
     B = CRAD
     CALL POVVEC(RZER,  A, RZER, S1)
     CALL POVVEC(RZER, -A, RZER, S2)
     WRITE(IUNIT,901) S1,S2,CRAD
     DO I=1,NTX
        WRITE(IUNIT,'(A)') TXR(I)
     ENDDO
     ! scale the texture to the cylinder size
     IF (QSCL) THEN
        CALL POVVEC(A,B,B,S5)
        WRITE(IUNIT,921) S5
     ENDIF
     S3 = 'Phi_Cyln   '
     WRITE(S3(9:11),'(I3.3)') NPVOBJ
     S4 = 'The_Cyln   '
     WRITE(S4(9:11),'(I3.3)') NPVOBJ
     S6 = 'Vec_Cyln   '
     WRITE(S6(9:11),'(I3.3)') NPVOBJ
     WRITE(IUNIT,931) S3,S4,S6
901  FORMAT ('cylinder { ',A,',',A,2X,F7.2)
921  FORMAT (' scale ',A,' }')
931  FORMAT (' rotate ',A,' rotate ',A,' translate ',A,' }')

     ! slab options
  ELSEIF (INDXA(COMLYN,COMLEN,'SLAB').GT.0) THEN
     THIC = GTRMF(COMLYN,COMLEN,'THIC',MINONE)
     ! based on min/max, no rotation; e.g. glass orthorhombic unit cell
     IF (INDXA(COMLYN,COMLEN,'PTS').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> box corners'
        CMN(1) = NEXTF(COMLYN,COMLEN)
        CMN(2) = NEXTF(COMLYN,COMLEN)
        CMN(3) = NEXTF(COMLYN,COMLEN)
        CMX(1) = NEXTF(COMLYN,COMLEN)
        CMX(2) = NEXTF(COMLYN,COMLEN)
        CMX(3) = NEXTF(COMLYN,COMLEN)
        AXISXL = ONE
        AXISYL = ZERO
        AXISZL = ZERO
        SX = ZERO
        SY = ONE
        SZ = ZERO
        XC = 0.5*(CMN(1)+CMX(1))
        YC = 0.5*(CMN(2)+CMX(2))
        ZC = 0.5*(CMN(3)+CMX(3))

        ! slab aligned to atom coord distrib, using principal axes
     ELSEIF (INDXA(COMLYN,COMLEN,'PAX').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> slab principal axes'
        CALL PAXATM(X,Y,Z,ISLCT,NATOM, AXISXL,AXISYL,AXISZL,AXISRL, &
             XC,YC,ZC, SX,SY,SZ, CMN,CMX, QDBG)

     ELSE
        CALL WRNDIE(0,'POVOBJ>','invalid POV SLAB spec')
        GOTO 750
     ENDIF

     ! done parsing slab options; store data and write defn to file
     NPVOBJ = NPVOBJ+1
     IF (NPVOBJ.GT.MAXPVO) THEN
        CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
        WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! object type
     KPVOBJ(1,NPVOBJ) = 3
     ! index offset into vertex array
     IF (NPVOBJ.EQ.1) THEN
        K = 0
     ELSE
        I = NPVOBJ - 1
        K = KPVOBJ(2,I) + KPVOBJ(3,I)
     ENDIF
     KPVOBJ(2,NPVOBJ) = K
     ! number of vertices for object 
     KPVOBJ(3,NPVOBJ) = 3
     IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
        CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
        WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! set vertex data; center, orientation vectors
     POVRTX(1,K+1) = XC
     POVRTX(2,K+1) = YC
     POVRTX(3,K+1) = ZC
     POVRTX(4,K+1) = RONE
     POVRTX(1,K+2) = AXISXL
     POVRTX(2,K+2) = AXISYL
     POVRTX(3,K+2) = AXISZL
     POVRTX(4,K+2) = RONE
     POVRTX(1,K+3) = SX
     POVRTX(2,K+3) = SY
     POVRTX(3,K+3) = SZ
     POVRTX(4,K+3) = RONE

     IF (THIC.GT.ZERO) THEN
        CMN(3) = -0.5*THIC
        CMX(3) =  0.5*THIC
     ENDIF
     A = CMN(1)
     B = CMN(2)
     C = CMN(3)
     CALL POVVEC(A,  B, C, S1)
     A = CMX(1)
     B = CMX(2)
     C = CMX(3)
     CALL POVVEC(A,  B, C, S2)
     WRITE(IUNIT,903) S1,S2
     DO I=1,NTX
        WRITE(IUNIT,'(A)') TXR(I)
     ENDDO
     IF (QSCL) THEN
        A = CMX(1)-CMN(1)
        B = CMX(2)-CMN(2)
        C = CMX(3)-CMN(3)
        CALL POVVEC(A,B,C,S5)
        WRITE(IUNIT,921) S5
     ENDIF
     S3 = 'Phi_Slab   '
     WRITE(S3(9:11),'(I3.3)') NPVOBJ
     S4 = 'The_Slab   '
     WRITE(S4(9:11),'(I3.3)') NPVOBJ
     S6 = 'Vec_Slab   '
     WRITE(S6(9:11),'(I3.3)') NPVOBJ
     S7 = 'Tau_Slab   '
     WRITE(S7(9:11),'(I3.3)') NPVOBJ
     WRITE(IUNIT,913) S7,S3,S4,S6
     IF (QSTERO)  WRITE(IUNIT,913) S7,S3,S4,S6
     IF (QSTERO) THEN
        DO I=1,NTX
        ENDDO
        IF (QSCL) THEN
        ENDIF
     ENDIF

903  FORMAT ('box { ',A,',',A)
913  FORMAT ('  rotate ',A,' rotate ',A,/, &
          '  rotate ',A,' translate ',A,' }')

     ! an added sphere at some arbitrary point in space
  ELSEIF (INDXA(COMLYN,COMLEN,'SPHE').GT.0) THEN
     IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> sphere'
     CRAD = GTRMF(COMLYN,COMLEN,'RAD',MINONE)
     IF (CRAD.LT.ZERO) CRAD = TWO
     B = GTRMF(COMLYN,COMLEN,'SCAL',CRAD)
     XC = NEXTF(COMLYN, COMLEN)
     YC = NEXTF(COMLYN, COMLEN)
     ZC = NEXTF(COMLYN, COMLEN)
     NPVOBJ = NPVOBJ+1
     IF (NPVOBJ.GT.MAXPVO) THEN
        CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
        WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! object type
     KPVOBJ(1,NPVOBJ) = 1
     ! index offset into vertex array
     IF (NPVOBJ.EQ.1) THEN
        K = 0
     ELSE
        I = NPVOBJ - 1
        K = KPVOBJ(2,I) + KPVOBJ(3,I)
     ENDIF
     KPVOBJ(2,NPVOBJ) = K
     ! number of vertices for object 
     KPVOBJ(3,NPVOBJ) = 1
     IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
        CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
        WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
        RETURN
     ENDIF
     ! set vertex data; center, orientation vectors
     POVRTX(1,K+1) = XC
     POVRTX(2,K+1) = YC
     POVRTX(3,K+1) = ZC
     POVRTX(4,K+1) = RONE
     ! write to file
     S1 = 'Vec_Sphr   '
     WRITE(S1(9:11),'(I3.3)') NPVOBJ
     WRITE (IUNIT,904) S1,CRAD
     DO I=1,NTX
        WRITE(IUNIT,'(A)') TXR(I)
     ENDDO
     IF (QSCL) THEN
        CALL POVVEC(B,B,B,S5)
        WRITE(IUNIT,921) S5
     ENDIF
     WRITE(IUNIT,'(A)') ' }'
904  FORMAT('sphere { ',A,', ',F7.2)

     ! triangle mesh
  ELSEIF (INDXA(COMLYN,COMLEN,'MESH').GT.0) THEN
     IPUNIT = GTRMI(COMLYN,COMLEN,'PUNI',-1)
     ! read an array of points defining the triangles (9 per line)
     IF (INDXA(COMLYN,COMLEN,'PTS').GT.0) THEN
        NPTS = NEXTI(COMLYN,COMLEN)
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> mesh points'
        NPVOBJ = NPVOBJ + 1
        IF (NPVOBJ.GT.MAXPVO) THEN
           CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
           WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        ! object type
        KPVOBJ(1,NPVOBJ) = 4
        ! index offset into vertex array
        IF (NPVOBJ.EQ.1) THEN
           K = 0
        ELSE
           I = NPVOBJ - 1
           K = KPVOBJ(2,I) + KPVOBJ(3,I)
        ENDIF
        KPVOBJ(2,NPVOBJ) = K
        ! number of vertices for object 
        KPVOBJ(3,NPVOBJ) = NPTS*3
        IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
           CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
           WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        IF(IPUNIT.GT.0) THEN
           KK = IPUNIT
        ELSE
           KK = ISTRM
        ENDIF
        ! write to file; first the texture
        S7 = 'Txr_Mesh   '
        WRITE(S7(9:11),'(I3.3)') NPVOBJ
        WRITE (IUNIT,971) S7
        DO I=1,NTX
           WRITE(IUNIT,'(A)') TXR(I)
        ENDDO
971     FORMAT('#declare ',A,' = ')
        WRITE(IUNIT,'(/,A)') 'mesh { '
        DO JJ=1,NPTS
           II = 3*(JJ-1)
           READ(KK,*,END=760,ERR=760) ((US(I,J),I=1,3),J=1,3)
           WRITE(IUNIT,'(A)') '  triangle { '
           DO J=1,3
              DO I=1,3
                 POVRTX(I,K+II+J) = US(I,J)
              ENDDO
           ENDDO
           WRITE(IUNIT,972) II+1,NPVOBJ,II+2,NPVOBJ,II+3,NPVOBJ
           WRITE(IUNIT,973) NPVOBJ
        ENDDO
        WRITE(IUNIT,'(A)') ' }'
972     FORMAT(4X,'Atm',I3.3,'_Mesh',I3.3,', Atm',I3.3,'_Mesh', &
             I3.3,', Atm',I3.3,'_Mesh',I3.3)
973     FORMAT(4X,'texture { Txr_Mesh',I3.3,' } }')

        ! triangle mesh constructed from selected atoms as vertices
        ! for N atoms, there will be N-2 triangles; J = 2,N-1
        ! vertices for each triangle from atoms  J-1, J, J+1
     ELSEIF (INDXA(COMLYN,COMLEN,'ATOM').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> atom-based mesh'
        IF (NSLCT.LT.3) THEN
           CALL WRNDIE(0,'POVOBJ>','MESH needs more than 3 atoms')
           GOTO 750
        ENDIF
        NPVOBJ = NPVOBJ + 1
        IF (NPVOBJ.GT.MAXPVO) THEN
           CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
           WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        ! object type
        KPVOBJ(1,NPVOBJ) = 4
        ! index offset into vertex array
        IF (NPVOBJ.EQ.1) THEN
           K = 0
        ELSE
           I = NPVOBJ - 1
           K = KPVOBJ(2,I) + KPVOBJ(3,I)
        ENDIF
        KPVOBJ(2,NPVOBJ) = K
        ! number of vertices for object; same as selected atoms
        KPVOBJ(3,NPVOBJ) = NSLCT
        IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
           CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
           WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        ! write to file; first the texture
        S7 = 'Txr_Mesh   '
        WRITE(S7(9:11),'(I3.3)') NPVOBJ
        WRITE (IUNIT,971) S7
        DO I=1,NTX
           WRITE(IUNIT,'(A)') TXR(I)
        ENDDO
        ! fill the vertex array from the selected atoms in PSF order
        KK = 0
        DO II=1,NATOM
           IF(ISLCT(II).EQ.1) THEN
              KK = KK+1
              POVRTX(1,K+KK) = X(II)
              POVRTX(2,K+KK) = X(II)
              POVRTX(3,K+KK) = X(II)
           ENDIF
        ENDDO
        ! now define the mesh of triangles from the atom coords
        WRITE(IUNIT,'(/,A)') 'mesh { '
        DO II=2,NSLCT-1
           WRITE(IUNIT,'(A)') '  triangle { '
           WRITE(IUNIT,972) II-1,NPVOBJ,II,NPVOBJ,II+1,NPVOBJ
           WRITE(IUNIT,973) NPVOBJ
        ENDDO
        WRITE(IUNIT,'(A)') ' }'

     ELSE
        CALL WRNDIE(0,'POVOBJ>','invalid POV MESH spec')
        GOTO 750
     ENDIF

     ! ribbon-like smoothed triangle mesh
  ELSEIF (INDXA(COMLYN,COMLEN,'RIBB').GT.0) THEN
     IF (INDXA(COMLYN,COMLEN,'ATOM').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> atom-based ribbon'
        IF (NSLCT.LT.3) THEN
           CALL WRNDIE(0,'POVOBJ>','RIBBon needs more than 3 atoms')
           GOTO 750
        ENDIF
        NPVOBJ = NPVOBJ + 1
        IF (NPVOBJ.GT.MAXPVO) THEN
           CALL WRNDIE(0,'POVOBJ>','Object limit exceeded')
           WRITE (IUNIT,'(A)') '// OBJECT LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        ! object type
        KPVOBJ(1,NPVOBJ) = 5
        ! index offset into vertex array
        IF (NPVOBJ.EQ.1) THEN
           K = 0
        ELSE
           I = NPVOBJ - 1
           K = KPVOBJ(2,I) + KPVOBJ(3,I)
        ENDIF
        KPVOBJ(2,NPVOBJ) = K
        ! number of vertices for object; 2x selected atoms; vertex, normal
        KPVOBJ(3,NPVOBJ) = NSLCT*2
        IF (NPOVRTX.LT.KPVOBJ(2,NPVOBJ)+KPVOBJ(3,NPVOBJ)) THEN
           CALL WRNDIE(0,'POVOBJ>','Vertex limit exceeded')
           WRITE (IUNIT,'(A)') '// VERTEX LIMIT EXCEEDED, NO OBJECT'
           RETURN
        ENDIF
        ! write to file; first the texture
        S7 = 'Txr_Mesh   '
        WRITE(S7(9:11),'(I3.3)') NPVOBJ
        WRITE (IUNIT,971) S7
        DO I=1,NTX
           WRITE(IUNIT,'(A)') TXR(I)
        ENDDO
        ! fill the vertex array from the selected atoms in PSF order
        KK = -1
        DO II=1,NATOM
           IF(ISLCT(II).EQ.1) THEN
              KK = KK+2
              POVRTX(1,K+KK) = X(II)
              POVRTX(2,K+KK) = Y(II)
              POVRTX(3,K+KK) = Z(II)
           ENDIF
        ENDDO
        ! compute the normal defined from the vector cross product
        ! vectors defined from the central atom J to J-1 and J+1
        ! first and last normal set to second and next to last, resp.
        DO KK=4,2*(NSLCT-1)
           I  = K+KK-3
           II = K+KK-1
           J  = K+KK+1
           ! compute normalized vectors
           XT=POVRTX(1,I)-POVRTX(1,II)
           YT=POVRTX(2,I)-POVRTX(2,II)
           ZT=POVRTX(3,I)-POVRTX(3,II)
           CRAD = SQRT( XT**2 + YT**2 + ZT**2 )
           XT = XT/CRAD
           YT = YT/CRAD
           ZT = ZT/CRAD
           XC=POVRTX(1,J)-POVRTX(1,II)
           YC=POVRTX(2,J)-POVRTX(2,II)
           ZC=POVRTX(3,J)-POVRTX(3,II)
           CRAD = SQRT( XC**2 + YC**2 + ZC**2 )
           XC = XC/CRAD
           YC = YC/CRAD
           ZC = ZC/CRAD
           ! now the cross product
           POVRTX(1,K+KK) = YT*ZC - YC*ZT
           POVRTX(2,K+KK) = ZT*XC - ZC*XT
           POVRTX(3,K+KK) = XT*YC - XC*YT
        ENDDO
        ! fill in the first and last normal by duplication
        POVRTX(1,K+2) = POVRTX(1,K+4)
        POVRTX(2,K+2) = POVRTX(2,K+4)
        POVRTX(3,K+2) = POVRTX(3,K+4)
        J = 2*(NSLCT-1)
        POVRTX(1,2*NSLCT) = POVRTX(1,J)
        POVRTX(2,2*NSLCT) = POVRTX(2,J)
        POVRTX(3,2*NSLCT) = POVRTX(3,J)
        ! now define the mesh of smooth triangles 
        WRITE(IUNIT,'(/,A)') 'mesh { '
        DO II=4,2*(NSLCT-1),2
           I  = II-2
           J  = II+2
           WRITE(IUNIT,'(A)') '  smooth_triangle { '
           WRITE(IUNIT,975) I-1,NPVOBJ, I,NPVOBJ
           WRITE(IUNIT,975) II-1,NPVOBJ, II,NPVOBJ
           WRITE(IUNIT,976) J-1,NPVOBJ, J,NPVOBJ
           WRITE(IUNIT,973) NPVOBJ
        ENDDO
        WRITE(IUNIT,'(A)') ' }'
975     FORMAT(4X,'Atm',I3.3,'_Ribn',I3.3,', Nrm',I3.3,'_Ribn', &
             I3.3,', ')
976     FORMAT(4X,'Atm',I3.3,'_Ribn',I3.3,', Nrm',I3.3,'_Ribn',I3.3)

     ELSE
        CALL WRNDIE(0,'POVOBJ>','invalid POV RIBBon spec')
        GOTO 750
     ENDIF

     ! unit cell from CRYSTL facility (eventually)
  ELSEIF (INDXA(COMLYN,COMLEN,'CELL').GT.0) THEN
     CONTINUE

     ! various backgrounds 
  ELSEIF (INDXA(COMLYN,COMLEN,'BACK').GT.0) THEN
     IF (INDXA(COMLYN,COMLEN,'SIMP').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> simple background'
        WRITE (IUNIT,906) CPOV(1)
906     FORMAT('background { ',A,' }')
     ELSEIF (INDXA(COMLYN,COMLEN,'PLAN').GT.0) THEN
        IF (QDBG) WRITE(OUTU,'(A)') 'POVOBJ> textured background'
        A = GTRMF(COMLYN,COMLEN,'ZPOS',HUNDRD)
        B = GTRMF(COMLYN,COMLEN,'SCAL',FIVE)
        WRITE (IUNIT,'(A,F8.2,1X)') 'plane {z,',A
        DO I=1,NTX
           WRITE(IUNIT,'(A)') TXR(I)
        ENDDO
        IF (QSCL) THEN
           WRITE (IUNIT,'(A,F8.2,A)') ' scale ',B,' } }'
        ELSE
           WRITE (IUNIT,'(A)') ' }'
        ENDIF
     ELSE
        CALL WRNDIE(0,'POVOBJ>','invalid POV BACKground spec')
        GOTO 750
     ENDIF

     ! get a user color defn in rgb format
  ELSEIF (INDXA(COMLYN,COMLEN,'RGB').GT.0) THEN
     CALL GTRMWD(COMLYN,COMLEN,'NAME',4,CLRNAM,35,J)
     IF (CLRNAM(J:J).EQ.'"') THEN
        CLRNAM(J:J) = ' '
        J = J-1
     ENDIF
     CLRNAM = CLRNAM(2:)
     DO I=1,3
        UO(I,1) = NEXTF(COMLYN, COMLEN)
     ENDDO
     CALL POVVEC(UO(1,1), UO(2,1), UO(3,1), S1)
     WRITE (IUNIT,905) CLRNAM(1:J),S1
905  FORMAT('#declare ',A,' = color rgb ',A)

     ! flame out, missing or invalid keyword
  ELSE
     CALL WRNDIE(0,'POVOBJ>','invalid POV object')
  ENDIF

750 CONTINUE
  !      CALL XTRANE(COMLYN,COMLEN)
  RETURN
760 CALL WRNDIE(0,'POVOBJ>','Error reading POV OBJ MESH PTS')
  NPVOBJ = NPVOBJ - 1
  RETURN
END SUBROUTINE POVOBJ


SUBROUTINE PAXATM(X,Y,Z,ISLCT,NATOM, AXISX,AXISY,AXISZ, AXISR, &
     XC,YC,ZC, SECX, SECY, SECZ, CMN,CMX, QDBG)
  ! calculate principal axes of selected atoms
  use chm_kinds
  use dimens_fcm
  use stream
  implicit none

  real(chm_real) X(*),Y(*),Z(*), AXISR, AXISX,AXISY,AXISZ
  real(chm_real) XC,YC,ZC, CMN(3),CMX(3), SECX, SECY, SECZ
  INTEGER NATOM, ISLCT(*)
  LOGICAL QDBG

  real(chm_real), PARAMETER :: ZERO=0.D0, ONE=1.D0
  real(chm_real) U(9), V(9), SCR(24),AMOM(6),RN(3),PHI
  real(chm_real) XN,YN,ZN,AMASST,AM,XX,XY,XZ,YY,YZ,ZZ,DET
  INTEGER I,J, IPT, NSEL
  LOGICAL OK

  ! get the center of geometry
  XC=ZERO
  YC=ZERO
  ZC=ZERO
  AMASST=ZERO
  NSEL=0
  DO I=1,NATOM
     IF(ISLCT(I).EQ.1) THEN
        XC=XC+X(I)
        YC=YC+Y(I)
        ZC=ZC+Z(I)
        NSEL=NSEL+1
        AMASST=AMASST+ONE
     ENDIF
  ENDDO
  IF (AMASST.GT.ZERO) THEN
     XC=XC/AMASST
     YC=YC/AMASST
     ZC=ZC/AMASST
  ENDIF
  IF (QDBG) THEN
     WRITE(OUTU,'(A,3F12.3)') 'PAXATM> center ', XC,YC,ZC
  ENDIF

  ! get the rotation wrt. the lab frame
  XX=ZERO
  XY=ZERO
  XZ=ZERO
  YY=ZERO
  YZ=ZERO
  ZZ=ZERO
  DO I=1,NATOM
     IF(ISLCT(I).EQ.1) THEN
        XN = X(I) - XC
        YN = Y(I) - YC
        ZN = Z(I) - ZC
        XX=XX+XN*XN
        XY=XY+XN*YN
        XZ=XZ+XN*ZN
        YY=YY+YN*YN
        YZ=YZ+YN*ZN
        ZZ=ZZ+ZN*ZN
     ENDIF
  ENDDO
  AMOM(1)=ZZ
  AMOM(2)=YZ
  AMOM(3)=XZ
  AMOM(4)=YY
  AMOM(5)=XY
  AMOM(6)=XX
  IF (QDBG) THEN
     WRITE(OUTU,'(A)') 'PAXATM> moment SSQ:'
     WRITE(OUTU,'(''XX      '', G13.6)') XX
     WRITE(OUTU,'(''XY YY   '',2G13.6)') XY, YY
     WRITE(OUTU,'(''XZ YZ ZZ'',3G13.6)') XZ, YZ, ZZ
  ENDIF
  CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),SCR(1), &
       SCR(16),SCR(19),SCR(22),0)
  AXISR = SQRT(SCR(3))
  DO I=1,3
     DET=U(I)
     U(I)=U(I+6)
     U(I+6)=DET
  ENDDO
  DO I=1,3
     IPT=(I-1)*3
     DET=U(IPT+1)
     U(IPT+1)=U(IPT+3)
     U(IPT+3)=DET
     IF(U(IPT+I).LT.ZERO) THEN
        DO J=1,3
           IPT=IPT+1
           U(IPT)=-U(IPT)
        ENDDO
     ENDIF
  ENDDO
  DET=U(1)*(U(5)*U(9)-U(6)*U(8))+U(2)*(U(6)*U(7)-U(4)*U(9))+ &
       U(3)*(U(4)*U(8)-U(5)*U(7))
  IF(DET.LT.ZERO) THEN
     U(7)=-U(7)
     U(8)=-U(8)
     U(9)=-U(9)
     DET=-DET
  ENDIF
  ! get the inverse for the back transformation
  CALL INVT33(V,U,OK)
  IF (QDBG) THEN
     WRITE(OUTU,'(A,3G13.6)') 'PAXATM> eigenvals: ',(SCR(J),J=1,3)
     WRITE(OUTU,822) (I,U(I),V(I),I=1,9)
  ENDIF
822 FORMAT(3('PAXATM>',3(I3,F8.4,F8.4)/))
  ! transform a unit vector on the +X axis (1,0,0)
  AXISX = V(1)
  AXISY = V(4)
  AXISZ = V(7)
  ! transform a unit vector on the +Y axis (0,1,0)
  SECX = V(2)
  SECY = V(5)
  SECZ = V(8)
  !      CALL FNDROT(V,RN,PHI,QDBG)
  !      AXISX = RN(1)
  !      AXISY = RN(2)
  !      AXISZ = RN(3)
  ! get the xyz extent of the atoms after alignment
  DO I = 1,3
     CMN(I) = ZERO
     CMX(I) = ZERO
  END DO
  DO I = 1,NATOM
     IF(ISLCT(I).EQ.1) THEN
        XX = X(I) - XC
        YY = Y(I) - YC
        ZZ = Z(I) - ZC
        XN=U(1)*XX+U(2)*YY+U(3)*ZZ
        YN=U(4)*XX+U(5)*YY+U(6)*ZZ
        ZN=U(7)*XX+U(8)*YY+U(9)*ZZ
        CMN(1) = MIN( XN, CMN(1) )
        CMN(2) = MIN( YN, CMN(2) )
        CMN(3) = MIN( ZN, CMN(3) )
        CMX(1) = MAX( XN, CMX(1) )
        CMX(2) = MAX( YN, CMX(2) )
        CMX(3) = MAX( ZN, CMX(3) )
     ENDIF
  END DO
  IF (QDBG) THEN
     WRITE(OUTU,'(A,3G13.6)') 'PAXATM> min ',(CMN(I),I=1,3)
     WRITE(OUTU,'(A,3G13.6)') 'PAXATM> max ',(CMX(I),I=1,3)
  ENDIF

  RETURN
END SUBROUTINE PAXATM

SUBROUTINE POVINIT(IUNIT,QINIT)
  !
  !  modify USCREN for POV-Ray device coordinates
  !  send initialization sequence to IUNIT
  !  qinit flag added for 'erase off' option; true on first call only
  !
  use chm_kinds
  use dimens_fcm
  use graph
  use stream
  use startup,only:getnam,sysid
  use machutil,only:daytim

  implicit none

  INTEGER IUNIT,IMO,IDA,IYR,IHR,IMI,ISE,I,J
  CHARACTER(len=16) CUSERNAM
  CHARACTER(len=9)  CAMPOS
  CHARACTER(len=80) TMP
  real(chm_real) A,B,C
  real(chm_real), PARAMETER :: RZER=0.0, RONE=1.0
  CHARACTER(len=32) TIMBUF,T
  CHARACTER(len=80) CSYSNAM
  CHARACTER(len=1) CSL,CSP,CCO
  LOGICAL QINIT

  !
  ! preserve initial 1 A scaling, i.e. 1 A = 1 POV unit
  GRDUPA=RONE
  DO I=1,4
     DO J=1,4
        USCREN(J,I) = RZER
     ENDDO
  ENDDO
  ! invert Z coord for left-hand POV coord frame; Angstrom scale
  USCREN(1,1)= GRDUPA
  USCREN(2,2)= GRDUPA
  USCREN(3,3)=-GRDUPA
  USCREN(4,4)= RONE
  !
  IF(IOLEV.LT.0) RETURN
  !
  IF (QINIT) THEN
     CSL = '/'
     CSP = ' '
     CCO = ':'
     WRITE (IUNIT,'(A)') '// CHARMM autogenerated POV-Ray 3.x file'
     WRITE (IUNIT,'(A)') '// Code by RM Venable, FDA Biophysics Lab'
     WRITE (IUNIT,'(A)') '// Initial implementation April 1997'
     WRITE (IUNIT,'(A)') '// rvenable@deimos.cber.nih.gov'
     WRITE (IUNIT,'(A)') '//'
     WRITE (IUNIT,'(A)') '// TITLE '//GTITLE(1:IGRTLEN)
     CALL GETNAM(CUSERNAM)
     CALL SYSID(CSYSNAM)
     WRITE (IUNIT,'(A)') '// Creator: CHARMM '//CUSERNAM//CSYSNAM
     CALL DAYTIM(IMO,IDA,IYR,IHR,IMI,ISE)
     WRITE (TIMBUF,'(I2,A,I2,A,I2,A,I2,A,I2,A,I2)') &
          IMO,CSL,IDA,CSL,IYR,CSP,IHR,CCO,IMI,CCO,ISE
     WRITE (IUNIT,'(A)') '// CreationDate: '//TIMBUF
     WRITE (IUNIT,'(A)') ''
     ! user override of default header
     IF (IGPOVIU.GT.0) THEN
        WRITE (IUNIT,'(A)') '// BEGIN USER OVERRIDE OF HEADER'
25      READ(IGPOVIU,'(A)',END=30) TMP
        WRITE (IUNIT,'(A)') TMP
        GOTO 25
30      CONTINUE
        IF (QSTERO) REWIND(IGPOVIU)
        WRITE (IUNIT,'(A)') '// USER HEADER ENDS'
        ! write the default header
     ELSE
        WRITE (IUNIT,'(A)') '// BEGIN DEFAULT CHARMM HEADER'
        WRITE (IUNIT,'(A)') '#version 3.0'
        WRITE (IUNIT,'(A)') 'global_settings { assumed_gamma 2.2'
        WRITE (IUNIT,'(A)') '  max_trace_level 144 }'
        WRITE (IUNIT,'(A)') '#include "colors.inc"'
        WRITE (IUNIT,'(A)') '#include "textures.inc"'
        WRITE (IUNIT,'(A)') '#include "finish.inc"'
        WRITE (IUNIT,'(A)') 'camera {'
        ! deriving camera position based on scale factor; exp for now
        !c        A = -50.*EXP(-0.5*GRSCAL)
        ! deriving camera position based on scale factor; hyperbolic
        A = (-27./GRSCAL)
        WRITE (CAMPOS,'(F9.2)') A
        WRITE (IUNIT,'(A)') '  location  <0, 0,'//CAMPOS//'>'
        WRITE (IUNIT,'(A)') '  direction <0, 0,  1>'
        WRITE (IUNIT,'(A)') '  up        <0, 1,  0>'
        WRITE (IUNIT,'(A)') '  right   <4/3, 0,  0>'
        WRITE (IUNIT,'(A)') '  look_at   <0, 0, 0.5>'
        WRITE (IUNIT,'(A)') '  orthographic }'
        ! calc light positions, behind camera
        A = A - 10.
        B = 0.1*A
        C = -1.0*B
        CALL POVVEC(C,C,A,T)
        WRITE (IUNIT,'(A)') 'light_source {'//T//' color Gray90}'
        CALL POVVEC(B,B,A,T)
        WRITE (IUNIT,'(A)') 'light_source {'//T//' color Gray50}'
        !c        WRITE (IUNIT,'(A)') 'sky_sphere { pigment { gradient y'
        !c        WRITE (IUNIT,'(A)') '  color_map { [0 color Gray50]'
        !c        WRITE (IUNIT,'(A)') '  [1 color NewMidnightBlue] }'
        !c        WRITE (IUNIT,'(A)') '  scale .1 translate -.05 } }'
        WRITE (IUNIT,'(A)') 'background { White }'
        WRITE (IUNIT,'(A)') '// HEADER ENDS; NO USER OVERRIDE'
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE POVINIT


SUBROUTINE POVETRM(IUNIT)
  !
  ! terminate POV file in ERAse OFF mode; POV UNIT N TERM
  !

  use chm_kinds
  use dimens_fcm
  use graph
  implicit none

  INTEGER IUNIT, I
  LOGICAL QERROR

  IF (QTITLE) THEN
     CALL WRNDIE(0,'POVETRM>','no title written by POV TERMinate')
  ENDIF
  CALL VCLOSE(IUNIT,'KEEP',QERROR)

  RETURN
END SUBROUTINE POVETRM

SUBROUTINE POVVEC(A,B,C,S)
  ! pack 3 reals into the string S in POV vector format
  use chm_kinds
  implicit none
  real(chm_real) A,B,C
  CHARACTER(len=*) S
  WRITE (S,911) A,B,C
911 FORMAT('<',F7.2,',',F7.2,',',F7.2,'>')
  RETURN
END SUBROUTINE POVVEC

SUBROUTINE SPOVCLR(ICOLR,MAP,S,IER)
  !
  use chm_kinds
  use stream
  implicit none
  !
  INTEGER ICOLR,MAP(0:*),MC,J100,IER
  CHARACTER(len=*) S
  real(chm_real) RED,GREEN,BLUE, DFF,A

  DFF  = 255.0
  J100 = 256
  IER = 0
  S = 'Coral'
  IF (ICOLR.LT.0 .OR. ICOLR.GT.191) THEN
     IER = -1
  ELSE IF (5.EQ.MOD(ICOLR,16)) THEN
     A = 1.0 - (ICOLR/191.)
     WRITE (S,901) A,A,A
  ELSE
     MC = MAP(ICOLR)
     BLUE = REAL( MOD(MC,J100)) / DFF
     MC = MC / J100
     GREEN = REAL( MOD(MC,J100)) / DFF
     MC = MC / J100
     RED = REAL( MOD(MC,J100)) / DFF
     MC = MOD(ICOLR,16)
     WRITE (S,901) RED,GREEN,BLUE
901  FORMAT('color rgb <',F6.3,',',F6.3,',',F6.3,'>')
  ENDIF
  RETURN
END SUBROUTINE SPOVCLR
#endif /* (nograph)*/


