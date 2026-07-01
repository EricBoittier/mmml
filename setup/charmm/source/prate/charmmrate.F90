module charmmrate_mod
  use chm_kinds
  use dimens_fcm
  implicit none

  integer chmrate_dummy_var

#if KEY_CHARMMRATE==1 /*charmmrate*/
  INTEGER,save :: ICHMAT(MAXA),NACTIVE
  REAL(chm_real),save :: AMSPR(maxa)
  integer,save :: itag_a, itag_b, itag_c, rndf 
  logical,save :: lpmf





#endif /* (charmmrate)*/


contains
#if KEY_CHARMMRATE==1 /*charmmrate1*/

  SUBROUTINE CHARMMRATE(COMLYN,COMLEN)
    !------------------------------------------------------
    !
    use coord
    use psf
    use stream
    use energym
    use memory
    use exfunc
    !
    !
    integer,allocatable,dimension(:) :: ISLCT
    real(chm_real),allocatable,dimension(:) :: XRACT,YRACT,ZRACT, &
         XPACT,YPACT,ZPACT,XTSACT,YTSACT,ZTSACT,XOPACT,YOPACT,ZOPACT
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    !
    INTEGER   RUNIT, PUNIT, TSUNIT, OPUNIT
    INTEGER   I0, I10 
    real(chm_real)    XIN
    CHARACTER(len=8) BADGE
    real(chm_real)    ATMZ(NATOM)

    call chmalloc('charmmrate.src','CHARMMRATE','ISLCT',NATOM,intg=ISLCT)

    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    CALL CR_INI(COMLYN,COMLEN,ISLCT,NATOM,IAC,AMASS,AMSPR, &
         ATMZ,NACTIVE,ICHMAT) 
    !
    ! activate projection reaction coordinate during EA-VTST/MT stage-1 calculation
    !
    LPMF = (INDXA(COMLYN,COMLEN,'PMFZ').GT.0)
    IF(LPMF) THEN
       ITAG_A  = GTRMI(COMLYN,COMLEN,'ATMA',-1)
       ITAG_B  = GTRMI(COMLYN,COMLEN,'ATMB',-1)
       ITAG_C  = GTRMI(COMLYN,COMLEN,'ATMC',-1)
    END IF
    !
    !
    ! unit number to assign R, P, SP, optimzed structure channels 
    !
    RUNIT  = GTRMI(COMLYN,COMLEN,'RUNI',-1)
    PUNIT  = GTRMI(COMLYN,COMLEN,'PUNI',-1)
    TSUNIT = GTRMI(COMLYN,COMLEN,'TSUN',-1)
    OPUNIT = GTRMI(COMLYN,COMLEN,'OPUN',-1)
    COMLEN = 0
    !
    IF(NACTIVE.LE.0)  &
         CALL WRNDIE(-3,'<CHARMMRATE>','ZERO DYNAMICS ATOMS SELECTED')
    call chmalloc('charmmrate.src','CHARMMRATE','XRACT',NACTIVE,crl=XRACT)
    call chmalloc('charmmrate.src','CHARMMRATE','YRACT',NACTIVE,crl=YRACT)
    call chmalloc('charmmrate.src','CHARMMRATE','ZRACT',NACTIVE,crl=ZRACT)
    call chmalloc('charmmrate.src','CHARMMRATE','XPACT',NACTIVE,crl=XPACT)
    call chmalloc('charmmrate.src','CHARMMRATE','YPACT',NACTIVE,crl=YPACT)
    call chmalloc('charmmrate.src','CHARMMRATE','ZPACT',NACTIVE,crl=ZPACT)
    call chmalloc('charmmrate.src','CHARMMRATE','XTSACT',NACTIVE,crl=XTSACT)
    call chmalloc('charmmrate.src','CHARMMRATE','YTSACT',NACTIVE,crl=YTSACT)
    call chmalloc('charmmrate.src','CHARMMRATE','ZTSACT',NACTIVE,crl=ZTSACT)
    call chmalloc('charmmrate.src','CHARMMRATE','XOPACT',NACTIVE,crl=XOPACT)
    call chmalloc('charmmrate.src','CHARMMRATE','YOPACT',NACTIVE,crl=YOPACT)
    call chmalloc('charmmrate.src','CHARMMRATE','ZOPACT',NACTIVE,crl=ZOPACT)
    !    
    ! reactant
    !
    IF(RUNIT.GT.0) THEN
       COMLYN = 'COOR CARD UNIT '
       I10 = INT(RUNIT/10)
       I0  = MOD(RUNIT,10)
       IF(I10.NE.0) THEN
          COMLYN(16:16) = CHAR(48+I10)
          COMLYN(17:17) = CHAR(48+I0)
          IF(I10.GE.10)  &
               CALL WRNDIE(-5,'CHARMMRATE>','RUNIT VALUE IS TOO LARGE')
          COMLEN = 17
       ELSE
          COMLYN(16:16) = CHAR(48+RUNIT)
          COMLEN = 16
       ENDIF
       BADGE='REACTANT'
       CALL CR_READ(RUNIT,COMLYN,COMLEN,XRACT,YRACT, &
            ZRACT,AMSPR,NACTIVE,ICHMAT,BADGE)
    ENDIF
    !
    ! product
    !
    IF(PUNIT.GT.0) THEN
       COMLYN = 'COOR CARD UNIT '
       I10 = INT(PUNIT/10)
       I0  = MOD(PUNIT,10)
       IF(I10.NE.0) THEN
          COMLYN(16:16) = CHAR(48+I10)
          COMLYN(17:17) = CHAR(48+I0)
          IF(I10.GE.10)  &
               CALL WRNDIE(-5,'CHARMMRATE>','PUNIT VALUE IS TOO LARGE')
          COMLEN = 17
       ELSE
          COMLYN(16:16) = CHAR(48+PUNIT)
          COMLEN = 16
       ENDIF
       BADGE='PRODUCTS'
       CALL CR_READ(PUNIT,COMLYN,COMLEN,xpact, &
            YPACT,ZPACT,AMSPR,NACTIVE,ICHMAT,BADGE)
    ENDIF
    !
    ! saddle point
    !
    IF(TSUNIT.GT.0) THEN
       COMLYN = 'COOR CARD UNIT '
       I10 = INT(TSUNIT/10)
       I0  = MOD(TSUNIT,10)
       IF(I10.NE.0) THEN
          COMLYN(16:16) = CHAR(48+I10)
          COMLYN(17:17) = CHAR(48+I0)
          IF(I10.GE.10)  &
               CALL WRNDIE(-5,'CHARMMRATE>','TSUNIT VALUE IS TOO LARGE')
          COMLEN = 17
       ELSE
          COMLYN(16:16) = CHAR(48+TSUNIT)
          COMLEN = 16
       ENDIF
       BADGE='T. STATE'
       CALL CR_READ(TSUNIT,COMLYN,COMLEN,XTSACT, &
            YTSACT,ZTSACT,AMSPR,NACTIVE,ICHMAT,BADGE)
    ENDIF
    !
    ! Polyrate starts
    !
    CALL CR_LAUNCH(xract,YRACT,ZRACT, &
         xpact,YPACT,ZPACT,XTSACT, &
         YTSACT,ZTSACT,XOPACT,YOPACT, &
         ZOPACT,AMSPR,ATMZ,NACTIVE,RUNIT,PUNIT,TSUNIT,OPUNIT)

    !
    ! write opt structure from polyrate
    !
    IF(OPUNIT.GT.0) THEN
       COMLYN = 'COOR CARD UNIT '
       I10 = INT(OPUNIT/10)
       I0  = MOD(OPUNIT,10)
       IF(I10.NE.0) THEN
          COMLYN(16:16) = CHAR(48+I10)
          COMLYN(17:17) = CHAR(48+I0)
          IF(I10.GE.10) &
               CALL WRNDIE(-5,'CHARMMRATE>','OPUNIT VALUE IS TOO LARGE')
          COMLEN = 17
       ELSE
          COMLYN(16:16) = CHAR(48+OPUNIT)
          COMLEN = 16
       ENDIF
       CALL CR_WRITE(OPUNIT,COMLYN,COMLEN,XOPACT,YOPACT, &
            ZOPACT,NACTIVE,ICHMAT)
    ENDIF

    call chmdealloc('charmmrate.src','CHARMMRATE','ISLCT',NATOM,intg=ISLCT)
    call chmdealloc('charmmrate.src','CHARMMRATE','XRACT',NACTIVE,crl=XRACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','YRACT',NACTIVE,crl=YRACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','ZRACT',NACTIVE,crl=ZRACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','XPACT',NACTIVE,crl=XPACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','YPACT',NACTIVE,crl=YPACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','ZPACT',NACTIVE,crl=ZPACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','XTSACT',NACTIVE,crl=XTSACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','YTSACT',NACTIVE,crl=YTSACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','ZTSACT',NACTIVE,crl=ZTSACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','XOPACT',NACTIVE,crl=XOPACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','YOPACT',NACTIVE,crl=YOPACT)
    call chmdealloc('charmmrate.src','CHARMMRATE','ZOPACT',NACTIVE,crl=ZOPACT)

    RETURN
  END SUBROUTINE CHARMMRATE
  !
  !==========================================================================
  !
  SUBROUTINE CR_LAUNCH(XRACT,YRACT,ZRACT,XPACT,YPACT,ZPACT,XTSACT, &
       YTSACT,ZTSACT,XOPACT,YOPACT,ZOPACT,AMSPR,ATMZ,NM,RU,PU,SPU,OPU)
    !
    !      Set the arry xr that will be passed to polyrate with the initial
    !      coordinates for R, P, and SP. 
    !      Converts the coordinates to atomic units. 
    ! 
    !      itype = 1  : reactant
    !      itype = 3  : product 
    !      itype = 5  : saddle point
    !
    !==========================================================================
    !
    !    include 'param.inc' 
    !#######################################################################
    !
    !   param2.inc (the old poly6775.inc) 
    !
    !   This file can be used as the param.inc INCLUDE file required by
    !   the POLYRATE source code.
    !
    !   PARAMETER STATEMENTS FOR POLYRATE 
    !
    !   THESE ARE THE ONLY PARAMETERS WHICH SHOULD BE MODIFIED BY THE USER.
    !
    !   NOTE:   IF THESE PARAMETERS ARE MODIFIED THE SOURCE CODE FOR 
    !           POLYRATE MUST BE RECOMPILED 
    !
    !               NSDM      =    THE MAXIMUM NUMBER OF SAVE POINTS 
    !                              ALONG THE MINIMUM ENERGY PATH
    !
    !               NSDML     =    THE MAXIMUM NUMBER OF SAVE POINTS 
    !                              FOR THE TUNNELING CALCULATIONS 
    !                              (NSDML .LE. NSDM)
    !
    !               NATOMS    =    MAXIMUM NUMBER OF ATOMS
    !
    !               NPOTPT    =    MAXIMUM NUMBER OF PIECES OF INFORMATION
    !                              WHICH MAY BE PASSED FROM THE POTENTIAL
    !                              ENERGY SURFACE (THIS INFORMATION IS NOT
    !                              USED IN THE CALCULATION BUT MAY BE 
    !                              PRINTED TO UNIT FU6 USING THE LGS2(5)
    !                              FLAG IN THE UNIT FU5 INPUT DECK)
    !
    !               MAXPS     =    MAXIMUM NUMBER OF PRODUCT STATES ALLOWED
    !                              FOR THE LCG3 TUNNELING CALCULATION
    !
    !               MAXWKB    =    MAXIMUM NUMBER OF REACTANT STATES ALLOWED
    !                              QUANTIZED REACTANT STATE TUNNELING CALCUL
    !
    !               NMSPEC    =    MAXIMUM NUMBER OF EXTRA SAVE POINTS
    !
    !               INMM      =    MAXIMUM VALUE FOR RATIO BETWEEN GRADIENT
    !                              SAVE POINTS AND HESSIAN SAVE POINTS NUMBE
    !                              (USED ONLY FOR FU31 RUNS)
    !
    !               FU#       =    FORTRAN UNIT NUMBER FOR SPECIFYING THE
    !                              POLYRATE INPUT AND OUTPUT FILES 
    !                              (SEE THE POLYRATE MANUAL FOR A FULL 
    !                               DESCRIPTION OF THE FILES ASSOCIATED 
    !                               WITH EACH #)
    !
    !
    !
    !
    integer,PARAMETER :: NSDIM= 501, NSDM = 501,NSDML = 501,NATOMS = 30, &
         MAXINT=3*NATOMS+6, MAXCAR=3*NATOMS,NPOTPT = 1, MAXPS = 5, &
         MAXWKB = 50,NMSPEC = 20,INMM = 10

    !  Note: the integer specification must be made before the parameter 
    !        declaration.

    integer,PARAMETER :: FU1 = 1, FU2 = 2, FU3 = 3, FU5 = 5, FU6 = 6, FU8 = 8, &
         FU14 = 14, FU15 = 15, FU18 = 18, FU19 = 19, &
         FU20 = 20, FU21 = 21, FU22 = 22, &
         FU25 = 25, FU26 = 26, FU27 = 27, FU28 = 28, &
         FU30 = 30, FU31 = 31, &
         FU71 = 71, FU72 = 72, FU73 = 73, FU74 = 74, FU75 = 75, FU77 = 77, FU78 = 78, &
         FU40 = 40, FU41 = 41, FU42 = 42, FU43 = 43, &
         FU44 = 44, FU45 = 45, FU46 = 46, FU47 = 47, &
         FU50 = 50, FU51 = 51, FU60 = 60, FU61 = 61, FU65 = 65

    !#######################################################################

    !include 'percon.inc'

    !include 'common.inc'
    !
    real(chm_real) XRACT(*),YRACT(*),ZRACT(*)
    real(chm_real) XPACT(*),YPACT(*),ZPACT(*)
    real(chm_real) XTSACT(*),YTSACT(*),ZTSACT(*)
    real(chm_real) XOPACT(*),YOPACT(*),ZOPACT(*)
    !     real(chm_real) XO(3*NM,8)
    real(chm_real) XOP(3*NM)
    real(chm_real) AMSPR(*), ATMZ(*)
    INTEGER NM, RU, PU, OPU, SPU, I1, I2, ITYPE
    !   
    real(chm_real),PARAMETER :: FACTOR=0.529177249_chm_real
    !   
    CALL CR_FILL1(AMSPR,XMASS,NM,NATOM,ATMZ,LABEL)
    !
    DO I1=1,NM*3
       DO I2=1,8
          XR(I1,I2)=0.D0
       ENDDO
    ENDDO
    !
    IF(RU.GT.0) THEN
       ITYPE=1 
       NRATOM(ITYPE) = NM
       I1 = 0
       DO I1=1,NM
          XR(I1*3-2,ITYPE)=XRACT(I1)/FACTOR 
          XR(I1*3-1,ITYPE)=YRACT(I1)/FACTOR 
          XR(I1*3  ,ITYPE)=ZRACT(I1)/FACTOR 
          IATSV(I1,ITYPE) = I1
       ENDDO
    ENDIF
    !
    IF(PU.GT.0) THEN
       ITYPE=3 
       NRATOM(ITYPE) = NM
       I1 = 0
       DO I1=1,NM
          XR(I1*3-2,ITYPE)=XPACT(I1)/FACTOR
          XR(I1*3-1,ITYPE)=YPACT(I1)/FACTOR
          XR(I1*3  ,ITYPE)=ZPACT(I1)/FACTOR
          IATSV(I1,ITYPE) = I1
       ENDDO
    ENDIF
    !
    IF(SPU.GT.0) THEN
       ITYPE=5
       NRATOM(ITYPE) = NM
       I1 = 0    
       DO I1=1,NM
          XR(I1*3-2,ITYPE)=XTSACT(I1)/FACTOR
          XR(I1*3-1,ITYPE)=YTSACT(I1)/FACTOR
          XR(I1*3  ,ITYPE)=ZTSACT(I1)/FACTOR
          IATSV(I1,ITYPE) = I1
       ENDDO
    ENDIF
    !
    ! Launch polyrate  
    !
    CALL CHARATE(1)
    !
    ! ---------------------------------------------------------
    !   before leaving put the optimized coordinates into xop()
    ! ---------------------------------------------------------
    !
    IF((IPATH.EQ.0).OR.(IRATE.EQ.0).OR.(ITUNNL.EQ.0)) THEN
       IF(OPU.GT.0) THEN
          I1 = 0
          DO I1=1,NM
             !        XOPACT(I1)=X(I1*3-2)*FACTOR
             !        YOPACT(I1)=X(I1*3-1)*FACTOR
             !        ZOPACT(I1)=X(I1*3  )*FACTOR
             ! jcc
             IF (ITYPE.NE.5) THEN
                XOPACT(I1)=X(I1*3-2)*FACTOR
                YOPACT(I1)=X(I1*3-1)*FACTOR
                ZOPACT(I1)=X(I1*3  )*FACTOR
             ELSE
                XOPACT(I1)=X(I1*3-2)*FACTOR/AMASS(I1*3-2)
                YOPACT(I1)=X(I1*3-1)*FACTOR/AMASS(I1*3-1)
                ZOPACT(I1)=X(I1*3  )*FACTOR/AMASS(I1*3)
             ENDIF
             ! jcc
          ENDDO
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE CR_LAUNCH
  !
  !==========================================================================
  !
  SUBROUTINE CR_INI(COMLYN,COMLEN,ISLCT,NATOM,IAC,AMASS,AMSPR, &
       ATMZ,NACTIVE,ICHMAT)
    !
    !==========================================================================

    use stream
    !
    CHARACTER(len=*) COMLYN
    CHARACTER(len=6) ELE
    INTEGER       COMLEN, ISLCT(*), IAC(*), ICHMAT(*)
    INTEGER       NATOM, NACTIVE
    INTEGER       I1,I2, i 
    real(chm_real)       ZNUM,  ATMZ(*)
    real(chm_real)       AMASS(*), AMSPR(*) 
    !
    ! find the selected atoms 
    ! fill the ichmat array
    !
    NACTIVE = 0 
    DO I1 = 1,NATOM 
       IF(ISLCT(I1).EQ.1) THEN 
          NACTIVE=NACTIVE+1
          ICHMAT(NACTIVE)=I1 
          AMSPR(NACTIVE)=AMASS(I1) 
          ELE = ATCT(IAC(I1))
          CALL FINDEL(ATCT(IAC(I1)),AMASS(I1),I1,ELE,ZNUM,.FALSE.) 
          ATMZ(NACTIVE) = ZNUM 
       ENDIF
    ENDDO
    !
    ! Write out selection 
    ! 
    IF(PRNLEV.GT.1) THEN 
       WRITE(OUTU,'(/,1X,A,i8)')  &
            'CR_INI> Atoms selected in the active region = ', NACTIVE 
       DO I2 = 1,NACTIVE
          WRITE(OUTU,'(2i6,2f10.5)') I2, ICHMAT(I2), ATMZ(I2),  &
               AMSPR(I2)         
       ENDDO
       WRITE(OUTU,*)
       !
    ENDIF
    !
    RETURN 
  END SUBROUTINE CR_INI
  !
  !=======================================================================
  !
  SUBROUTINE CR_READ(UNIT,COMLYN,COMLEN,XR,YR,ZR,AMSPR,NACTIVE, &
       ICHMAT,BADGE)
    !
    !=======================================================================
    !
    use coord
    use stream
    !
    INTEGER UNIT,NACTIVE,ICHMAT(*),COMLEN
    INTEGER I
    CHARACTER(len=8) BADGE
    CHARACTER(len=*) COMLYN
    CHARACTER(len=4) MODE
    !     real(chm_real) X(*),Y(*),Z(*),W(*),XR(*),YR(*),ZR(*)
    real(chm_real) XR(*),YR(*),ZR(*),AMSPR(*)
    !
    IF(PRNLEV.GE.1) WRITE(OUTU,'(/,A,i5,/)') &
         'CR_INI> Initial coordinates will be read from unit ', UNIT
    !
    MODE ='READ'
    CALL MAINIO(MODE)
    !
    DO I = 1,NACTIVE
       XR(I) = X(ICHMAT(I))
       YR(I) = Y(ICHMAT(I))
       ZR(I) = Z(ICHMAT(I))
    ENDDO
    !
    ! WRITE COORDINATES
    WRITE(6,'(/,a,/)') BADGE 
    DO I = 1,NACTIVE
       WRITE(OUTU,'(4f12.8)') AMSPR(I), XR(I), YR(I),ZR(I)
    ENDDO
    !
    RETURN
  END SUBROUTINE CR_READ
  !
  !=======================================================================
  !
  SUBROUTINE CR_WRITE(UNIT,COMLYN,COMLEN,XOPACT,YOPACT,ZOPACT, &
       NACTIVE,ICHMAT)
    !
    !=======================================================================
    !
    use coord
    use stream
    !
    INTEGER UNIT,NACTIVE,ICHMAT(*),COMLEN
    INTEGER I
    real(chm_real) XOPACT(*),YOPACT(*),ZOPACT(*)
    CHARACTER(len=*) COMLYN
    CHARACTER(len=4) MODE
    !
    !
    DO I = 1,NACTIVE
       X(ICHMAT(I)) = XOPACT(I) 
       Y(ICHMAT(I)) = YOPACT(I) 
       Z(ICHMAT(I)) = ZOPACT(I) 
    ENDDO
    !
    IF(PRNLEV.GE.1) WRITE(OUTU,'(/,A,i5,/)') &
         'CR_INI> Optimized coordinates for the active atoms will be  &
         writen as cards on unit ', UNIT
    !
    MODE ='WRITE'
    CALL MAINIO(MODE)
    !
    RETURN
  END SUBROUTINE CR_WRITE
  !
  !====================================================================
  !
  SUBROUTINE CR_ENER(XRATE,DXRATE,VPOTE)
    !
    !     Gets energy and gradients 
    !
    !==================================================================== 
    !
    use psf
    use bases_fcm
    use coord
    use energym
    use deriv

    INTEGER K1, K2
    real(chm_real) VPOTE,VKCAL
    real(chm_real) XRATE(*),DXRATE(*)
    !
    real(chm_real), PARAMETER :: CKCAL=627.5095D0,AU2AN=0.529177249D0
    !
    ! incorporate the pr coordinates for the active atoms into x,y,z
    ! 
    CALL CR_HOOK(XRATE,X,Y,Z,AU2AN,NACTIVE,ICHMAT)
    !
    ! call energy to evaluate the potential energy and gradients
    !
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    !
    ! assign potential energy
    !
    VKCAL=EPROP(EPOT)                      
    VPOTE=VKCAL/CKCAL
    !     write(6,*) ' VKCAL EPROP(EPOT) ' , vkcal, eprop(epot)
    !
    ! pick the gradients for the active atoms
    !
    DO K1=1,NACTIVE
       K2=ICHMAT(K1)
       DXRATE((K1-1)*3+1)=DX(K2)*AU2AN/CKCAL
       DXRATE((K1-1)*3+2)=DY(K2)*AU2AN/CKCAL
       DXRATE((K1-1)*3+3)=DZ(K2)*AU2AN/CKCAL
    ENDDO
    !
    RETURN
  END SUBROUTINE CR_ENER
  !
  !=====================================================================
  SUBROUTINE CR_HOOK(XRATE,X,Y,Z,AU2AN,NACTIVE,ICHMAT)
    !
    ! sets the charmm array with the coordinates for the matrix and active 
    ! passed from polyrate
    !
    !=====================================================================
    !

    INTEGER K1, K2
    INTEGER NACTIVE 
    INTEGER ICHMAT(*)
    real(chm_real) AU2AN
    real(chm_real) XRATE(*),X(*),Y(*),Z(*)
    !
    DO K1=1,NACTIVE 
       K2 = ICHMAT(K1)
       X(K2)=XRATE((K1-1)*3+1)*AU2AN
       Y(K2)=XRATE((K1-1)*3+2)*AU2AN
       Z(K2)=XRATE((K1-1)*3+3)*AU2AN
    ENDDO
    !
    RETURN
  END SUBROUTINE CR_HOOK
  !
  !=====================================================================
  !
  SUBROUTINE CR_FILL1(AMSPR,XMASS,NACTIVE,NATOM,ATMZ,LABEL)
    !
    !  dump NACTIVE into NATOM (as it should in polyrate since
    ! the set defalut general routine is not going to be used to
    ! fill out this variable.
    !  dump AMSPR into XMASS
    !  fill out the LABEL array
    !
    !=====================================================================
    !

    INTEGER      NACTIVE, NATOM, I
    INTEGER      LABEL(*)
    real(chm_real)       ATMZ(*), AMSPR(*), XMASS(*)
    !
    NATOM = NACTIVE
    !
    DO I = 1,NACTIVE
       XMASS(I) = AMSPR(I)
       LABEL(I) = INT( ATMZ(I) )
    ENDDO
    !
    RETURN
    !
  END SUBROUTINE CR_FILL1

#else /* (charmmrate1)*/

  SUBROUTINE CHARMMRATE(COMLYN,COMLEN)
    !------------------------------------------------------
    !
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    !
    !
    CALL WRNDIE(-1,'<CHARMMRATE>', &
         'CHARMM-POLYRATE code not compiled.')
    return
  end SUBROUTINE CHARMMRATE
#endif /* (charmmrate1)*/

end module charmmrate_mod

