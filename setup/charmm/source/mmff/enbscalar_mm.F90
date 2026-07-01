#if KEY_MMFF==1
SUBROUTINE EVDW_MM(ENB,EEL,NATOM,JNB,INBLO,CG,CNBA,CNBB,MAXROW, &
     IAC,ITC,NATC,IOFF,LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT, &
     DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     QECONT,ECONT,DD1,IUPT,QSECD,LVTRUNC,CTVTRN,LMSHFT &
#if KEY_FLUCQ==1
     ,QFLUC,FQCFOR    & 
#endif
     ,DELBUF,GAMBUF,DELQ &
#if KEY_IMCUBES==1
     ,lbycbim                              & 
#endif
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES NON BONDED INTERACTION ENERGIES AND FORCES
  !
  !     Jay Banks, 20 October 1995.  Copied from Subroutine EVDW in
  !     MSI/Merck version of code.  Will be modified to include only MMFF.
  !
  !     ENB    - vdw energy returned
  !     EEL    - electrostatic energy returned
  !     NATOM  - number of atoms
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     CNBA   - vdw well distance squared  (MAXROW*2)
  !     CNBB   - vdw well depths  (MAXROW*2)
  !     MAXROW  - offset for 1-4 interaction in CNBA and CNBB
  !     LELEC,,,,,,,, - logical flags used in BNBND.FCM
  !     IAC(J)  - lookup pointers into CNBA and CNBB  (NATOM)
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     DD1,QSECD - second derivative specifiers
  !
  !
  !     Include BLOCK energy partition
  !     By Youngdo Won         12/15/90
  !----------------------------------------------------------------------
  !
  ! 28 Nov 95 Jay Banks: removed spaces from "END IF"s.
  !
  ! 10-11 Jan 96 Jay Banks: corrected errors in CSHIFT (CDIE FSHIFT) and
  ! RSHFT (RDIE SHIFT) calculations.
  !
  use ewald,only:kappa,lewald
  use erfcd_mod
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  !
  use consta
  use stream
#if KEY_BLOCK==1
  use block_fcm       
#endif
  implicit none
  !
  real(chm_real) ENB,EEL
  INTEGER NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),CNBA(*),CNBB(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  INTEGER NATC,IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,LVTRUNC,LMSHFT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,CTVTRN,EPS,E14FAC
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
#if KEY_IMCUBES==1
  logical lbycbim                             
#endif
  real(chm_real) DELBUF,GAMBUF,DELQ,G0
  !
  real(chm_real) ETEMP1,ETEMP2,ENBPR,EELPR
  real(chm_real) C2OFNB,C4ROF,C2ROF2,CHROF2,C2ONNB
  real(chm_real) RUL3,RUL12
  real(chm_real) CGF,CGT
  real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,CGT2,DXI,DYI,DZI
  real(chm_real) S,S1,R2,RIJL,RIJU,FUNCT,DFN,D2FN
  real(chm_real) EN,DEN,DF,DDF,G2,R1,G1,G3,P,DXIT,DYIT,DZIT
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ
  real(chm_real) EADD,EADDR,ON3,ON6,OFF3,OFF6, &
       DENOM,ACOEF,BCOEF,CCOEF,DCOEF,TWOA, &
       TWOB,TWOC,TWOD,FOURD,COVER3,DOVER5,SIXA,SIXD,CONST,CONSTR, &
       CTOFM,CTONM,C2OFM,C2ONM,ONOFF2M,OFF3M,ON3M,OFF4M,OFF5M, &
       S1D,S2D,S3D,R1D,R2D,R3D,R4D
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       RECOF,RECOF2,MIN2OF
  real(chm_real) Q,Q2,Q4,Q6,Q7,F,H,GH,RPE,EREP,EATT,rp7g,pe
  real(chm_real) PE2,PE4
  real(chm_real) DFDP,DGDP,DGEDP,d2fdp2,d2gedp2,d2gedr2,ss,d2gdp2
  real(chm_real) poff, poff2, poff3, poff6, poff7, poffdi, poff7gi
  real(chm_real) foff, hoff, ghoff, vtoff, vtpoff,  &
       CBETA, CBBM1, CVSH, DVSH
  INTEGER I,NB,IADD,ITEMP,J,NPR,I1,IACI,JPR,J1,IC,II,JJ,NBPRINT
  real(chm_real) ERFCX,DERFC,E14F,E14M1
  real(chm_real) XVAL,REM,VAL0,VAL1,VAL2,D1,D2
  INTEGER IXVAL
  LOGICAL LROOT
  !
  !
  !     QSECD  - NO SECOND DERIVATIVES CALCULATED
  !     ELECFG - DO ELECTROSTATICS HERE
  !     LVDW   - DO VDW INTERACTION
  !
  ! Nonbonded potential is of the form (for non force-based cutoffs)
  !
  !   POTENTIAL' = POTENTIAL * S
  !
  !   where POTENTIAL is "pure" potential and
  !   S is shifting or switching functions
  !
  !   S-functions have following forms:
  !
  !   SHFT   : (1 - R**2/CTOFNB**2)**2
  !   SHIFT  : (1 - R/CTOFNB)**2
  !
  !             = 1 for R < CTONNB
  !   SWITCH :{ = (CTOFNB**2-R**2)*(CTOFNB**2+2*R**2-3*CTONNB**2)/
  !                    (CTOFNB**2-CTONNB**2)
  !             = 0 for R > CTOFNB
  !
  !   MSHFT  : [(1 - R**2/CTOFNB**2)**2 + (1 - R/CTOFNB)**3]/2
  !
  !   POTENTIAL has following forms:
  !
  !   ELEC CHARMM CDIE : QI*QJ/(EPS*R)
  !   ELEC CHARMM RDIE : QI*QJ/(EPS*R**2)
  !   ELEC MMFF   CDIE : QI*QJ/(EPS*(R+DELQ)) (Buffered Electrostatic)
  !   ELEC MMFF   RDIE : QI*QJ/(EPS*(R+DELQ)**2) (Buffered Electrostatic)
  !   VDW  CHARMM      : ES*( (RS/R)**12 - 2*(RS/R)**6) (LJ 12-6)
  !   VDW  MMFF        : ES*{ [(1+GAMBUF)*RS/(R+GAMBUF*RS)]**7 *
  !                      [(1+DELBUF*RS**7/(R**7 + DELBUF*RS**7) - 2 ] }
  !                      (Buffered 14-7)
  !
  !   For the MMFF electrostatic potential, R/CTOFNB is replaced in the
  !   various shifting functions by (R+DELQ)/(CTOFNB+DELQ).  A different
  !   form of shifting is used for MMFF VDW.  See below.
  !
  ! For force-based cutoffs potential gradient is multiplied by
  ! short range S-function. Potential has to be reconstructed
  ! as a integral ( FdR ). ( F = - grad(POTENTIAL))
  !
  LOGICAL LEWLDX ! CONSTANT DIEL. WITH EWALD SUMMATION
  LOGICAL RSHFT  ! R-DIELECTRIC WITH SHIFTED POTENTIAL
  LOGICAL RSHIFT ! R-DIELECTRIC WITH SHIFTED FORCE (NOT SUPPORTED)
  LOGICAL RSWIT  ! R-DIELECTRIC WITH SWITCHING FUNCTIONS
  LOGICAL RFSWIT ! R-DIELECTRIC WITH FORCE SWITCHING
  LOGICAL RMSHI  ! R-DIELECTRIC WITH MODIFIED FORCE/POTENTiAL SHIFTING
  LOGICAL CSHFT  ! CONSTANT DIEL. WITH SHIFTED POTENTIAL (1-(r/off)**2)**2
  LOGICAL CSHIFT ! CONSTANT DIEL. WITH SHIFT: S(r) = (1-r/roff)**2
  LOGICAL CSWIT  ! CONSTANT DIEL. WITH SWITCHING FUNCTIONS
  LOGICAL CFSWIT ! CONSTANT DIEL. WITH FORCE SWITCHING
  LOGICAL CMSHI  ! CONSTANT DIEL. WITH MODIFIED FORCE/POTENTiAL SHIFTING
  !
  LOGICAL DTRUNC ! VDW TRUNCATION
  LOGICAL DSHFT  ! VDW DISTANCE SHIFTING
  LOGICAL DSWIT  ! VDW DISTANCE SWITCHING (DEFAULT)
  LOGICAL DFSWIT ! VDW FORCE SWITCHING (NOT SUPPORTED FOR MMFF)
  !
  LOGICAL LSECD,ELECFG
  LOGICAL ELCFG,LUSED
  !
  !
#if KEY_BLOCK==1
  INTEGER IBL,JBL,KK
  real(chm_real)  COEF
#endif /*  BLOCK*/
  !
  ! derivatives of radially symmetric function F(R)
  !
  ! F' = dF/dR ; F'' = d(dF/dR)/dR
  !
  ! dF/dX = (X/R) * F' = X * DF; DF = F'/R
  !
  ! d2F/dR2 = F'/R + R.R * (F'' - F'/R)/(R*R)
  !
  ! d2F/dR2 =  DF  + R.R *     DDF
  !
  ! NOTE: CONTRIBUTIONS TO DF = F'/R, AND A TEMPORARY DDF = F''/(R*R), ARE
  ! CALCULATED SEPARATELY FOR EACH PART OF THE ENERGY (ELECTROSTATIC AND
  ! VDW FOR EACH PAIR OF ATOMS).  AFTER THESE ARE ACCUMULATED, THE
  ! TEMPORARY DDF IS REPLACED BY THE "CORRECT" DDF = DDF - DF/(R*R), WHICH
  ! IS USED IN CALCULATING CARTESIAN SECOND DERIVATIVES.
  !
  ! R.R - tensor product of vector R=(X,Y,Z)
  !
  !         WRITE(6,*) 'IN EVDW: LMSHFT = ', LMSHFT
  LSECD=QSECD
  E14M1 = E14FAC-ONE
  !      TKAPPA = TWO*KAPPA*RSQPI !to put this in, need PARAMETER
  !                               !statement for RSQPI
  ! Set flags for electrostatics options (8 possibilities)
  RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT
  RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT
  RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT
  RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
       .AND. .NOT.LMSHFT
  CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT
  CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT
  CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT
  CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT &
       .AND. .NOT.LMSHFT
  CMSHI =      LCONS .AND.      LMSHFT
  RMSHI = .NOT.LCONS .AND.      LMSHFT
  !
#if KEY_FLUCQ==1
  IF(QFLUC) CALL WRNDIE(-4,'<EVDW_MM>', &
       'No FlucQ implementation for EVDW_MM yet')
#endif 
  LEWLDX=LEWALD
  IF(LEWALD) THEN
     RSHFT=.FALSE.
     RSWIT=.FALSE.
     CSWIT=.FALSE.
     CSHFT=.FALSE.
  ENDIF
  !
  ELECFG=LELEC.AND.(EPS.NE.ZERO)
  ELCFG=.FALSE.
  LROOT=.FALSE.
  !
  ! Set flags for van der Waals options
  IF (LVDW) THEN
     DTRUNC = LVTRUNC
     DFSWIT = LVFSWT
     DSHFT = .NOT.LVFSWT .AND. .NOT.LVTRUNC  .AND.  LVSHFT
     DSWIT = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. .NOT.LVTRUNC
  ELSE
     DTRUNC = .FALSE.
     DFSWIT = .FALSE.
     DSWIT  = .FALSE.
     DSHFT  = .FALSE.
  ENDIF
  !
  IF (.NOT.(ELECFG.OR.LVDW)) RETURN
  !
  C2ONNB=CTONNB*CTONNB
  C2OFNB=CTOFNB*CTOFNB
  !
  ! Precompute electrostatic coefficients
  IF(LEWLDX) THEN
     !       switching function needed for the ewald delq term
     IF (CTOFNB.GT.CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ELSE IF(RSHIFT) THEN
     CALL WRNDIE(-5,'<EVDW_MM>','FSHIft not supported with RDIE')
     RETURN
  ELSE IF (CFSWIT.OR.RFSWIT) THEN
     !       force-based electrostatic switching coeffs
     IF(CTONNB .LT. CTOFNB) THEN
        CTOFM = CTOFNB + DELQ
        C2OFM = CTOFM * CTOFM
        CTONM = CTONNB + DELQ
        C2ONM = CTONM * CTONM
        ONOFF2M = C2ONM*C2OFM
        OFF3M = C2OFM * CTOFM
        ON3M  = C2ONM * CTONM
        OFF4M = C2OFM * C2OFM
        OFF5M = OFF3M * C2OFM
        DENOM  = ONE/(C2OFM-C2ONM)**3
        ACOEF  = OFF4M*(C2OFM-THREE*C2ONM)*DENOM
        BCOEF  = SIX*ONOFF2M*DENOM
        COVER3 = -(C2ONM+C2OFM)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        TWOA   = TWO*ACOEF
        TWOC   = TWO*CCOEF
        TWOB   = TWO*BCOEF
        TWOD   = TWO*DCOEF
        SIXA   = SIX*ACOEF
        SIXD   = SIX*DCOEF
        FOURD  = FOUR*DCOEF
        DOVER5 = DCOEF/FIVE
        EADD   = (ONOFF2M*(CTOFM-CTONM)-(OFF5M-ON3M*C2ONM)/FIVE)* &
             EIGHT*DENOM
        CONST  = BCOEF*CTOFM-ACOEF/CTOFM+COVER3*OFF3M+DOVER5*OFF5M
        !           rdie
        EADDR  = (TWELVE*ONOFF2M*LOG(CTOFM/CTONM) - &
             THREE*(OFF4M-C2ONM*C2ONM))*DENOM
        CONSTR=TWOB*LOG(CTOFM)-ACOEF/C2OFM+CCOEF*C2OFM+OFF4M*DENOM
     ELSE
        !clbiii added for proper initialization
        CTOFM = CTOFNB + DELQ
        C2OFM = CTOFM * CTOFM
        !clbiii
        EADD  = -ONE/CTOFM
        EADDR = -ONE/C2OFM
     ENDIF
     !
  ELSE IF (LSHFT.OR.LMSHFT) THEN
     !       shifted dielectric coefficients
     CTOFM = CTOFNB + DELQ
     C2OFM = CTOFM * CTOFM
     RECOF2 = ONE/C2OFM
     RECOF = ONE/CTOFM
     MIN2OF = MINTWO/CTOFM
     C4ROF=ONE/(C2OFM*C2OFM)
     C2ROF2=MINTWO/C2OFM
     CHROF2=-HALF/C2OFM 

  ELSE
     !     switching electrostatic options
     IF (CTOFNB.GT.CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ENDIF
  !
  ! Precompute van der waal coefficients
  IF (DSWIT) THEN
     !     van der waal distance switching coefficients
     IF (CTOFNB.GT.CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ELSEIF(LVDW.AND..NOT.(DSHFT.OR.DTRUNC)) THEN
     CALL WRNDIE(-5,'<EVDW_MM>','UNSUPPORTED VDW OPTION')
  ENDIF
  !
  ! Initialize pointers ands accumulation variables
  NB=0
  ITEMP=0
  ENB=0.0
  EEL=0.0
  IF(ELECFG) CGF=CCELEC/EPS
  !
  ! Initialize the vdw code lookup offsets
  !
  J=0
  DO I=1,NATC
     IOFF(I)=J
     J=J+I
  ENDDO
  !
  if(QECONT .and. prnlev.ge.6) then
     NBPRINT=0
     WRITE(OUTU,800)
800  FORMAT(//,' SIGNIFICANT NONBONDED INTERACTIONS:'//4X, &
          ' ATOM  PAIR',11X,'R',7X,'VDW      EREP    EATTR      EQ', &
          '     R*    EPS'/)
     write(outu,'(a,2f10.2)') ' CTONNB,CTOFNB       =', &
          CTONNB,CTOFNB
     write(outu,'(a,3f10.2)') ' EPS,E14FAC,DELQ     =', &
          EPS,E14FAC,DELQ
     write(outu,'(a,3f10.2)') ' GAMBUF,DELBUF=', &
          GAMBUF,DELBUF
  endif
  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  loop40:DO I=1,NATOM
     ETEMP1=0.0
     ETEMP2=0.0
#if KEY_IMCUBES==1
     if(lbycbim)itemp=inblo(i+natom)                   
#endif
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF (NPR.EQ.0) cycle loop40
     I1=IAC(I)
     IACI=IOFF(I1)
     !
     IF (ELECFG) THEN
        CGT=CGF*CG(I)
        ELCFG=CGT.NE.0.0
     ENDIF
     !
     !     USE FDXI,FDYI,FDZI FOR ITH COMPONENT OF FORCE VECTORS
     !     USE CRXI,CRYI,CRZI FOR ITH COMPONENT OF THE COORDINATES
     !
     FDXI=DX(I)
     FDYI=DY(I)
     FDZI=DZ(I)
     CRXI=X(I)
     CRYI=Y(I)
     CRZI=Z(I)
     !
     loop20:DO JPR=1,NPR
        EATT=ZERO
        EREP=ZERO
        NB=NB+1
        J=ABS(JNB(NB))
        DXI=CRXI-X(J)
        DYI=CRYI-Y(J)
        DZI=CRZI-Z(J)
        S=DXI*DXI+DYI*DYI+DZI*DZI
        if(S.ge.C2OFNB) cycle loop20 ! skip the loop if out of CTOFNB
        R2=1.0/S
        S1=SQRT(S)
        R1=ONE/S1
        LROOT=.TRUE.
        IF (JNB(NB).LT.0) THEN
           IF(ELECFG.AND..NOT.LEWALD) CGT2=CGT*E14FAC
           E14F = E14M1
           IC=MAXROW
        ELSE
           CGT2=CGT
           E14F=ZERO
           IC=0
        ENDIF
        J1=IAC(J)
        IF (I1.LT.J1) THEN
           IC=IC+IOFF(J1)+I1
        ELSE
           IC=IC+IACI+J1
        ENDIF
        !
        !
        !     TO COMPUTE VDW INTERACTION FOR THIS PAIR
        !
        ! LUSED = vdw calculation done, LROOT = square root done.
        LUSED = .FALSE.
        !
        if(LVDW.and.CNBB(IC).ne.0) then
           LUSED=.TRUE.
           IF (DSWIT)  THEN       ! VDW DISTANCE SWITCHING FUNCTION
              Q=S1/CNBA(IC)
              Q2=Q*Q
              Q4=Q2*Q2
              Q6=Q4*Q2
              Q7=Q6*Q
              RP7G=1./(Q7+GAMBUF)
              RPE=1./(Q+DELBUF)
              H=(1.+GAMBUF)*RP7G
              GH=H-2.
              PE=(1.+DELBUF)*RPE
              PE2=PE*PE
              PE4=PE2*PE2
              F=CNBB(IC)*PE4*PE2*PE
              EREP=F*H
              EATT=-2.*F
              EN=EREP+EATT
              IF (S.GT.C2ONNB) THEN
                 RIJL=C2ONNB-S
                 RIJU=C2OFNB-S
                 FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                 DFN=RIJL*RIJU*RUL12
                 ENBPR=FUNCT*EN
                 DFDP=-7.*F*RPE
                 DGDP=-7.*Q6*H*RP7G
                 DGEDP=F*DGDP+GH*DFDP
                 DEN=DGEDP*R1/CNBA(IC)
                 DF=DFN*EN+FUNCT*DEN
                 IF(LSECD) THEN
                    D2FDP2=-8.*DFDP*RPE
                    D2GDP2=7.*Q4*Q*H*RP7G*(-6.+14.*Q7*RP7G)
                    D2GEDP2=GH*D2FDP2+F*D2GDP2+2.*DFDP*DGDP
                    D2GEDR2=D2GEDP2/CNBA(IC)**2
                    SS=D2GEDR2*R2 ! /(R*R)
                    DDF=FUNCT*SS+ &
                         TWO*DFN*DEN+EN*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                 ENDIF
              ELSE
                 ENBPR=EN
                 DFDP=-7.*F*RPE
                 DGDP=-7.*Q6*H*RP7G
                 DGEDP=F*DGDP+GH*DFDP
                 DF=DGEDP*R1/CNBA(IC)
                 if(LSECD) then
                    D2FDP2=-8.*DFDP*RPE
                    D2GDP2=7.*Q4*Q*H*RP7G*(-6.+14.*Q7*RP7G)
                    D2GEDP2=GH*D2FDP2+F*D2GDP2+2.*DFDP*DGDP
                    D2GEDR2=D2GEDP2/CNBA(IC)**2
                    DDF=D2GEDR2*R2 ! /(R*R)
                 endif
              ENDIF
              !             LUSED=.TRUE.
              !==TAH
           ELSEIF (DTRUNC) THEN       ! vdw truncation
              IF(S1.LE.CTVTRN) THEN
                 !      write(6,*) ' MMFF vdw truncation'
                 Q=S1/CNBA(IC)
                 Q2=Q*Q
                 Q4=Q2*Q2
                 Q6=Q4*Q2
                 Q7=Q6*Q
                 RP7G=1./(Q7+GAMBUF)
                 RPE=1./(Q+DELBUF)
                 H=(1.+GAMBUF)*RP7G
                 GH=H-2.
                 PE=(1.+DELBUF)*RPE
                 PE2=PE*PE
                 PE4=PE2*PE2
                 F=CNBB(IC)*PE4*PE2*PE
                 EREP=F*H
                 EATT=-2.*F
                 EN=EREP+EATT
                 ENBPR=EN
                 DFDP=-7.*F*RPE
                 DGDP=-7.*Q6*H*RP7G
                 DGEDP=F*DGDP+GH*DFDP
                 DF=DGEDP*R1/CNBA(IC)
                 if(LSECD) then
                    D2FDP2=-8.*DFDP*RPE
                    D2GDP2=7.*Q4*Q*H*RP7G*(-6.+14.*Q7*RP7G)
                    D2GEDP2=GH*D2FDP2+F*D2GDP2+2.*DFDP*DGDP
                    D2GEDR2=D2GEDP2/CNBA(IC)**2
                    DDF=D2GEDR2*R2 ! /(R*R)
                 ENDIF
              ELSE
                 EREP = ZERO
                 EATT = ZERO
                 ENBPR = ZERO
                 DF = ZERO
                 DDF = ZERO
              ENDIF
              !
           ELSEIF (DFSWIT) THEN       ! vdw force-based switching
              call wrndie(-5,'<EVDW_MM>', &
                   'Unsupported VDW/MMFF option: DFSWIT')
              return
           ELSEIF (DSHFT)  THEN       ! VDW DISTANCE SHIFTED FUNCTION
              !      This is the "generalization of force shifting" discussed in P.J.
              !      Steinbach and B.R. Brooks, "New Spherical Cutoff Methods..." (J.
              !      Comp. Chem., in press).  We use the equivalent of their Equation
              !      15 for the Buffered 14-7 potential, with beta = 4.
              !            V(r) = V_true(r) + C r**beta + D,
              !      where
              !            C = -V_true'(r_off)/(beta * r_off**(beta-1)),
              !            D = -V_true(r_off) + r_off*V_true'(r_off)/beta,
              !      to make V(r_off) = V'(r_off) = 0.
              !
              poff = CTOFNB/CNBA(IC)
              poff2 = poff * poff
              poff3 = poff * poff2
              poff6 = poff3 * poff3
              poff7 = poff6 * poff
              poffdi = 1.0 / (poff + DELBUF)
              poff7gi = 1.0 / (poff7 + GAMBUF)
              foff = CNBB(IC) * ((1.+DELBUF)*poffdi)**7
              hoff = (1.+GAMBUF)*poff7gi
              ghoff = hoff - 2.
              vtoff = foff * ghoff                         !V_true(r_off)
              vtpoff = -(SEVEN*foff/CNBA(IC)) * &
                   (poffdi*ghoff + poff6*hoff*poff7gi) !V_true'(r_off)
              CBETA = -vtpoff/(CTOFNB*C2OFNB)              !beta=4
              CBBM1 = 3.0 * CBETA                  ! C * beta * (beta-1)
              CVSH = 0.25 * CBETA
              DVSH = -vtoff + 0.25*vtpoff*CTOFNB
              Q=S1/CNBA(IC)
              Q2=Q*Q
              Q4=Q2*Q2
              Q6=Q4*Q2
              Q7=Q6*Q
              RP7G=1./(Q7+GAMBUF)
              RPE=1./(Q+DELBUF)
              H=(1.+GAMBUF)*RP7G
              GH=H-2.
              PE=(1.+DELBUF)*RPE
              PE2=PE*PE
              PE4=PE2*PE2
              F=CNBB(IC)*PE4*PE2*PE
              EREP=F*H
              EATT=-2.*F
              EN=EREP+EATT
              ENBPR = EN + CVSH * S * S + DVSH ! C r**4 + D
              DFDP=-7.*F*RPE
              DGDP=-7.*Q6*H*RP7G
              DGEDP=F*DGDP+GH*DFDP
              DEN=DGEDP*R1/CNBA(IC)
              DF = DEN + CBETA * S      ! beta * C r**3 / r
              IF(LSECD) THEN
                 D2FDP2=-8.*DFDP*RPE
                 D2GDP2=7.*Q4*Q*H*RP7G*(-6.+14.*Q7*RP7G)
                 D2GEDP2=GH*D2FDP2+F*D2GDP2+2.*DFDP*DGDP
                 D2GEDR2=D2GEDP2/CNBA(IC)**2
                 SS=D2GEDR2*R2 ! /(R*R)
                 DDF = SS + CBBM1            !beta(beta-1) * C r**2/r**2
              ENDIF
           ELSE
              CALL WRNDIE(-5,'<EVDW_MM>','Unsupported VDW/MMFF option')
              return
           ENDIF
        endif ! if(LVDW) then
        !
        !     do electrostatics
        EELPR=0.0
        IF (ELCFG) THEN
           IF (CG(J).NE.0.0) THEN
              !             IF(S.LT.C2OFNB) THEN
              IF (.NOT.LUSED) THEN
                 !                 R2=1.0/S
                 DF=0.0
                 DDF=0.0
                 LUSED=.TRUE.
                 ENBPR=0.0
              ENDIF
              !-----------------------------------------------------------------------
              ! MMFF ewald nonbond option
              IF(LEWLDX) THEN
                 ! Inline erfc calculation for speed (from ERFCD).
                 XVAL=S1*KAPPA*EWRDEL
                 IXVAL=XVAL+HALF
                 REM=XVAL-IXVAL
                 IXVAL=IXVAL+2
                 IXVAL=MIN(IXVAL,EWNPTS-1)
                 VAL0=Ewldt(IXVAL-1)
                 VAL1=EwldT(IXVAL)
                 VAL2=EwldT(IXVAL+1)
                 D1=(VAL0-VAL2)*HALF
                 D2=(VAL1+VAL1-VAL0-VAL2)*REM
                 ERFCX=VAL1-(D1+HALF*D2)*REM
                 DERFC=(D1+D2)*EWRDEL*KAPPA
                 ! end of erfc inlined code.
                 S1D=S1+DELQ
                 R1D=ONE/S1D
                 G1 = CGT*CG(J)
                 IF(S.GT.C2ONNB) THEN
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
                    EELPR=G1*(R1*(ERFCX-FUNCT*DELQ*R1D) + R1D*E14F)
                    DF= DF -  G1*(R1*R1D*R1D*E14F + &
                         R2*(DERFC + R1*ERFCX - FUNCT*R1D*DELQ*(R1D+R1))+ &
                         R1*DFN*DELQ*R1D  )
                 ELSE
                    EELPR=G1*(R1*(ERFCX-DELQ*R1D) + R1D*E14F)
                    DF= DF -  G1*(R1*R1D*R1D*E14F + &
                         R2*(DERFC + R1*ERFCX - R1D*DELQ*(R1D+R1)) )
#if KEY_DEBUG==1
                    FUNCT=ONE   
#endif
#if KEY_DEBUG==1
                    DFN=ZERO    
#endif
                 ENDIF
#if KEY_DEBUG==1
                 write(60,456) I,J,S1,EELPR,DF,FUNCT,DFN
456              format(2i5,5f16.8)
#endif 
                 !-----------------------------------------------------------------------
                 !  rdie switching function inner section
              ELSE IF (RSWIT) THEN
                 S1D = S1 + DELQ
                 R1D = ONE/S1D
                 R2D = R1D * R1D
                 G1=CGT2*CG(J)*R2D
                 IF (S.GT.C2ONNB) THEN
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
                    EELPR=FUNCT*G1
                    DEN=-TWO*R1*G1*R1D
                    DF=DF+DFN*G1+FUNCT*DEN
                    IF(LSECD) THEN
                       DDF=DDF+TWO*(THREE*FUNCT*G1*R2*R2D + &
                            DFN*DEN) + &
                            G1*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                    ENDIF
                 ELSE
                    EELPR=G1
                    DF=DF-TWO*R1*G1*R1D
                    IF(LSECD) THEN
                       DDF=DDF+SIX*G1*R2*R2D
                    ENDIF
                 ENDIF
                 !-----------------------------------------------------------------------
                 !  rdie force-based switching
              ELSEIF (RFSWIT) THEN
                 G2=CGT2*CG(J)
                 S1D = S1+DELQ
                 S2D = S1D * S1D
                 S3D = S1D * S2D
                 R1D = ONE/S1D
                 R2D = R1D * R1D
                 R3D = R1D * R2D
                 R4D = R2D * R2D
                 IF (S .GT. C2ONNB) THEN
                    EELPR = G2*(ACOEF*R2D + TWOB*LOG(R1D) - &
                         S2D*(CCOEF + S2D*DENOM) + CONSTR)
                    DF = DF-G2*R1*(TWOA*R3D+TWOB*R1D+TWOC*S1D+TWOD*S3D)
                    IF(LSECD) DDF = DDF+G2*R2*((SIXA*R2D+TWOB)*R2D- &
                         TWOC-SIXD*S2D)
                 ELSE
                    EELPR = G2*(R2D + EADDR)
                    DF = DF - TWO*G2*R1*R3D
                    IF(LSECD) DDF = DDF + SIX*G2*R2*R4D
                 ENDIF
                 !
                 !-----------------------------------------------------------------------
                 !  cdie constant shift inner section
              ELSEIF (CSHFT)  THEN
                 !
                 !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                 !     EEL = QI*QJ/(EPS*(R+QDEL)) * (1 - ((R+QDEL)/(CTOFNB+QDEL)**2)**2
                 !         WITH CTOFM = CTOFNB+QDELC     
                 !     EEL=0.0  ( R > CTOFNB )
                 !
                 S1D = S1 + DELQ
                 S2D = S1D*S1D
                 R1D = ONE/S1D
                 R2D = R1D * R1D
                 G0 = CGT2*CG(J)
                 G1 = G0*R1D
                 G2 = G1*S2D*C2ROF2
                 G3 = G2*S2D*CHROF2
                 EELPR=G1+G2+G3
                 DF=DF+R1*R1D*(G2-G1+THREE*G3)
                 IF(LSECD) DDF=DDF+R2*R2D*(TWO*G1+SIX*G3)
                 !
                 !-----------------------------------------------------------------------
                 !  cdie constant-shift-inner-section
              ELSEIF (CSHIFT) THEN
                 !
                 !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                 !     EEL=QI*QJ/(EPS*(R+QDEL))*(1 - (R+QDEL)/(CTOFNB+QDEL))**2
                 !         WITH CTOFM = CTOFNB+QDEL
                 !     EEL=0.0  ( R > CTOFNB )
                 !
                 S1D = S1+DELQ
                 S2D = S1D*S1D
                 R1D = ONE/S1D
                 R2D = R1D*R1D
                 G0 = CGT2*CG(J)
                 G1 = G0*R1D
                 EELPR = G1*(ONE + MIN2OF*S1D + RECOF2*S2D)
                 DF=DF + G0*R1*(RECOF2-R2D)
                 IF(LSECD) DDF=DDF+TWO*G1*R2D*R2 
                 !
                 !-----------------------------------------------------------------------
                 !  rdie shift-inner-section
              ELSEIF (RSHFT)  THEN
                 !
                 !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                 ! EEL=QI*QJ/EPS*(1./(R+DELQ)**2)*(1-(R+DELQ)**2/((CTOFNB+DELQ)**2))**2
                 !     FUNCT = (1 - ((R+DELQ)/(CTOFNB+DELQ))**2)**2
                 !           = 1+RD**2*(-2/CTOFD**2+RD**2/CTOFD**4)
                 !     Where RD = R + DELQ, CTOFD = CTOFNB + DELQ.
                 !     G2 = QI*QJ/(EPS*(R+DELQ)**2)
                 !     DEN = (1/R)*(dG2/dR)
                 !         = -(2/R) * (QI*QJ/(EPS*(R+DELQ)**3)) = -2*G2/(R*(R+DELQ))
                 !     d2G1/dR2 = 6*(QI*QJ/(EPS*(R+DELQ)**4)) = 6*G2/(R+DELQ)**2
                 !     EEL=0.0  ( R > CTOFNB )
                 !
                 S1D = S1 + DELQ
                 S2D = S1D * S1D
                 R1D = ONE/S1D
                 R2D = R1D * R1D
                 FUNCT = ONE + S2D*(C2ROF2 + S2D*C4ROF)
                 DFN = C2ROF2*S1D*(TWO + S2D*C2ROF2)
                 G2=CGT2*CG(J)*R2D
                 DEN = -TWO*G2*R1D
                 EELPR=G2*FUNCT
                 DF=DF+(FUNCT*DEN+G2*DFN)*R1
                 IF(LSECD) DDF=DDF+R2*(SIX*G2*R2D*FUNCT + &
                      TWO*DEN*DFN         + &
                      G2*C2ROF2*(TWO+THREE*S2D*C2ROF2))
                 !
                 !-----------------------------------------------------------------------
                 !  cdie shift-inner-section
              ELSEIF (CMSHI)  THEN
                 !
                 !     THE FORM OF THE POTENTIAL IS
                 !     EEL=QI*QJ/EPS*(1./(R+DELQ)) * ((1-P**2)**2 + (1-P)**3)/2,
                 !         where P = (R+DELQ)/(CTOFNB+DELQ)
                 !     EEL=0.0  ( R > CTOFNB )
                 !
                 S1D = S1+DELQ
                 R1D = ONE/S1D
                 R2D = R1D*R1D
                 P = S1D*RECOF
                 G1 = ONE - P
                 G2 = ONE - P*P
                 FUNCT = HALF*(G2**2 + G1**3)
                 DFN = -HALF*(FOUR*S1D*RECOF2*G2+THREE*RECOF*G1**2)
                 G3 = CGT2*CG(J)*R1D
                 EELPR = G3*FUNCT
                 DF=DF + R1*(-EELPR*R1D+G3*DFN)
                 IF(LSECD) THEN
                    D2FN = RECOF2*(MINTWO*G2+FOUR*P*P+3*G1)
                    DDF=DDF+R2*(TWO*EELPR*R2D-TWO*G3*R1D*DFN+G3*D2FN)
                 ENDIF
                 !-----------------------------------------------------------------------
                 !  rdie modified shift-inner-section
              ELSEIF (RMSHI)  THEN
                 !
                 !     THE FORM OF THE POTENTIAL IS
                 !     EEL=QI*QJ/EPS*(1./(R+DELQ)**2) * ((1-P**2)**2 + (1-P)**3)/2,
                 !         where P = (R+DELQ)/(CTOFNB+DELQ)
                 !     EEL=0.0  ( R > CTOFNB )
                 !
                 S1D = S1+DELQ
                 R1D = ONE/S1D
                 R2D = R1D*R1D
                 P = S1D*RECOF
                 G1 = ONE - P
                 G2 = ONE - P*P
                 FUNCT = HALF*(G2**2 + G1**3)
                 DFN = -HALF*(FOUR*S1D*RECOF2*G2+THREE*RECOF*G1**2)
                 G3 = CGT2*CG(J)*R2D
                 EELPR = G3*FUNCT
                 DF=DF + R1*(MINTWO*EELPR*R1D+G3*DFN)
                 IF(LSECD) THEN
                    D2FN = RECOF2*(MINTWO*G2+FOUR*P*P+3*G1)
                    DDF=DDF+R2*(SIX*EELPR*R2D-FOUR*G3*R1D*DFN+G3*D2FN)
                 ENDIF
                 !-----------------------------------------------------------------------
                 !  cdie switching function
              ELSEIF (CSWIT)  THEN
                 G1=CGT2*CG(J)/(S1+DELQ)
                 IF (S.GT.C2ONNB) THEN
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
                    EELPR=FUNCT*G1
                    DEN=-R2*G1*S1/(S1+DELQ)
                    DF=DF+DFN*G1+FUNCT*DEN
                    IF(LSECD) THEN
                       !                     DDF=DDF+TWO*(FUNCT*G1*R2*R2+DFN*DEN)+
                       !    1                  G1*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                       DDF=DDF+TWO*(FUNCT*G1*R2/(S1+DELQ)**2+DFN*DEN)+ &
                            G1*(DFN*R2-TWO*(RIJU+RIJL)*RUL12)
                    ENDIF
                 ELSE
                    EELPR=G1
                    DF=DF-R2*G1*S1/(S1+DELQ)
                    IF(LSECD) THEN
                       !                     DDF=DDF+TWO*G1*R2*R2
                       DDF=DDF+TWO*G1*R2/(S1+DELQ)**2
                    ENDIF
                 ENDIF
                 !-----------------------------------------------------------------------
                 !  cdie force-based switching
              ELSEIF (CFSWIT) THEN
                 G1 = CGT2*CG(J)
                 S1D = S1 + DELQ
                 S2D = S1D * S1D
                 R1D = ONE/S1D
                 R2D = R1D * R1D
                 R3D = R1D * R2D
                 IF (S .GT. C2ONNB) THEN
                    EELPR = &
                         G1*(R1D*(ACOEF-S2D*(BCOEF+S2D*(COVER3+DOVER5*S2D))) &
                         + CONST)
                    DF = &
                         DF-G1*R1*(ACOEF*R2D+BCOEF+S2D*(CCOEF+DCOEF*S2D))
                    IF(LSECD) DDF=DDF+G1*R2*(TWOA*R3D &
                         -(TWOC+FOURD*S2D)*S1D)
                 ELSE
                    EELPR = G1*(R1D+EADD)
                    DF = DF - G1*R2D*R1
                    IF(LSECD) DDF = DDF + TWO*G1*R3D*R2
                 ENDIF
                 !-----------------------------------------------------------------------
              ELSE
                 CALL WRNDIE(-5,'<EVDW_MM>', &
                      'Unsupported ELEC/MMFF option')
                 return
              ENDIF
              !-----------------------------------------------------------------------
              !             ENDIF
           ENDIF
        ENDIF
        !
        IF(LUSED) THEN
           !
#if KEY_BLOCK==1
           IF (QBLOCK) THEN
              IBL = IBLCKP(I)
              JBL = IBLCKP(J)
              IF (JBL .LT. IBL) THEN
                 KK=IBL
                 IBL=JBL
                 JBL=KK
              ENDIF
              KK=IBL+JBL*(JBL-1)/2
              COEF = BLCOEP(KK)
              ENBPR=ENBPR*COEF
              EELPR=EELPR*COEF
              DF=DF*COEF
              IF (LSECD) DDF=DDF*COEF
           ENDIF
           !
           IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
              DXIT=DXI*DF
              DYIT=DYI*DF
              DZIT=DZI*DF
              FDXI=FDXI+DXIT
              FDYI=FDYI+DYIT
              FDZI=FDZI+DZIT
              DX(J)=DX(J)-DXIT
              DY(J)=DY(J)-DYIT
              DZ(J)=DZ(J)-DZIT
              !     TO CALCULATE-SECOND-DERIVATIVES
              IF(LSECD) THEN
                 !
                 DDF=DDF-DF*R2
                 !
                 !     NOW UPDATE DERIVATIVE MATRICIES
                 !
                 AXX=DXI*DXI*DDF+DF
                 AYY=DYI*DYI*DDF+DF
                 AZZ=DZI*DZI*DDF+DF
                 AXY=DXI*DYI*DDF
                 AXZ=DXI*DZI*DDF
                 AYZ=DYI*DZI*DDF
                 !
                 II=3*I-2
                 JJ=3*J-2
                 !
                 IADD=IUPT(II)+II
                 DD1(IADD)=DD1(IADD)+AXX
                 IADD=IUPT(II+1)+II+1
                 DD1(IADD)=DD1(IADD)+AYY
                 IADD=IUPT(II+2)+II+2
                 DD1(IADD)=DD1(IADD)+AZZ
                 IADD=IUPT(II)+II+1
                 DD1(IADD)=DD1(IADD)+AXY
                 IADD=IUPT(II)+II+2
                 DD1(IADD)=DD1(IADD)+AXZ
                 IADD=IUPT(II+1)+II+2
                 DD1(IADD)=DD1(IADD)+AYZ
                 !
                 IADD=IUPT(JJ)+JJ
                 DD1(IADD)=DD1(IADD)+AXX
                 IADD=IUPT(JJ+1)+JJ+1
                 DD1(IADD)=DD1(IADD)+AYY
                 IADD=IUPT(JJ+2)+JJ+2
                 DD1(IADD)=DD1(IADD)+AZZ
                 IADD=IUPT(JJ)+JJ+1
                 DD1(IADD)=DD1(IADD)+AXY
                 IADD=IUPT(JJ)+JJ+2
                 DD1(IADD)=DD1(IADD)+AXZ
                 IADD=IUPT(JJ+1)+JJ+2
                 DD1(IADD)=DD1(IADD)+AYZ
                 !
                 IF (JJ.LT.II) THEN
                    IADD=IUPT(JJ)+II
                    DD1(IADD)=DD1(IADD)-AXX
                    IADD=IUPT(JJ+1)+II+1
                    DD1(IADD)=DD1(IADD)-AYY
                    IADD=IUPT(JJ+2)+II+2
                    DD1(IADD)=DD1(IADD)-AZZ
                    IADD=IUPT(JJ)+II+1
                    DD1(IADD)=DD1(IADD)-AXY
                    IADD=IUPT(JJ+1)+II
                    DD1(IADD)=DD1(IADD)-AXY
                    IADD=IUPT(JJ)+II+2
                    DD1(IADD)=DD1(IADD)-AXZ
                    IADD=IUPT(JJ+2)+II
                    DD1(IADD)=DD1(IADD)-AXZ
                    IADD=IUPT(JJ+1)+II+2
                    DD1(IADD)=DD1(IADD)-AYZ
                    IADD=IUPT(JJ+2)+II+1
                    DD1(IADD)=DD1(IADD)-AYZ
                 ELSE
                    IADD=IUPT(II)+JJ
                    DD1(IADD)=DD1(IADD)-AXX
                    IADD=IUPT(II+1)+JJ+1
                    DD1(IADD)=DD1(IADD)-AYY
                    IADD=IUPT(II+2)+JJ+2
                    DD1(IADD)=DD1(IADD)-AZZ
                    IADD=IUPT(II+1)+JJ
                    DD1(IADD)=DD1(IADD)-AXY
                    IADD=IUPT(II)+JJ+1
                    DD1(IADD)=DD1(IADD)-AXY
                    IADD=IUPT(II+2)+JJ
                    DD1(IADD)=DD1(IADD)-AXZ
                    IADD=IUPT(II)+JJ+2
                    DD1(IADD)=DD1(IADD)-AXZ
                    IADD=IUPT(II+2)+JJ+1
                    DD1(IADD)=DD1(IADD)-AYZ
                    IADD=IUPT(II+1)+JJ+2
                    DD1(IADD)=DD1(IADD)-AYZ
                 ENDIF
                 !
              ENDIF
#if KEY_BLOCK==1
           ENDIF
#endif /*  BLOCK*/
           ETEMP1=ETEMP1+ENBPR
           ETEMP2=ETEMP2+EELPR
           IF(QECONT) THEN
              !
              if(QECONT .and. prnlev.ge.6 .and.NBPRINT.lt.1000) then
                 NBPRINT=NBPRINT+1
                 write(outu,'(1X,A,1X,A,F7.3,4F9.3,2F7.3)') &
                      QNAME(I),QNAME(J),SQRT(S),ENBPR,EREP,EATT,EELPR, &
                      CNBA(IC),CNBA(IC)
              endif
              !
              S=HALF*(ENBPR+EELPR)
              ECONT(I)=ECONT(I)+S
              ECONT(J)=ECONT(J)+S
           ENDIF
        ENDIF ! IF(LUSED) THEN
     enddo loop20
     !
     !     RESTORE ITH COMPONENT OF FORCE IN THE ARRAY
     !
#if KEY_BLOCK==1
     IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
        DX(I)=FDXI
        DY(I)=FDYI
        DZ(I)=FDZI
#if KEY_BLOCK==1
     ENDIF
#endif /*  BLOCK*/
     ENB=ENB+ETEMP1
     EEL=EEL+ETEMP2
  enddo loop40
  !
  return
end SUBROUTINE EVDW_MM

#endif 
SUBROUTINE NULL_enbsMM
  RETURN
END SUBROUTINE NULL_enbsMM

