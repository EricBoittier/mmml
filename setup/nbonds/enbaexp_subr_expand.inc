#undef PASTE_
#undef ADD_B_SUFFIX
#undef ADD_E_SUFFIX
#undef ADD_V_SUFFIX

#define PASTE_(a) a
#define ADD_B_SUFFIX(name) PASTE_(name)ENBAEXP_B_SUFFIX
#define ADD_E_SUFFIX(name) PASTE_(name)ENBAEXP_E_SUFFIX
#define ADD_V_SUFFIX(name) PASTE_(name)ENBAEXP_V_SUFFIX

SUBROUTINE ADD_B_SUFFIX(ADD_E_SUFFIX(ADD_V_SUFFIX(ENBA))) &
    (ENBX,EELX,ELECFG,FIRST,NATOMX, &
     CGX,JNBL,INBL, &
     XCCNBA,XCCNBB,XCCNBC,XCCNBD, &
#if ENBAEXP_SUBR_E0==1
     LEWLD,CSHIFT,CFSWIT,RSHFT,GESWIT,  & 
#endif
#if ENBAEXP_SUBR_V0==1
     LVSW,LVFSW,LVSH,GVSWIT,DBEXP,            & 
#endif
#if KEY_NBIPS==1
     LVIPSX,LEIPSX,DBEXPIPS,                     & 
#endif
     IACNBX)
  !----------------------------------------------------------------------
  ! Expanded routines called by ENBAEXP
  !----------------------------------------------------------------------
  use ewald_1m,only:lewald,kappa
  use erfcd_mod

  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use coord
  use deriv
  use param
  use inbnd
  use fast
#if KEY_NBIPS==1
  use nbips        
#endif
#if ENBAEXP_SUBR_BLOCK==1
  use block_fcm        
  use lambdam           /*ldm*/
#endif
  use trunk
  !
  ! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#endif 
  !
  implicit none
  real(chm_real)  ENBX,EELX
  LOGICAL ELECFG
  INTEGER FIRST,NATOMX,JNBL(*),INBL(*)
  real(chm_real)  CGX(*)
  real(chm_real)  XCCNBA(*),XCCNBB(*),XCCNBC(*),XCCNBD(*)
#if ENBAEXP_SUBR_E0==1
  LOGICAL LEWLD,CSHIFT,CFSWIT,RSHFT,GESWIT       
#endif
#if ENBAEXP_SUBR_V0==1
  LOGICAL LVSW,LVFSW,LVSH,GVSWIT,DBEXP                 
#endif
  INTEGER IACNBX(*)
  !
#if KEY_NBIPS==1 /*nbips_comm*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  LOGICAL LEIPSX,LVIPSX,DBEXPIPS,DOVIPS,DOEIPS
  real(chm_real) U1,U2,U4,U6R,U12R
  real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA
  real(chm_real) ENEP,ENEVC,ENEVA,ENBC
#endif /*   (nbips_comm)*/
#if ENBAEXP_SUBR_BLOCK==1 || ENBAEXP_SUBR_LEWHYB==1
  INTEGER IBL,JBL,KK
#endif /*  BLOCK*/
#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1
  real(chm_real)  E14FM1,E14F,RS,R1S,ERFC2,ERFCX,DRFC,DELKAP
  real(chm_real)  XVAL,REM,VAL0,VAL1,VAL2,D1,D2
  INTEGER IXVAL
#endif 
#if ENBAEXP_SUBR_LEWHYB==1
  !.ab.HybH: E8 familly of routines.
  !.ab. Note: trunk.f90: table for truncation scheme. Quite rigid,
  !.ab. but fast. Could use heap instead or use log formula,
  !.ab. which would be better at the end, but still under development.
  !.ab. Contact A.Blondel.
  LOGICAL QOUT
  INTEGER IDXM,PT,ID,J
  real(chm_real) R02,CZZ,A,B,L6THR,L6THP,DL112R,DL112P
  real(chm_real) FR,FP,DFR,DFP,DNNR,DNER,DNNP,DNEP
  !.ab.
#endif 
  INTEGER IVECT,JVECT,KVECT
  real(chm_real) CA,CC,CH,ENE,ENN,ENB,EEL
  real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
  real(chm_real) TFELEC,TFVDW
  real(chm_real) S2,TR2,TR6,FSW,DFSW,FSH
  real(chm_real) EADD,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6,R1,R3, &
       DENOM, &
       ACOEF,BCOEF,CCOEF,DCOEF,COVER3,DOVER5,CONST,ENEVDW
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       CR6,CR12,RJUNK3,RJUNK6,MIN2OF
  !
  real(chm_real) SIG,DFA,DFC,EMIN,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC
  !
  real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
  real(chm_real) CGF,CGT,CRXI,CRYI,CRZI
  INTEGER ITEMP,I,II,JJ,NPR,IACI
  real(chm_real) CG2ONNB,CG2OFNB,ALPHA,AG,BG,CG,G2,G3,DG2,DG3,RMCGON
  real(chm_real) A6,B6,C6,G62,DG62,G63,DG63
  real(chm_real) RMCTON,rmcton2,rmcton3,rmcton4
  real(chm_real) A12,B12,C12,G122,DG122,G123,DG123
  real(chm_real) TR3,TR7,TR12,TR13,TR14
  real(chm_real) cofcon,cofcon2,cofcon3,cofcon4,cof6,cof8,cof14
  real(chm_real) cgofcgon,cgof3,cgofcgon2,cgofcgon3,rmcgon2,rmcgon3
  !
#if ENBAEXP_SUBR_E8==1
  !write(*,*) 'MGL> GESWIT' 
#endif
#if ENBAEXP_SUBR_V6==1
  !write(*,*) 'MGL> GVSWIT' 
#endif
  !
  CGF=ZERO
  IF(ELECFG) CGF=CCELEC/EPS
  C2OFNB=CTOFNB*CTOFNB
  !
#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1 /*lewald_init*/
#if ENBAEXP_SUBR_E0==1
  IF(LEWLD) THEN                      
#endif
     DELKAP = KAPPA*EWRDEL
     E14FM1 = E14FAC - ONE
#if ENBAEXP_SUBR_E0==1
  ENDIF                               
#endif
#endif /* (lewald_init)*/
  !
#if ENBAEXP_SUBR_CSHIFT==1 /*cshift_init*/
#if ENBAEXP_SUBR_E0==1
  IF (CSHIFT) THEN                    
#endif
     MIN2OF = MINTWO/CTOFNB
     CTROF2=-ONE/C2OFNB
#if ENBAEXP_SUBR_E0==1
  ENDIF                               
#endif
#endif /* (cshift_init)*/
  !
#if ENBAEXP_SUBR_GESWIT==1 /*geswit_init*/
#if ENBAEXP_SUBR_E0==1
      IF (GESWIT) THEN          
#endif
         
!     ! cutoffs must be set here so that S2 < C2OFNB, etc. can be done
!     ! later.
         cgof3 = cgofnb**3
         CG2ONNB = CGONNB * CGONNB
         CG2OFNB = CGOFNB * CGOFNB
         cgofcgon = cgofnb - cgonnb
         cgofcgon2 = cgofcgon*cgofcgon
         cgofcgon3 = cgofcgon2*cgofcgon
     
#if ENBAEXP_SUBR_E0==1
      ENDIF                     
#endif
#endif /* (geswit_init)*/
!
#if ENBAEXP_SUBR_CFSWIT==1 /*cfswit_init*/
#if ENBAEXP_SUBR_E0==1
  IF (CFSWIT) THEN                    
#endif
     !       force-based cdie switching coeffs
     C2ONNB=CTONNB*CTONNB
     IF(CTONNB  <  CTOFNB) THEN
        ONOFF2 = C2ONNB*C2OFNB
        ON3    = C2ONNB*CTONNB
        OFF3   = C2OFNB*CTOFNB
        OFF4   = C2OFNB*C2OFNB
        OFF5   = OFF3*C2OFNB
        DENOM  = ONE/(C2OFNB-C2ONNB)**3
        EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
             EIGHT*DENOM
        ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
        BCOEF  = SIX*ONOFF2*DENOM
        COVER3 = -(C2ONNB+C2OFNB)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        DOVER5 = DCOEF/FIVE
        CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
     ELSE
        EADD  = -ONE/CTOFNB
     ENDIF
#if ENBAEXP_SUBR_E0==1
  ENDIF                               
#endif
#endif /* (cfswit_init)*/
  !
#if ENBAEXP_SUBR_RSHFT==1 /*rshft_init*/
#if ENBAEXP_SUBR_E0==1
  IF (RSHFT) THEN                     
#endif
     CTROF2=-ONE/C2OFNB
     C4ROF2=FOUR*CTROF2
#if ENBAEXP_SUBR_E0==1
  ENDIF                               
#endif
#endif /* (rshft_init)*/
  !
#if ENBAEXP_SUBR_VSWIT==1 /*switch_init*/
#if ENBAEXP_SUBR_V0==1
  IF (LVSW) THEN                      
#endif
     C2ONNB=CTONNB*CTONNB
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=TWELVE*RUL3
     ENDIF
#if ENBAEXP_SUBR_V0==1
  ENDIF                               
#endif
#endif /* (switch_init)*/
  !
#if ENBAEXP_SUBR_GVSWIT==1 /*gvswit_init*/
#if ENBAEXP_SUBR_V0==1
      IF (GVSWIT) THEN          
#endif
         C2ONNB=CTONNB*CTONNB 
         cofcon = ctofnb - ctonnb 
         cofcon2 = cofcon*cofcon 
         cofcon3 = cofcon2*cofcon 
         cofcon4 = cofcon2*cofcon2 
          
         cof6 = ctofnb**6 
         cof8 = cof6*ctofnb*ctofnb 
         cof14 = cof6*cof8 
          
!      !cof8 = ctofnb**8 
!      !cof14 = cof8*ctofnb*ctofnb*ctofnb*ctofnb*ctofnb*ctofnb 
#if ENBAEXP_SUBR_V0==1
      ENDIF                     
#endif
#endif /* (gvswit_init) */
  !
#if ENBAEXP_SUBR_VFSWIT==1 /*vfswit_init*/
#if ENBAEXP_SUBR_V0==1
  IF (LVFSW) THEN                     
#endif
     C2ONNB=CTONNB*CTONNB
     OFF3 = C2OFNB*CTOFNB
     OFF6 = OFF3*OFF3
     RECOF6 = ONE/OFF6
     IF(CTONNB  <  CTOFNB) THEN
        ON3 = C2ONNB*CTONNB
        ON6 = ON3*ON3
        RECOF3 = ONE/OFF3
        OFDIF6 = OFF6/(OFF6 - ON6)
        OFDIF3 = OFF3/(OFF3 - ON3)
        ONOFF6 = RECOF6/ON6
        ONOFF3 = RECOF3/ON3
     ELSE
        ONOFF6 = RECOF6*RECOF6
        ONOFF3 = RECOF6
     ENDIF
#if ENBAEXP_SUBR_V0==1
  ENDIF                               
#endif
#endif /* (vfswit_init)*/
  !
  ENB=ZERO
  EEL=ZERO

  !.ab.Coefs, tables... for HYBH.
#if ENBAEXP_SUBR_LEWHYB==1 /*lewhyb-init*/
  !yw      IF (.NOT.QHYBH) CALL WRNDIE(-5,'<ENERGY>'
  !yw     $     ,'Should use E8 routines with HybH')
#if ENBAEXP_SUBR_E0==1
  IF (QHYBH) THEN                               
#endif
#if KEY_SOFTVDW==1
     IF(QGAMIN) CALL WRNDIE(-5,'<ENERGY>',        & 
#endif
#if KEY_SOFTVDW==1
          'HYBH and SOFTVDW incompatible.')      
#endif
     QBLOCK=.FALSE.
     DNNR=ZERO
     DNER=ZERO
     DNNP=ZERO
     DNEP=ZERO
     FR=(THREE*HYBHLB-FOUR)*(HYBHLB-ONE)/FOUR
     DFR=(SIX*HYBHLB-SEVEN)/FOUR
     FP=(THREE*HYBHLB+ONE)*HYBHLB/FOUR
     DFP=(SIX*HYBHLB+ONE)/FOUR
     IF (HYBHLB == ZERO) THEN
        L6THR=ZERO
        L6THP=ONE
        DL112R=ZERO
        DL112P=MINONE/TWELVE
     ELSE IF (HYBHLB == ONE) THEN
        L6THR=ONE
        L6THP=ZERO
        DL112R=ONE/TWELVE
        DL112P=ZERO
     ELSE
        L6THR=HYBHLB**SIXTH
        L6THP=(ONE-HYBHLB)**SIXTH
        DL112R=ONE/HYBHLB/TWELVE
        DL112P=MINONE/(ONE-HYBHLB)/TWELVE
     ENDIF
     !.ab. Make table for Indeces
     IF (PTABLE < NATOMX) THEN
        IDXM=0
        DO I=1,NATOMX
           DO J=I-1,1,-1
              IF ((IACNB(J) == IACNB(I)).AND.(CGX(I).EQ.CGX(J))) &
                   IDX(I)=IDX(J)
           ENDDO
           IF (IDX(I) == -1) THEN
              IDXM=IDXM+1
              IDX(I)=IDXM
           ENDIF
        ENDDO
        !            IF(IOLEV >= 0)WRITE(OUTU,'(i5,a20)') IDXM,
        !     $           ' Atom types for HYBH'
        IF ((IDXM+1)*IDXM > MAXPAI) CALL WRNDIE(-6,'<ENBFAST>', &
             'Too many atom types, Increase MAXPAI.')
        PTABLE=NATOMX
     ENDIF
#if ENBAEXP_SUBR_E0==1
  ENDIF                                     
#endif
  !.ab.
#endif /* (lewhyb-init)*/

  ! Do vdw and electrostatic terms as requested
  !
  ITEMP=0
  IF(FIRST > 1) ITEMP=INBL(FIRST-1)
  !
  atomloop: DO I=FIRST,NATOMX
#if KEY_IMCUBES==1
     IF(LBYCBIM) ITEMP=INBL(I+NATOMX)        
#endif
     NPR=INBL(I)-ITEMP
     npr_gt_0:IF(NPR > 0) THEN
        IACI=IACNBX(I)
        CGT=CGF*CGX(I)
        CRXI=X(I)
        CRYI=Y(I)
        CRZI=Z(I)
        DTX=ZERO
        DTY=ZERO
        DTZ=ZERO
#if ENBAEXP_SUBR_LEWHYB==1
        !.ab.
#if ENBAEXP_SUBR_E0==1
        IF (QHYBH) THEN                       
#endif
           IBL=IBLCKP(I)
#if ENBAEXP_SUBR_E0==1
        ENDIF                                 
#endif
        !.ab.
#endif 

        !
        jjloop: DO JJ=1,NPR
           KVECT=JNBL(ITEMP+JJ)
           JVECT=ABS(KVECT)
           ! namkh 01/20/04
           ! MH-JUL06: there was a problem here without QGMREM:
           !           IGMSEL(I) is not defined [GRRR] unless a quantum
           !           command is in the input script
           !           This error is fixed in many places:
           !             enbfast,enbfs8p,enbonda,enbondg,ewald.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
           ! Now that we moved allocation of IGMSEL out of static memory
           ! we need additional checks here:
           ! if none of the QM methods is used then skip over the igmsel if
           if(qmused) then
           IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)) == 2) .AND. &
                (ABS(IGMSEL(JVECT)) == 1.OR.ABS(IGMSEL(JVECT)) == 2) &
                .AND.QGMREM) cycle jjloop
           endif
#endif 
           !
           TX=CRXI-X(JVECT)
           TY=CRYI-Y(JVECT)
           TZ=CRZI-Z(JVECT)
           S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
           IF(S2 < C2OFNB) THEN

#if ENBAEXP_SUBR_LEWHYB==1 /*get-r02*/
              !.ab.HybH stuf. Block and truncation scheme setup
#if ENBAEXP_SUBR_E0==1
              IF (QHYBH) THEN                          
#endif
                 !               JBL=I4VAL(IBLCKP,JVECT)
                 JBL=IBLCKP(JVECT)
                 KK=MAX(IBL,JBL)
                 KK=KK*(KK-1)/2+MIN(IBL,JBL)
                 IF (KK /= 1) THEN
                    IF (KK == 5) THEN
                       cycle jjloop
                    ELSE
                       PT=MAX(IDX(I),IDX(JVECT))
                       PT=PT*(PT-1)/2+MIN(IDX(I),IDX(JVECT))
                       ID=NDX(PT)
                       R02=R02L(ID)
                       !.ab. R02L(1)=-1. : not attributed: Give an index and calculate R02.
                       IF (R02 < 0.) THEN
                          IF( (IDX(I) == -1).OR.(IDX(JVECT) == -1) )THEN
                             CALL WRNDIE(-5,'<ENBFAST>','HYBH: IDX table misinitiated.')
                          ENDIF
                          NDXM=NDXM+1
                          NDX(PT)=NDXM
                          CZZ=CGT*CGX(JVECT)
                          !                        IVECT=IACNB(JVECT)+IACI
                          IVECT=LOWTP(MAX(IACNBX(JVECT),IACI))+ &
                               IACNBX(JVECT)+IACI
                          !                        A=CCNBA(IVECT)
                          !                        B=CCNBB(IVECT)
                          A=XCCNBA(IVECT)
                          B=XCCNBB(IVECT)
                          CALL GETR02(R02L(NDXM),CZZ,A,B, &
                               IDXM,NDXM,I,JVECT)
                          R02=R02L(NDXM)
                          ! WRITE(OUTU,'(i5,a,f12.5,2i7)') NDXM,
                          !     $ ' Pair R02: ',R02,I,JVECT
                       ENDIF
                       IF (KK < 4) THEN
                          R02=R02*L6THR
                       ELSE
                          R02=R02*L6THP
                       ENDIF
                       QOUT=.TRUE.
                       IF (S2 < R02) THEN
                          ! write(6,'(a,2i,a,2f)') 'Pair:',i,jvect,
                          !  A ' at D/A',S2,R02
                          S2=R02
                          QOUT=.FALSE.
                       ENDIF
                       ! if(ihybh == 16)write(6,*)
                       ! $ i,jvect,kk,s2,r02,qout,ibl,jbl 
                    ENDIF
                 ENDIF
                 !.KK=1, do nothing.
#if ENBAEXP_SUBR_E0==1
              ENDIF                                     
#endif
              !.ab.
#endif /* (get-r02)*/

              IVECT=LOWTP(MAX(IACNBX(JVECT),IACI))+IACNBX(JVECT)+IACI
              CH=CGT*CGX(JVECT)
#if KEY_NBIPS==1 && ENBAEXP_SUBR_EIPS==1
              DOEIPS=LEIPSX           
#endif
#if KEY_NBIPS==1 && (ENBAEXP_SUBR_VIPS==1 || ENBAEXP_SUBR_DBEXPIPS==1)
              DOVIPS=LVIPSX.OR.DBEXPIPS          
#endif
              !
              IF(KVECT < 0) THEN     
#if KEY_NBIPS==1 && ENBAEXP_SUBR_EIPS==1
                 DOEIPS=.FALSE.        
#endif
#if KEY_NBIPS==1 
#if ENBAEXP_SUBR_VIPS==1 || ENBAEXP_SUBR_DBEXPIPS==1 
                 DOVIPS=.FALSE.        
#endif
#endif
                 IVECT=IVECT+NITCC2
#if ENBAEXP_SUBR_E0==1
                 IF(LEWLD) THEN                 
#endif
#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1
                    E14F = E14FM1               
#endif
#if ENBAEXP_SUBR_E0==1
                 ELSE                           
#endif
#if ENBAEXP_SUBR_CSHIFT==1 || ENBAEXP_SUBR_CFSWIT==1 || ENBAEXP_SUBR_RSHFT==1 
                    CH=CH*E14FAC                
#endif
#if ENBAEXP_SUBR_E0==1
                 ENDIF                          
#endif
#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1
              ELSE                             
#endif
#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1
                 E14F = ZERO                   
#endif
              ENDIF
              !
              TR2=ONE/S2
#if KEY_NBIPS==1 && (ENBAEXP_SUBR_EIPS==1 || ENBAEXP_SUBR_VIPS==1)
              U2=S2*RIPS2R
#endif

#if (ENBAEXP_SUBR_VFSWIT == 1 || ENBAEXP_SUBR_GVSWIT == 1||ENBAEXP_SUBR_DBEXP == 1 || ENBAEXP_SUBR_DBEXPIPS == 1) && ENBAEXP_SUBR_LKUP != 1 
              R1=SQRT(TR2)
#endif
              ! Above line used to look like:
              !R1=SQRT(TR2)                     !##vfswit ##.not.lkup

              !
              !------ Electrostatic energies (only if there are charges)
              !
              IF(CH /= ZERO) THEN

#if ENBAEXP_SUBR_LEWALD==1 || ENBAEXP_SUBR_LEWHYB==1 /*elecopt*/
                 ! cdie ewald
#if ENBAEXP_SUBR_E0==1
                 IF (LEWLD) THEN               
#endif
                    ! Inline erfc calculation for speed.
#if ENBAEXP_SUBR_VFSWIT==0
                    R1=SQRT(TR2)               
#endif
                    XVAL = S2*R1*DELKAP
                    IXVAL = XVAL+HALF
                    REM = XVAL-IXVAL
                    IXVAL=IXVAL+2
                    IXVAL = MIN(IXVAL,EWNPTS-1)
                    VAL0 = EwldT(IXVAL-1)
                    VAL1 = EwldT(IXVAL)
                    VAL2 = EwldT(IXVAL+1)
                    D1 = (VAL0-VAL2)*HALF
                    D2 = (VAL1+VAL1-VAL0-VAL2)*REM
                    ERFCX = VAL1-(D1+HALF*D2)*REM
                    DRFC = (D1+D2)*DELKAP
                    ERFC2 = ERFCX + E14F
                    ENE = CH*ERFC2*R1
                    TFELEC = (CH*DRFC + ENE)*TR2
#endif /* (elecopt)*/

#if ENBAEXP_SUBR_CSHIFT==1 /*elecopt*/
                    ! cdie original shift
#if ENBAEXP_SUBR_E0==1
                 ELSE IF (CSHIFT) THEN         
#endif
#if ENBAEXP_SUBR_VFSWIT==0
                    R1=SQRT(TR2)               
#endif
                    CH=CH*R1
                    ENE=CH*(ONE + S2*(MIN2OF*R1-CTROF2))
                    TFELEC= CH*(CTROF2 + TR2)
#endif /* (elecopt)*/
#if ENBAEXP_SUBR_GESWIT==1 /*elecopt*/
#if ENBAEXP_SUBR_E0==1
                 ELSE IF (GESWIT) THEN 
#endif

! If you're reading this, you're probably a REAL
! PROGRAMMER who believes that comments are for
! the weak. Not me, though. If you're implementing
! a new switch/shift function, you might find the
! following useful:
!
! A note on some differences between this and the
! slow routines:
!
! S from the slow routines is S2 here, both being
!     the square of the vector between the two atoms.
! R1 == 1/S in both.
! R2 == R1*R1 in the slow routines. That's TR2 here.
! R3 typically becomes R1*TR2 here.
!
! for charges in the slow routines, CG(I), CG(J)
! are the charges. CGT is CGF*CG(I).  CGF =
! CCELEC/EPS.  for the fast routines, we CGX(I),
! CGX(J), CH instead of CGT.
!
! for the electrostatics, we find CNBA = VDW
! distance squared, CNBB = VDW depth. In the fast
! routines, we have CCNBA, CCNBB
!
! CGT2*CG(J) in the slow routines is CH here,
!     which is CGT*CGX(JVECT). That's the same
!     thing, because CGT2 is just CGT corrected
!     for E14FAC.
!
! TFELEC is the incremental contribution to the
! total force due to electrostatics. almost. it's
! really 1/r times the derivative of the
! potential.
!
! ENE is the incremental contribution to the total
! energy due to electrostatics. This is EELPR in
! the slow routine.
!
! Similarly, ENN and ENBPR for vdw.
!
! TFVDW is the incremental contribution to the
! total force due to the van der
! waals. almost. it's really 1/r times the
! derivative of the potential. It's also
! multiplied by -1. this is DF in the slow
! routines.
!
! en is the incremental contribution to the total
! energy due to van der waals

! write(*,*) 'MGL> ELECTROSTATICS',CGONNB,'-->',CGOFNB
                    R1=SQRT(TR2)
                    AG= (2.0*CGONNB -5.0*CGOFNB) / (cgof3*cgofcgon2)
                    BG= -(2.0*CGONNB - 4.0*CGOFNB) / (cgof3*cgofcgon3)
                    CG= 1.0/CGOFNB -cgofcgon3*AG/3.0 & 
                         -cgofcgon2*cgofcgon2*BG/4.0
                    IF (S2  >  CG2ONNB) THEN
                       RMCGON = SQRT(S2) - CGONNB
                       rmcgon2 = rmcgon*rmcgon
                       rmcgon3 = rmcgon2*rmcgon
                       G2=(AG/3.0)*RMCGON3
                       DG2=AG*RMCGON2
                       G3=(BG/4.0)*RMCGON2*rmcgon2
                       DG3=BG*RMCGON3
                       ENE=     CH*(R1-G2-G3-CG)
                       TFELEC= -CH*R1*(-TR2-DG2-DG3)
! You could try these for when we require
! CGONNB == 0, but it does not seem to make
! much difference in practice. 
! If you use it, you do not need the IF S2 > CG2ONNB.
! S = 1/R1
! TRC = 1/CGOFNB
! TRC4 = TRC**4
! TRC5 = TRC4*TRC
! ENE = CH*(R1 - (5.0/3.0)*(TRC + S2*S*TRC4) - S2*S2*TRC5)
! TFELEC = CH*R1*(TR2 - 5*S2*TRC4 + 4*S2*S*TRC5)
! TFELEC = CH*R1*TR2 - CH*5*S*TRC4 + CH*4*S2*TRC5
                    ELSE
                       ENE=    CH*(R1-CG)
                       TFELEC= CH*R1*TR2
                    ENDIF
                     
#endif /* (elecopt)*/

#if ENBAEXP_SUBR_CFSWIT==1 /*elecopt*/
                    ! cdie force switch
#if ENBAEXP_SUBR_E0==1
                 ELSE IF (CFSWIT) THEN         
#endif
#if ENBAEXP_SUBR_VFSWIT==0
                    R1=SQRT(TR2)               
#endif
                    IF (S2 > C2ONNB) THEN
                       ENE = CH*( R1* (ACOEF - S2*(BCOEF + S2*(COVER3 &
                            + DOVER5*S2))) + CONST)
                       TFELEC = CH*R1*( ACOEF*TR2 + BCOEF + &
                            S2*(CCOEF + DCOEF*S2) )
                    ELSE
                       ENE = CH*(R1+EADD)
                       TFELEC = CH*R1*TR2
                    ENDIF
#endif /* (elecopt)*/
                    !
#if ENBAEXP_SUBR_RSHFT==1 /*elecopt*/
                    ! rdie original shift
#if ENBAEXP_SUBR_E0==1
                 ELSE IF (RSHFT) THEN                      
#endif
                    FSH=ONE+S2*CTROF2
                    CH=CH*TR2*FSH
                    ENE=CH*FSH
                    TFELEC= TWO*ENE*TR2-C4ROF2*CH
#endif /* (elecopt)*/

#if ENBAEXP_SUBR_EIPS==1 /*elecopt*/
#if KEY_NBIPS==1 /*nbips_eips*/
                    ! ips for electrostatic
#if ENBAEXP_SUBR_E0==1
                 ELSE IF (DOEIPS) THEN                      
#endif
#if ENBAEXP_SUBR_E7==1
                    IF (DOEIPS) THEN                   
#endif
                       U1=SQRT(U2)
                       PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                            +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                       DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                            +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                       ENEP=CH*RIPSR
                       ENE=ENEP*(PE-PIPSEC)
                       TFELEC=-ENEP*DPE*TR2
#if ENBAEXP_SUBR_E7==1 /*e7*/
                    ELSE                                 
                       ENE=ZERO                            
                       TFELEC=ZERO                            
                    ENDIF
#endif /*  (e7)*/
#endif /* (nbips_eips)*/
#endif /* (elecopt)*/
                    !
#if ENBAEXP_SUBR_NOELEC==1 /*elecopt*/
#if ENBAEXP_SUBR_E0==1
                 ELSE                                      
#endif
                    ENE=ZERO
                    TFELEC = ZERO
#endif /* (elecopt)*/
#if ENBAEXP_SUBR_E0==1
                 ENDIF                                     
#endif
              ELSE 
                 ENE=ZERO
                 TFELEC = ZERO
              ENDIF
              !
              !------ End of Electrostatic energies
              !
              !------ VDW energies
              !
#if ENBAEXP_SUBR_VSHFT==1 /*vdwopt*/
              ! vdw shift
#if ENBAEXP_SUBR_V0==1
              IF (LVSH) THEN                         
#endif
                 TR6=TR2*TR2*TR2
                 CA=XCCNBA(IVECT)*TR6*TR6
                 ENEVDW = CA-XCCNBB(IVECT)*TR6
                 CC=S2*S2*S2*XCCNBC(IVECT)
                 ENN=ENEVDW-CC+XCCNBD(IVECT)
                 TFVDW=MINSIX*(ENEVDW+CA+CC)*TR2
#endif /*  (vdwopt)*/

#if ENBAEXP_SUBR_GVSWIT==1 /*vdwopt*/
#if ENBAEXP_SUBR_V0==1
              ELSE IF (GVSWIT) THEN                   
#endif
! IF (S2  <  C2OFNB) THEN !taken care above 
!                 write(*,*) 'MGL VDW',CTONNB,'-->',CTOFNB
                 rmcton = sqrt(s2) - ctonnb 
                 a6= 6.0*(7.0*ctonnb - 10.0*ctofnb)/(cof8*cofcon2) 
                 b6=-6.0*(7.0*ctonnb - 9.0*ctofnb)/(cof8*cofcon3) 
! c6= 1.0/(ctofnb**6)   - (a6/3.)*cofcon3   - (b6/4.)*cofcon4 
                 c6= 1.0/(cof6)   - (a6/3.)*cofcon3   - (b6/4.)*cofcon4 
                  
                 a12= 12.0*(13.0*ctonnb - 16.0*ctofnb)/(cof14*cofcon2) 
                 b12=-12.0*(13.0*ctonnb - 15.0*ctofnb)/(cof14*cofcon3) 
! c12= 1.0/(ctofnb**12) - (a12/3.)*cofcon3 - (b12/4.)*cofcon4 
                 c12= 1.0/(cof6*cof6) - (a12/3.)*cofcon3 - &
                      (b12/4.)*cofcon4 
                  
                 tr3 = r1*tr2 
                 tr6 = tr3*tr3 
                 tr12 = tr6*tr6 
                 if (s2 > c2onnb) then 
                    rmcton2 = rmcton*rmcton 
                    rmcton3 = rmcton2*rmcton 
                    rmcton4 = rmcton2*rmcton2 
                     
                    enn =  xccnba(ivect)*(tr12-rmcton3*a12/3.0-rmcton4* &
                         b12/4.0-c12) -  &
                         xccnbb(ivect)*(tr6 -rmcton3*a6/3.0  &
                         -rmcton4*b6/4.0 -c6) 
                    tfvdw = xccnba(ivect)*r1*(-12.0*tr12*r1-a12*rmcton2 &
                         -b12*rmcton3) - &
                         xccnbb(ivect)*r1*( -6.0*tr6*r1 -a6*rmcton2  &
                         -b6*rmcton3) 
                    
! g62=rmcton3*a6/3.0 
! dg62=a6*rmcton2 
! g63=rmcton4*b6/4.0 
! dg63=b6*rmcton3 
! g122=rmcton3*a12/3.0 
! dg122=a12*rmcton2 
! g123=rmcton4*b12/4.0 
! dg123=b12*rmcton3 

! tr7 = tr6*r1 
! tr13 = tr12*r1 
! enn = xccnba(ivect)*(tr12-g122-g123-c12) - xccnbb(ivect)*(tr6 -g62 -g63 -c6) 
! tfvdw = xccnba(ivect)*r1*(-12.0*tr13-dg122-dg123) - & 
!     xccnbb(ivect)*r1*( -6.0*tr7 -dg62 -dg63) 
                 else 
                    ca = xccnba(ivect)*tr12 
                    enevdw= ca  -xccnbb(ivect)*tr6 
                    enn = enevdw - xccnba(ivect)*c12 +&
                         xccnbb(ivect)*c6 
                    tfvdw=minsix*tr2*(enevdw+ca) 
                 ENDIF 
!                 !  ! ENDIF ! taken care of above 
 
#endif /* (vdwopt) */

!

                 !
#if ENBAEXP_SUBR_VFSWIT==1 /*vdwopt*/
                 ! vdw force switch
#if ENBAEXP_SUBR_V0==1
              ELSE IF(LVFSW) THEN                    
#endif
                 TR6=TR2*TR2*TR2
                 IF (S2 > C2ONNB) THEN
                    R3 = R1*TR2
                    RJUNK6 = TR6-RECOF6
                    RJUNK3 = R3-RECOF3
                    CR12 = XCCNBA(IVECT)*OFDIF6*RJUNK6
                    CR6  = XCCNBB(IVECT)*OFDIF3*RJUNK3
                    ENN = CR12*RJUNK6 - CR6*RJUNK3
                    TFVDW = TR2*(SIX*CR6*R3 - TWELVE*CR12*TR6)
                 ELSE
                    CA=XCCNBA(IVECT)*TR6*TR6
                    ENEVDW = CA-XCCNBB(IVECT)*TR6
                    ENN = ENEVDW+XCCNBB(IVECT)*ONOFF3 - &
                         XCCNBA(IVECT)*ONOFF6
                    TFVDW = MINSIX*TR2*(ENEVDW+CA)
                 ENDIF
#endif /*  (vdwopt)*/
                 !
#if ENBAEXP_SUBR_VSWIT==1 /*vdwopt*/
                 ! vdw switch
#if ENBAEXP_SUBR_V0==1
              ELSE IF(LVSW) THEN                     
#endif
                 TR6=TR2*TR2*TR2
                 CA=XCCNBA(IVECT)*TR6*TR6
                 IF (S2 > C2ONNB) THEN
                    ENEVDW = CA-XCCNBB(IVECT)*TR6
                    RIJL=C2ONNB-S2
                    RIJU=C2OFNB-S2
                    FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFSW=RIJL*RIJU*RUL12
                    ENN=ENEVDW*FSW
                    TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                 ELSE
                    ENN = CA-XCCNBB(IVECT)*TR6
                    TFVDW =  MINSIX*TR2*(ENN+CA)
                 ENDIF
#endif /*  (vdwopt)*/
                 !
#if ENBAEXP_SUBR_DBEXP==1 /*vdwopt*/
#if ENBAEXP_SUBR_V0==1
              ELSE IF(DBEXP) THEN           
#endif
                 SIG=XCCNBB(IVECT)
                 IF(SIG>RSMALL)THEN
                  EMIN=SIG*SIG/XCCNBA(IVECT)/FOUR
                  SIG=(HALF*SIG/EMIN)**SIXTH
#if KEY_NBIPS == 1
                  AKR=AIPSVA(0)/R1/SIG
                  EXPAR=EXP(-AKR)
                  ! energy
                  ENEVA=AIPSVA(1)*EXPAR
                  ! force 
                  DPVA=BIPSVA(1)*EXPAR/SIG
                  !
                  CKR=AIPSVC(0)/R1/SIG
                  EXPCR=EXP(-CKR)
                  ! energy
                  ENEVC=AIPSVC(1)*EXPCR
                  ! force
                  DPVC=BIPSVC(1)*EXPCR/SIG
                  ENN=EMIN*(ENEVA-ENEVC)
                  TFVDW=EMIN*(DPVA-DPVC)*R1
#endif /* KEY_NBIPS */
                ELSE                                 
                  ENN=ZERO                            
                  TFVDW=ZERO                            
                ENDIF
#endif /*  (vdwopt)*/
#if KEY_NBIPS==1 /*nbips_vips*/
#if ENBAEXP_SUBR_VIPS==1 /*vdwopt*/
                 !  IPS  for vdw energy
#if ENBAEXP_SUBR_V0==1
              ELSE IF(LVIPSX) THEN           
#endif
#if ENBAEXP_SUBR_V5==1
                 IF(DOVIPS) THEN                
#endif
                    U4=U2*U2
                    U6R=ONE/U4/U2
                    U12R=U6R*U6R
                    !  L-J r6 term
                    !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                    !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                    !
                    PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                         +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))
                    DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                         +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                    !  L-J r12 term 
                    !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                    !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                    !
                    PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                         +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))
                    DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                         +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                    ENEVA =XCCNBA(IVECT)*RIPS12R
                    ENEVC =-XCCNBB(IVECT)*RIPS6R
                    ENN=ENEVA*(PVA-PIPSVAC)+ENEVC*(PVC-PIPSVCC)
                    TFVDW=(ENEVA*DPVA+ENEVC*DPVC)*TR2
#if ENBAEXP_SUBR_V5==1 /*v5*/
                 ELSE                                 
                    ENN=ZERO                            
                    TFVDW=ZERO                            
                 ENDIF
#endif /*  (v5)*/
#endif /*  (vdwopt)*/
#if ENBAEXP_SUBR_DBEXPIPS==1 /*vdwopt*/
#if ENBAEXP_SUBR_V0==1
              ELSE IF(DBEXPIPS) THEN           
#endif
                 SIG=XCCNBB(IVECT)
                  EMIN=SIG*SIG/XCCNBA(IVECT)/FOUR
                  SIG=(HALF*SIG/EMIN)**SIXTH
#if ENBAEXP_SUBR_V5B==1
                 IF(DOVIPS.AND.SRIPS*R1*SIG>ONE) THEN                
#endif
                  AKR=AIPSVA(0)/R1/SIG
                  EXPAR=EXP(-AKR)
                  !AKRC=AIPSVA(0)*RIPS/SIG
                  !EXPARC=EXP(-AKRC)
                  ! energy
                  !ENEVA=AIPSVA(1)*(EXPAR+EXPARC*(EXPARC/EXPAR+(TWELVE+SIX*EXPARC/AKRC)/AKRC/AKRC))
                  ENEVA=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                  ! force 
                  DPVA=BIPSVA(1)*(EXPAR-AIPSVA(4)/EXPAR)/SIG
                  !
                  CKR=AIPSVC(0)/R1/SIG
                  EXPCR=EXP(-CKR)
                  !CKRC=AIPSVC(0)*RIPS/SIG
                  !EXPCRC=EXP(-CKRC)
                  ! energy
                  !ENEVC=AIPSVC(1)*(EXPCR+EXPCRC*(EXPCRC/EXPCR+(TWELVE+SIX*EXPCRC/CKRC)/CKRC/CKRC))
                  ENEVC=AIPSVC(1)*(EXPCR+AIPSVC(4)/EXPCR+AIPSVC(5))
                  ! force
                  DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                  ENN=EMIN*(ENEVA-ENEVC)
                  TFVDW=EMIN*(DPVA-DPVC)*R1
#if ENBAEXP_SUBR_V5B==1 /*v5b*/
                 ELSE                                 
                    ENN=ZERO                            
                    TFVDW=ZERO                            
                 ENDIF
#endif /*  (v5b)*/
#endif /*  (vdwopt)*/
#endif /* (nbips_dbexpips)*/
#if ENBAEXP_SUBR_V0==1
              ENDIF                                  
#endif
              !
              !------ End of VDW energies
              !
#if ENBAEXP_SUBR_LEWHYB==0 /*noblock*/
#if ENBAEXP_SUBR_BLOCK==1 /*block_1*/
#if ENBAEXP_SUBR_B0==1
              IF(QBLOCK) THEN                        
#endif
!ldm
                if (qmld) then
                   call msld_nb_scale_enerforce(iblckp(i),iblckp(jvect),tfelec,ene,tfvdw,enn)
                else
! LDM
                 IBL=IBLCKP(I)
                 JBL=IBLCKP(JVECT)
                 KK=LOWTP(MAX(IBL,JBL))+IBL+JBL

                 IF(QBVSPLT) THEN
                    CA=XCCNBA(IVECT)*TR6*TR6
                    ENN = (ENN - CA)*BLCOEVA(KK) + CA*BLCOEVR(KK)
!%%AvdV-070721 bugfix
! fixed issue that Eelec was not scaled properly when using vdwa or vdwr
! e.g.: coef 1 2 1.0 vdwa 0.0 elec 0.0  !# old code: results in elec value for lambda=1
                                        !# new code: results in properly scaled elec
                    ene=ene*blcoee(kk)
!%%AvdV-070721--end
                    TFVDW = MINSIX*TR2*(CA+CA)*BLCOEVR(KK) + &
                         (TFVDW - MINSIX*TR2*(CA+CA))*BLCOEVA(KK)
                    TFELEC=TFELEC*BLCOEE(KK)
                 ELSE
                    ENE=ENE*BLCOEE(KK)
                    ENN=ENN*BLCOEV(KK)
                    TFVDW=TFVDW*BLCOEV(KK)
                    TFELEC=TFELEC*BLCOEE(KK)
                 ENDIF
                 endif                                !ldm
#if ENBAEXP_SUBR_B0==1
              ENDIF                                  
#endif
#endif /* (block_1)*/
#endif /* (noblock)*/
              !               TF=TFVDW-TFELEC
#if ENBAEXP_SUBR_LEWHYB==1
              !.ab.QHYBH code.
#if ENBAEXP_SUBR_E0==1
              IF (QHYBH) THEN                                     
#endif
                 IF (KK /= 1) THEN
                    IF (KK < 4) THEN
                       DNER=DNER+ENE*DFR
                       DNNR=DNNR+ENN*DFR
                       IF (QOUT) THEN
                          TF=TF*FR
                          ! write(6,*)'OUT ',I,JVECT,ENN,TFVDW,DL112R,R02,FR,DFR
                       ELSE
                          DNER=DNER-FR*TFELEC*R02*DL112R
                          DNNR=DNNR+FR*TFVDW *R02*DL112R
                          ! write(6,*)'IN ',I,JVECT,ENN,TFVDW,DL112R,R02,FR
                          ! TF=ZERO
                          TFELEC=ZERO
                          TFVDW =ZERO
                       ENDIF
                       ENE=ENE*FR
                       ENN=ENN*FR
                    ELSE
                       DNEP=DNEP+ENE*DFP
                       DNNP=DNNP+ENN*DFP
                       IF (QOUT) THEN
                          TF=TF*FP
                       ELSE
                          DNEP=DNEP-FP*TFELEC*R02*DL112P
                          DNNP=DNNP+FP*TFVDW *R02*DL112P
                          !                           TF=ZERO
                          TFELEC=ZERO
                          TFVDW =ZERO
                       ENDIF
                       ENE=ENE*FP
                       ENN=ENN*FP
                    ENDIF
                 ENDIF
                 !.ab. KK=1, do nothing.
#if ENBAEXP_SUBR_E0==1
              ENDIF                                     
#endif
              !.ab.
#endif 
              TF=TFVDW-TFELEC
              TX=TX*TF
              TY=TY*TF
              TZ=TZ*TF
              DTX=DTX+TX
              DTY=DTY+TY
              DTZ=DTZ+TZ
              DX(JVECT)=DX(JVECT)-TX
              DY(JVECT)=DY(JVECT)-TY
              DZ(JVECT)=DZ(JVECT)-TZ
              !
              ENB=ENB+ENN
              EEL=EEL+ENE
           ENDIF    ! IF (S2 < C2OFNB)
        ENDDO  jjloop                  ! DO JJ
        !
        !     Restore I-th component of force in the array
        DX(I)=DX(I)+DTX
        DY(I)=DY(I)+DTY
        DZ(I)=DZ(I)+DTZ
     ENDIF npr_gt_0  ! IF (NPR > 0)
     ITEMP=INBL(I)
  ENDDO atomloop
  !
  ENBX=ENB
  EELX=EEL
  !
#if ENBAEXP_SUBR_LEWHYB==1
  !.ab.Save and put back qblock.
#if ENBAEXP_SUBR_E0==1
  IF (QHYBH) THEN                                     
#endif
     CALL SUMHYB(IHYBH,DNNR,DNNP)
     CALL SUMHYB(IHYBH+1,DNER,DNEP)
     QBLOCK=.TRUE.
#if ENBAEXP_SUBR_E0==1
  ENDIF                                               
#endif
  !.ab.
#endif 
  RETURN
end subroutine ADD_B_SUFFIX(ADD_E_SUFFIX(ADD_V_SUFFIX(ENBA)))
