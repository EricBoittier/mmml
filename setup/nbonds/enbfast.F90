   SUBROUTINE ENBFS8(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
                        CGX,JNBL,INBL, &
                        IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
                        IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA,  & 
#endif
#if KEY_CHEQ==1
                        ETA,FQINV,                         & 
#endif
#if KEY_FLUCQ==1
                        QFLUC,FQCFOR,                      & 
#endif
#if KEY_WCA==1
                        LLSOFT,SCVDWCUTR,WCA,              & 
#endif
                        LUSED,CGRF)   ! GRF, Wei Chen 2015
!----------------------------------------------------------------------
!     This is the fast scalar version of the nonboned energy terms
!     for {constant dielectric} {electrostatic shifting} {vdW shifting}
!         {distance dielectric} {electrostatic switch  } {vdW switch  }
!     All combinations of these nonbond energy options are supported
!
!     LFAST=0 or LFAST=1 is required to use this routine. If LFAST=0,
!     this routine will be called only if ENBAEXP cannot meet needs.
!
!     January 11, 1990  Youngdo Won
!     October 16, 1991  Force-based methods added.  PJS
!     March,      2004  FASTENBFS8; performance optimizations; RG and SRB
!-----------------------------------------------------------------------

  use nb_module         ! has ccnba thru d
  use new_timer,only:timer_start,timer_stop,T_elecvdw  
#if KEY_CHEQ==1
  use cheq                                             
#endif
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use coord
  use deriv
  use param
  use inbnd
#if KEY_BLOCK==1
  use block_fcm         
#endif
#if KEY_BLOCK==1
  use trunk         
#endif
  use lambdam
#if KEY_PBOUND==1
  use pbound        
#endif
#if KEY_MTS==1
  use tbmts         
#endif
#if KEY_NBIPS==1
  use nbips         
#endif
  use galgor
  use gcmc
#if KEY_WCA==1
  use pert,only: lsoftcore0,lsoftcore1     
#endif
#if KEY_FASTENBFS8==1
  use bases_fcm
  use enbfs8pm
  use memory
  use stream
#endif 
!
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#endif 
      implicit none
      real(chm_real)  ENB,EEL
      LOGICAL LELECX,LVDWX
      INTEGER IFRSTA,NATOMX,JNBL(*),INBL(*)
      real(chm_real)  CGX(*)
      INTEGER IACNB(*),NITCC2,LOWTP(*)
#if KEY_FLUCQ==1
      LOGICAL QFLUC
      real(chm_real) FQCFOR(*)
#endif 
#if KEY_WCA==1
      real(chm_real) WCA(*),SCVDWCUTR
      LOGICAL LLSOFT
#endif 
      LOGICAL LUSED 
!
#if KEY_BLOCK==1
      INTEGER IBL,JBL,KK, IBLOCK(*)
      real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
!.ab.HybH Note: trunk.f90: table for truncation scheme. Quite rigid,
!.ab. but fast. Could use instead or use log formula,
!.ab. which would be better at the end, but still under development.
!.ab. Contact A.Blondel.
      LOGICAL QOUT
      INTEGER IDXM,PT,ID
      real(chm_real) R02,CZZ,A,B,L6THR,L6THP,DL112R,DL112P
      real(chm_real) FR,FP,DFR,DFP,DNNR,DNER,DNNP,DNEP
!.ab.
#if KEY_DOCK==1
      INTEGER KDOC
      real(chm_real)  DOCFI, DOCFJ
#endif 
!ldm
      real(chm_real) FALPHA
      INTEGER ILDM, JLDM
      real(chm_real) ENEORG 
! LDM
#endif /*  BLOCK*/
#if KEY_PBOUND==1 /*pbound*/
      real(chm_real) CORR
#endif /*     (pbound)*/
#if KEY_MTS==1 /* MTS*/
      real(chm_real)  RR1,RR2,RR3
      real(chm_real)  SWF,SWFE
#endif 
#if KEY_CHEQ==1
      real(chm_real) HIJ,ETA(NATOMX,*)
      LOGICAL FQINV
#endif 
!
#if KEY_NBIPS==1 /*nbips_comm*/
!WXW Long range potential using isotropic periodic sum (IPS)
      LOGICAL LEIPSX,LVIPSX,DOEIPS,DOVIPS
      real(chm_real) U1,U2,U4,U6R,U12R
      real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA
      real(chm_real) ENEP,ENEVC,ENEVA,ENBC
#endif /*   (nbips_comm)*/
      INTEGER IVECT,JVECT,KVECT
      real(chm_real) CA,CC,CH,ENE,ENN
      real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
      real(chm_real) TFELEC,TFVDW
      real(chm_real) S2,TR2,TR6,FSW,DFSW,FSH,TTP12,TTPW1,TTPW2,SWTMP
      real(chm_real) RC3,RC  ! Wei Chen 2015
      real(chm_real) EADD,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6,R1,R3, &
           DENOM, &
             ACOEF,BCOEF,CCOEF,DCOEF,COVER3,DOVER5,CONST,ENEVDW
      real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
             CR6,CR12,RJUNK3,RJUNK6,MIN2OF

      real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
      real(chm_real) CGF,CGT,CRXI,CRYI,CRZI
      INTEGER ITEMP,I,J,NPR,IACI
      LOGICAL ELECFG,LOUTER,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT, &
              SWITCH,LVSW,LVSH,LVFSW,RSHIFT,RFSWIT, &
              GESWIT,GVSWIT,CGRF
      INTEGER First
!
#if KEY_SOFTVDW==1
      INTEGER ifi,ifi1
      real(chm_real) rfimin,rdiff,s1,ediff,tfvdwn,enno
      real(chm_real) alfa, ct2, rc2,x1,emax,beta,cho,rc2o,ecut
      logical qalter,qvdw
#endif

      real(chm_real) :: emin
      
#if KEY_FASTENBFS8==1
      LOGICAL DONE
      LOGICAL ENBFS8P   ! ENBFS8 Performance optimized
      INTEGER MAXPAIRS  ! maximum number of pairs
      INTEGER SIZEINTC  ! size of the interger cache
      INTEGER SIZEREALC ! size of the real(chm_real) cache
      integer,allocatable,dimension(:,:) :: INTCACHE
      real(chm_real),allocatable,dimension(:,:) :: realCACHE
#endif 
!
#if KEY_WCA==1
      real(chm_real) EPRPL, EGLJTMP, EGLJRPL, TMP, VDWWL,RMIN6,RMIN2
#endif 
#if KEY_GCMC==1
      logical lgcmcon  
#endif
      LOGICAL DBEXP, DBEXPIPS
      real(chm_real) SIG,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC

    !---------- Sanity check -------------------------------------
      if(.not. allocated(ccnba))then
         ! How we got here without vdw table filled, who knows?
         call wrndie(-4,"ENBFS8<enbfast.src>", &
              "CCNBA not allocated")
      endif
      
      First = IFRSTA
#if KEY_GENETIC==1
      If(qGA_Ener) then
        First = Int(ENB)
      endif
#endif 

      ! Wei Chen 2015
      IF(CGRF)THEN
        RC = 1 / CTOFNB
        RC3 = RC / (CTOFNB*CTOFNB)
      ENDIF

      LUSED=.TRUE.
      ENB=ZERO
      EEL=ZERO
      ELECFG=(LELECX.AND.(EPS /= ZERO))
      IF (.NOT.(LVDWX.OR.ELECFG)) RETURN
      CGF=ZERO
      IF (ELECFG) CGF=CCELEC/EPS
!
! Set flags for electrostatics options (6 options supported)
      RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;   
#endif
      CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG &
              .AND. .NOT. LEGROM                                        &
#if KEY_NBIPS==1
              .AND. .NOT.LEIPS                
#endif
#if KEY_NBIPS==0
      ;    
#endif
!
      IF(RSHIFT .OR. RFSWIT) THEN
         LUSED=.FALSE.
         RETURN
      ENDIF
      IF(LEGROM) THEN
         LUSED=.FALSE.
         RETURN
      ENDIF
!
! No more premature returns; start timer.
      call timer_start(T_ELECVDW)       
!
      LVFSW=      LVFSWT                   .AND. LVDWX &
              .AND. .NOT. LVGROM.and. .not. LDBEXP 
      LVSH = .NOT.LVFSWT .AND.      LVSHFT .AND. LVDWX &
              .AND. .NOT. LVGROM.and. .not. LDBEXP 
      LVSW = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. LVDWX &
              .AND. .NOT. LVGROM.and. .not. LDBEXP 
      IF(LVGROM) THEN
         LUSED=.FALSE.
         RETURN
      ENDIF
#if KEY_NBIPS==1
      IF(LVIPS) THEN
        LVFSW=.FALSE.
        LVSH =.FALSE.
        LVSW =.FALSE.
      ENDIF
      LEIPSX=LEIPS.AND.ELECFG
      LVIPSX=LVIPS.AND.LVDWX .and. .not. LDBEXP 

      DBEXP = LDBEXP.AND.LVDWX .and. .not. LVIPS
      DBEXPIPS = LDBEXP.AND.LVDWX .and. LVIPS
#endif 

      SWITCH= RSWIT.OR.CSWIT.OR.LVSW

#if KEY_FASTENBFS8==1
#if KEY_NBIPS==1
      !IF(QIPS) GOTO 242                 
#endif
! Cache storage for the Performance optimized routine ENBFS8P
      MAXPAIRS = GETMAXNPR(NATOMX,INBL)
      IF(PRNLEV > 8) WRITE(OUTU,235) MAXPAIRS
!     8 is >6, but otherwise is a random number
 235  FORMAT(' ENBOND: The maximum number of pairs is ',I10)

      call chmalloc('enbfast.src','ENBFS8','INTCACHE',intcrecsz,maxpairs,intg=INTCACHE)
      call chmalloc('enbfast.src','ENBFS8','realCACHE',realcrecsz,maxpairs,crl=realCACHE)
      DONE =  ENBFS8P(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
                        CGX,JNBL,INBL,CCNBA,CCNBB,CCNBC,CCNBD, &
                        IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
                        IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA,  & 
#endif
#if KEY_CHEQ==1
                        ETA,FQINV,                         & 
#endif
#if KEY_FLUCQ==1
                        QFLUC,FQCFOR,                      & 
#endif
                        INTCACHE,REALCACHE, &
                        LUSED,CGRF)
! Cache storage cleanup
      call chmdealloc('enbfast.src','ENBFS8','realCACHE',realcrecsz,maxpairs,crl=realCACHE)
      call chmdealloc('enbfast.src','ENBFS8','INTCACHE',intcrecsz,maxpairs,intg=INTCACHE)
      If (DONE) then
        call timer_stop(T_ELECVDW)        
        RETURN
      ENDIF
#if KEY_NBIPS==1
  242 CONTINUE                            
#endif
#endif 

      C2OFNB=CTOFNB*CTOFNB
      C2ONNB=CTONNB*CTONNB
      CTROF2=-ONE/C2OFNB
      C4ROF2=FOUR*CTROF2
      IF (CSHIFT) MIN2OF = MINTWO/CTOFNB
!
      IF (SWITCH) THEN
         IF (CTOFNB > CTONNB) THEN
            RUL3=ONE/(C2OFNB-C2ONNB)**3
            RUL12=TWELVE*RUL3
         ENDIF
      ENDIF
      IF (CFSWIT) THEN
!       force-based cdie switching coeffs
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
        END IF
      ENDIF
      IF (LVFSW) THEN
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
        END IF
      END IF
!
!.ab.Coefs, tables... for HYBH.
#if KEY_BLOCK==1
      IF (QHYBH) THEN
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
         IF ( PTABLE < NATOMX ) THEN
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
      ENDIF
#endif 
!.ab.

!
!     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
!
      IF (First > 1) THEN
        ITEMP=INBL(First-1)
      ELSE
        ITEMP=0
      ENDIF
      DO I=First,NATOMX
#if KEY_IMCUBES==1
         IF(LBYCBIM)ITEMP = INBL(I+NATOMX) 
#endif
         NPR=INBL(I)-ITEMP
#if KEY_GCMC==1
!        ARD 01-05-21 Make invisible for grand canonical MC
         if(qgcmc) then
            lgcmcon = gcmcon(i)
         else
            lgcmcon = .true.
         endif

         IF (NPR > 0 .and. lGCMCON) THEN
#else /**/
         IF (NPR > 0) THEN
#endif 

            IACI=IACNB(I)
            CGT=CGF*CGX(I)
            CRXI=X(I)
            CRYI=Y(I)
            CRZI=Z(I)
            DTX=ZERO
            DTY=ZERO
            DTZ=ZERO
            !.ab.
#if KEY_BLOCK==1
            !         IF (QHYBH) IBL=I4VAL(IBLCKP,I)
            IF (QHYBH) IBL=IBLOCK(I)
#endif 
            !.ab.
            !
            DO 30 J=1,NPR
! T.S. initialize TFELEC,TFVDW
               TFELEC=ZERO
               TFVDW=ZERO
               KVECT=JNBL(ITEMP+J)
               JVECT=ABS(KVECT)
!
! namkh 01/20/04
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
                  if(qmused)then
                  IF((ABS(IGMSEL(I)) == 1.OR.ABS(IGMSEL(I)).EQ.2) .AND. &
                       (ABS(IGMSEL(JVECT)) == 1.OR.ABS(IGMSEL(JVECT)).EQ.2) &
                       .AND.QGMREM) GOTO 30
                  endif
#endif 
!
#if KEY_GCMC==1
                  if(qgcmc) then
                     IF (.NOT. GCMCON(JVECT)) GOTO 30
                  endif
#endif 
!
                  TX=CRXI-X(JVECT)
                  TY=CRYI-Y(JVECT)
                  TZ=CRZI-Z(JVECT)
#if KEY_PBOUND==1 /*pbound*/
      If(qBoun) then
         If(qCUBoun.or.qTOBoun) then
            TX      = BOXINV * TX
            TY      = BOYINV * TY
            TZ      = BOZINV * TZ
            tx = tx - nint(tx)
            ty = ty - nint(ty)
            tz = tz - nint(tz)
!!$            IF(TX >  HALF) TX = TX - ONE
!!$            IF(TX <  -HALF) TX = TX + ONE
!!$            IF(TY >  HALF) TY = TY - ONE
!!$            IF(TY <  -HALF) TY = TY + ONE
!!$            IF(TZ >  HALF) TZ = TZ - ONE
!!$            IF(TZ <  -HALF) TZ = TZ + ONE
            If (qTOBoun) Then
               CORR = HALF * AINT ( R75 * ( ABS ( TX ) + &
                                            ABS ( TY ) + &
                                            ABS ( TZ ) ) )
               TX      = TX    - SIGN ( CORR,  TX  )
               TY      = TY    - SIGN ( CORR,  TY  )
               TZ      = TZ    - SIGN ( CORR,  TZ  )
            Endif
            TX      = XSIZE * TX
            TY      = YSIZE * TY
            TZ      = ZSIZE * TZ
         Else
            Call PBMove(TX, TY, TZ)
         Endif
      Endif
#endif /*      (pbound)*/
            S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
            IF (S2 < C2OFNB) THEN
!.ab.HybH stuf. Block and truncation scheme setup
#if KEY_BLOCK==1
            IF (QHYBH) THEN
!               JBL=I4VAL(IBLCKP,JVECT)
               JBL=IBLOCK(JVECT)
               KK=MAX(IBL,JBL)
               KK=KK*(KK-1)/2+MIN(IBL,JBL)
               IF (KK /= 1) THEN
                  IF (KK == 5) THEN
                     GOTO 30
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
                        IVECT=IACNB(JVECT)+IACI
                        A=CCNBA(IVECT)
                        B=CCNBB(IVECT)
                        CALL GETR02(R02L(NDXM),CZZ,A,B, &
                                    IDXM,NDXM,I,JVECT)
                        R02=R02L(NDXM)
!                        WRITE(OUTU,'(i5,a,f12.5,2i7)') NDXM,
!     $                       ' Pair R02: ',R02,I,JVECT
                     ENDIF
                     IF (KK < 4) THEN
                        R02=R02*L6THR
                     ELSE
                        R02=R02*L6THP
                     ENDIF
                     QOUT=.TRUE.
                     IF (S2 < R02) THEN
!                        write(6,'(a,2i,a,2f)') 'Pair:',i,jvect,
!     A                                         ' at D/A',S2,R02
                        S2=R02
                        QOUT=.FALSE.
                     ENDIF
                  ENDIF
               ENDIF
!.KK=1, do nothing.
            ENDIF
#endif 
!.ab.

#if KEY_SOFTVDW==1 /*softvdw*/
!
! Check if the distance is less than softcore switching
!
               qalter=.false.
               qvdw=.false.
               if(qgamin) then
                  qalter=.true.
                  emax=rgamin
                  emin=egamina
               endif
#endif /*  (softvdw)*/
#if KEY_MTS==1 /*mts*/
               IF (SLFG) THEN
                  IF ((SLFG1.AND.(S2 > RSCUT2))) GOTO 999
                  IF ((SLFG2.AND.(S2 < RSHL2)))  GOTO 999
               ENDIF
#endif /* (mts)*/
               LOUTER=(S2 > C2ONNB)
!
!     Electrostatic / van der Waals switch function
               IF (SWITCH) THEN
                  IF (LOUTER) THEN
                     RIJL=C2ONNB-S2
                     RIJU=C2OFNB-S2
                     FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                     DFSW=RIJL*RIJU*RUL12
                  ENDIF
               ENDIF
               IVECT=LOWTP(MAX(IACNB(JVECT),IACI))+IACNB(JVECT)+IACI

#if KEY_CHEQ==1
               IF (QCG) THEN
                  CH=CGF
               ELSE
                  CH=CGT*CGX(JVECT)
               ENDIF
#else /**/
               CH=CGT*CGX(JVECT)
#endif 
!
#if KEY_NBIPS==1
               IF(QIPS)THEN
!WXW Long range potential using isotropic periodic sum (IPS)
                  U2=S2*RIPS2R     
                  DOEIPS=LEIPSX                   
                  DOVIPS=LVIPSX.OR.DBEXPIPS                   
               ENDIF
#endif /* KEY_NBIPS */
!
               IF(KVECT < 0) THEN
                  if (Lfma .and. (.not. lvdwx)) goto 30
                  CH=CH*E14FAC
                  IVECT=IVECT+NITCC2
#if KEY_NBIPS==1
                  DOEIPS=.FALSE.
                  DOVIPS=.FALSE.
#endif /* KEY_NBIPS */
               ENDIF
!
               TR2=ONE/S2
               TR6=TR2*TR2*TR2
!
!------ Electrostatic energies (only if there are charges)
!
#if KEY_SOFTVDW==1
               cho=ch
#endif 
               IF(CH /= ZERO) THEN
                  IF (LCONS) THEN
                     R1 = SQRT(TR2)

                     IF (CGRF) THEN ! Wei Chen 2015
                       ENE = CH*(R1 + (HALF*RFCON*RC3*S2) - ((ONE+HALF*RFCON)*RC))
                       TFELEC = CH*(RFCON*RC3-R1*TR2)
! cdie original shift
                     ELSE IF (CSHFT) THEN
                        FSH=ONE+S2*CTROF2
                        CH=CH*R1*FSH
                        ENE=CH*FSH
                        TFELEC= -ENE*TR2+C4ROF2*CH
! cdie force shift
                     ELSE IF (CSHIFT) THEN
                        CH=CH*R1
                        ENE=CH*(ONE + S2*(MIN2OF*R1-CTROF2))
                        TFELEC= - CH*(CTROF2 + TR2)
! cdie force switch
                     ELSE IF (CFSWIT) THEN
                        IF (LOUTER) THEN
                          ENE = CH*( R1* (ACOEF - S2*(BCOEF + S2*(COVER3 &
                                      + DOVER5*S2))) + CONST)
                          TFELEC = - CH*R1*( ACOEF*TR2 + BCOEF + &
                                        S2*(CCOEF + DCOEF*S2) )
                        ELSE
                          ENE = CH*(R1+EADD)
                          TFELEC = - CH*R1*TR2
                        ENDIF
! cdie switch
                     ELSE IF (CSWIT) THEN
                        IF (LOUTER) THEN
                          CH=CH*R1
                          ENE=CH*FSW
                          TFELEC= -ENE*TR2+CH*DFSW
                        ELSE
                          ENE=CH*R1
                          TFELEC = -ENE*TR2
                        ENDIF
#if KEY_NBIPS==1 /*nbips_elec*/
!WXW Electrostatic potential using isotropic periodic sum (IPS)
                     ELSE IF (DOEIPS) THEN
                        !  Electrostatic IPS
                        !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                        !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                        ! 
                        U1=SQRT(U2)
                        PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                         +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                        DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                         +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                        ENEP=CH*RIPSR
                        ENE=ENEP*(PE-PIPSEC)
                        TFELEC=ENEP*DPE*TR2
#endif /*  (nbips_elec)*/
! no elec
                     ELSE  
                          ENE=ZERO
                          TFELEC = ZERO
                     ENDIF
                  ELSE
! rdie original shift
                     IF (RSHFT) THEN
                        FSH=ONE+S2*CTROF2
                        CH=CH*TR2*FSH
                        ENE=CH*FSH
                        TFELEC= -TWO*ENE*TR2+C4ROF2*CH
! rdie switch
                     ELSE IF (RSWIT) THEN
                        IF (LOUTER) THEN
                          CH=CH*TR2
                          ENE=CH*FSW
                          TFELEC= -TWO*ENE*TR2+CH*DFSW
                        ELSE
                          ENE=CH*TR2
                          TFELEC= -TWO*ENE*TR2
                        ENDIF
#if KEY_NBIPS==1 /*nbips_eler*/
!WXW Electrostatic potential using isotropic periodic sum (IPS)
                     ELSE IF (DOEIPS) THEN
                        !  Electrostatic IPS
                        !   etr1=1/r2+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
                        !   detr1/dr*r1=-1/r2+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
                        ! 
                        PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                         +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))-PIPSEC
                        DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                         +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                                ENEP=CH*RIPS2R
                                ENE=ENEP*PE
                                TFELEC  =ENEP*DPE*TR2
#endif /*   (nbips_eler)*/
! no elec
                     ELSE
                        ENE=ZERO
                        TFELEC = ZERO
                     ENDIF
                  ENDIF
               ELSE
                  ENE=ZERO
                  TFELEC = ZERO
               ENDIF
#if KEY_CHEQ==1
           IF (QCG) THEN
               CH=CGX(I)*CGX(JVECT)
               TFELEC=TFELEC*CH
  101          CONTINUE

               IF (QCG.AND.(CGMODEL == 1)) THEN
                 IF ( (CGX(I) /= 0.0).AND.(CGX(JVECT).NE.0.0)) THEN
                  DCH(I)=DCH(I)+ENE*CGX(JVECT)
                  DCH(JVECT)=DCH(JVECT)+ENE*CGX(I)
                  IF(QREF) THEN
                     DCH(I)=DCH(I)+ENE*CGREF(JVECT)
                     DCH(JVECT)=DCH(JVECT)+ENE*CGREF(I)
                  ENDIF
                 ENDIF
               ENDIF

               IF (FQINV) THEN
                  ETA(I,JVECT)=ENE/CGF
                  ETA(JVECT,I)=ETA(I,JVECT)
               ENDIF
               ENE=ENE*CH
            ELSE
            ENDIF
#endif 
!
!------ End of Electrostatic energies
!
!------ VDW energies
!
            IF (.NOT.QETEN .AND. .NOT.QETSR) THEN  
!
! vdw shift
               IF (LVSH) THEN
                  CA=CCNBA(IVECT)*TR6*TR6
                  CC=S2*S2*S2*CCNBC(IVECT)
                  ENEVDW = CA-CCNBB(IVECT)*TR6
                  ENN=ENEVDW-CC+CCNBD(IVECT)
                  TFVDW=MINSIX*(ENEVDW+CA+CC)*TR2
                  
! vdw force switch
               ELSE IF(LVFSW) THEN
                  IF (LOUTER) THEN
                     IF(.NOT.LCONS .OR. CH == ZERO) R1 = SQRT(TR2)
                     R3 = R1*TR2
                     RJUNK6 = TR6-RECOF6
                     RJUNK3 = R3-RECOF3
                     CR12 = CCNBA(IVECT)*OFDIF6*RJUNK6
                     CR6  = CCNBB(IVECT)*OFDIF3*RJUNK3
                     ENN = CR12*RJUNK6 - CR6*RJUNK3
                     TFVDW = TR2*(SIX*CR6*R3 - TWELVE*CR12*TR6)
                  ELSE
                     CA=CCNBA(IVECT)*TR6*TR6
                     ENEVDW = CA-CCNBB(IVECT)*TR6
                     ENN = ENEVDW+CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
                     TFVDW = MINSIX*TR2*(ENEVDW+CA)
                  END IF
                  ! vdw switch
               ELSE IF(LVSW) THEN
                  CA=CCNBA(IVECT)*TR6*TR6
                  IF (LOUTER) THEN
                     ENEVDW = CA-CCNBB(IVECT)*TR6
#if KEY_WCA==1
                     ! MH12 (Mostly annoying for developers):
                     ! we need to protect this so results from LITE compile
                     ! match the results of a default compile
                     ! 
                     do_softcore0: IF(LSOFTCORE0.or.LSOFTCORE1) THEN
                        ENN=WCA(I)*WCA(JVECT)*ENEVDW*FSW
                        TFVDW= &
                             WCA(I)*WCA(JVECT)*(ENEVDW*DFSW-SIX*TR2*(ENEVDW+CA)*FSW)
                     ELSE do_softcore0
#endif 
                        ENN=ENEVDW*FSW
                        TFVDW = ENEVDW*DFSW-SIX*TR2*(ENEVDW+CA)*FSW
#if KEY_WCA==1
                     ENDIF do_softcore0
#endif
                  ELSE
#if KEY_WCA==1
                     do_softcore1: IF(LSOFTCORE0.or.LSOFTCORE1) THEN
                        IF (CCNBA(IVECT)  ==  0.0) THEN
                           VDWWL = 0.0
                           RMIN6 = 0.0
                        ELSE
                           VDWWL=PT25*CCNBB(IVECT)*CCNBB(IVECT)/CCNBA(IVECT)
                           RMIN6=HALF*CCNBB(IVECT)/VDWWL
                        ENDIF
                        IF (S2*S2*S2  >  RMIN6) THEN
                           ENN=CA-CCNBB(IVECT)*TR6
                           EGLJTMP = MINSIX*(ENN+CA)*TR2
                           EPRPL = 0.0D0
                           EGLJRPL = 0.0D0
                        ELSE
                           ENN=-VDWWL
                           EGLJTMP = 0.0D0
                           IF (LLSOFT) THEN
                              RMIN2=RMIN6**(1.0D0/3.0D0)
                              TMP = RMIN2*(1.0-SCVDWCUTR)*(1.0-SCVDWCUTR)
                              IF (S2  <=  (RMIN2 - TMP)) THEN
                                 TR2 = 1.0D0/(S2 + TMP)
                                 TR6 = TR2 * TR2 * TR2
                                 CA = CCNBA(IVECT)*TR6*TR6
                                 EPRPL=CA-CCNBB(IVECT)*TR6
                                 EGLJRPL=MINSIX*TR2*(EPRPL+CA)
                                 EPRPL=EPRPL+VDWWL
                              ELSE
                                 EPRPL=0.0D0
                                 EGLJRPL=0.0D0
                              ENDIF
                           ELSE
                              EPRPL=CA-CCNBB(IVECT)*TR6
                              EGLJRPL=MINSIX*TR2*(EPRPL+CA)
                              EPRPL=EPRPL+VDWWL
                           ENDIF
                        ENDIF
                        TFVDW = EGLJTMP*WCA(I)*WCA(JVECT)+EGLJRPL
                        ENN = (ENN*WCA(I)*WCA(JVECT)+EPRPL)
                     ELSE do_softcore1
#endif 
                        ENN = CA-CCNBB(IVECT)*TR6
                        TFVDW =  MINSIX*TR2*(ENN+CA)
#if KEY_WCA==1
                     ENDIF  do_softcore1
#endif
                  ENDIF
               ELSE IF (DBEXP) THEN
                 SIG=CCNBB(IVECT)
                 IF(SIG>RSMALL.AND.S2<C2OFNB)THEN
                   EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                   SIG=(HALF*SIG/EMIN)**SIXTH
                   R1 = SQRT(TR2)
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
#endif /* key_nbips */
                 ELSE 
                   ENN=ZERO 
                   TFVDW=ZERO 
                 ENDIF
#if KEY_NBIPS==1 /*nbips_vdw*/                  
                 !WXW VDW potential using isotropic periodic sum (IPS)
               ELSE IF (LVIPSX.AND.DOVIPS) THEN
                  U4=U2*U2
                  U6R=ONE/U4/U2
                  U12R=U6R*U6R
                  !  L-J r6 term
                  !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                  !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                  !
                  PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                       +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC
                  DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                       +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                  !  L-J r12 term 
                  !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                  !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                  !
                  PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                       +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC
                  DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                       +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                  ENEVA =CCNBA(IVECT)*RIPS12R
                  ENEVC =-CCNBB(IVECT)*RIPS6R
                  ENN=ENEVA*PVA+ENEVC*PVC
                  TFVDW=(ENEVA*DPVA+ENEVC*DPVC)*TR2
               ELSE IF (DBEXPIPS) THEN
                 SIG=CCNBB(IVECT)
                 EMIN=SIG*SIG/CCNBA(IVECT)/FOUR
                 SIG=(HALF*SIG/EMIN)**SIXTH
                 R1 = SQRT(TR2)
                 IF(DOVIPS.AND.SIG>RSMALL.AND.SRIPS*R1*SIG>ONE)THEN
                   AKR=AIPSVA(0)/R1/SIG
                   EXPAR=EXP(-AKR)
                   !AKRC=AIPSVA(0)*RIPS/SIG
                   !EXPARC=EXP(-AKRC)
                   ! energy
                   !ENEVA=AIPSVA(1)*(EXPAR+EXPARC*(EXPARC/EXPAR+(TWELVE+SIX*EXPARC/AKRC)/AKRC/AKRC))
                   ENEVA=AIPSVA(1)*(EXPAR+AIPSVA(4)/EXPAR+AIPSVA(5))
                   ! force 
                   !DPVA=BIPSVA(1)*(EXPAR-EXPARC*EXPARC/EXPAR)/SIG
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
                   !DPVC=BIPSVC(1)*(EXPCR-EXPCRC*EXPCRC/EXPCR)/SIG
                   DPVC=BIPSVC(1)*(EXPCR-AIPSVC(4)/EXPCR)/SIG
                   ENN=EMIN*(ENEVA-ENEVC)
                   TFVDW=EMIN*(DPVA-DPVC)*R1
                 ELSE 
                   ENN=ZERO 
                   TFVDW=ZERO 
                 ENDIF
#endif /*  (nbips_vdw)*/
                  ! no vdw
               ELSE
                  ENN = ZERO
               ENDIF

               !
           ELSE IF (QETSR) THEN  ! QETSR
               ! V_ETSR = V_ETEN / (1 + (2*r/(3*sigma))^12)
               ! Note:   CCNBA = eps * sigma^12
               !         CCNBB = 2 * eps * sigma^6
               ! SWTMP = (2*r/(3*sigma))^12
               ! TTP12 = (2/3)^12 /4
               TTPW1 = EIGHT*FOUR
               TTPW2 = NINE*NINE*NINE
               TTP12 = TTPW1*TTPW1/(TTPW2*TTPW2)
               SWTMP = TTP12*CCNBB(IVECT)*CCNBB(IVECT) &
                    /(TR6*TR6*CCNBA(IVECT)*CCNBA(IVECT))

               CA=THIRTN*CCNBA(IVECT)*TR6*TR6
               ENEVDW = (CA-NINE*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2+TWO*CCNBB(IVECT)*TR6 ) &
                    / ( ONE + SWTMP) 
               ENN=ENEVDW
               TFVDW=TR2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                    (45*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2)+(SIX*CCNBB(IVECT)*TR6)) &
                    / ( ONE + SWTMP) &
                    - TR2*ENEVDW*TWELVE*SWTMP/(ONE+SWTMP) 
           ELSE  ! QETEN
               CA=THIRTN*CCNBA(IVECT)*TR6*TR6
               ENEVDW = CA-NINE*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2+TWO*CCNBB(IVECT)*TR6
               ENN=ENEVDW
               TFVDW=TR2*MINTWO*((78*CCNBA(IVECT)*TR6*TR6)- &
                    (45*CCNBB(IVECT)* &
                    ((TWO*CCNBA(IVECT)/CCNBB(IVECT))**(TWO/THREE))* &
                    TR6*TR2*TR2)+(SIX*CCNBB(IVECT)*TR6))
             ENDIF
!
!------ End of VDW energies
!
#if KEY_MTS==1
!------ LONG-SHORT RANGE MTS METHOD
               IF(SLFG) THEN
                  SWFE = ONE
                  SWF  = ONE
                  IF((S2 >= RSHL2).AND.(S2 <= RSCUT2)) THEN
                     RR1 = SQRT(S2)
                     RR2 = (RR1 - RSHL)/RHEAL
                     RR3 = ONE-RR2*RR2*RR2*(6.0*RR2*RR2-15.0*RR2+10.0)
                     IF(SLFG1) SWFE=ZERO
                  ELSE IF(S2 < RSHL2) THEN
                     RR3 = ONE
                     IF(SLFG2) SWFE=ZERO
                  ELSE IF(S2 > RSCUT2) THEN
                     RR3 = ZERO
                     IF(SLFG1) SWFE=ZERO
                  ENDIF
                  IF(SLFG1) SWF=RR3
                  IF(SLFG2) SWF=ONE-RR3
                  ENE=ENE*SWFE
                  ENN=ENN*SWFE
                  TFELEC=TFELEC*SWF
                  TFVDW=TFVDW*SWF
               ENDIF
#endif /* mts*/
!
#if KEY_SOFTVDW==1
!
! IF soft core is used correcting the energies
!
        if(qalter) then
        s1=sqrt(s2)
        if(enn > emax/2.) then
!
! The correction of VDW is necessary - adjustment of rmin to
! overcome electrostatic catastrophy
! setting maximum hard core VDW to be Emax
!
! figuring Rmin from the function
!
! x1=1/rc**6
!
! Functional form is Esoft=emax+a*rij**b -  based on BB suggestion
!
! Subject to the following conditions
!
! Esof(rc)=Emax/2
! Force=Fh(rcut)
! Esof(rc)=Eh(rc)
!
! Emax has to be ge 0
!
! VDW soft core
!
! vdw shift not supported
!
! vdw force switch
               IF(LVFSW) THEN
                  IF (.NOT.LOUTER) THEN
!                    CA=CCNBA(IVECT)*TR6*TR6
!                    ENEVDW = CA-CCNBB(IVECT)*TR6
!                    ENN = ENEVDW+CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
!                    TFVDW = MINSIX*TR2*(ENEVDW+CA)
!
! soft core
!
        enevdw=CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
        x1=sqrt(CCNBB(IVECT)**2.0-4.*CCNBA(IVECT)*(enevdw-emax/2.))
        x1=(CCNBB(IVECT)+x1)/(2.0*CCNBA(IVECT))
        rc2=x1**(-1.0D0/6.0D0)
        rc2o=rc2*rc2
        ca=CCNBA(IVECT)*(x1**2)
        alfa=-6.0*(2.0*ca-CCNBB(IVECT)*x1)
        alfa=alfa/rc2
!
! alfa is the force computed at Rcut
!
        if(qvdwexp) then
        beta=-2.0*alfa*rc2/emax
        alfa=-0.5*emax/(rc2**beta)
        enn=emax+alfa*s1**beta
        tfvdw=(enn-emax)*beta/s2
        qvdw=.true.
        else
!
! linear soft core VDW
!
        enn=0.5*emax+alfa*(s1-rc2)
        tfvdw=alfa/s1
        endif
                  END IF
! vdw switch
               ELSE IF(LVSW) THEN
                  IF (.not.LOUTER) THEN
        x1=CCNBB(IVECT)+sqrt(CCNBB(IVECT)**2.0+2.*CCNBA(IVECT)*emax)
        x1=x1/(2.0*CCNBA(IVECT))
        rc2=x1**(-1.0D0/6.0D0)
        rc2o=rc2*rc2
        ca=CCNBA(IVECT)*(x1**2)
        alfa=-6.0*(2.0*ca-CCNBB(IVECT)*x1)
        alfa=alfa/rc2
        if(qvdwexp) then
        beta=-2.0*alfa*rc2/emax
        alfa=-0.5*emax/(rc2**beta)
        enn=emax+alfa*s1**beta
        tfvdw=(enn-emax)*beta/s2
        else
!
! Linear soft core VDW energy and force
!
        enn=0.5*emax+alfa*(s1-rc2)
        tfvdw=alfa/s1
        endif
                  ENDIF
              ENDIF
            endif

!        if(ene < emin.or.ene > egaminr) then
         if(ene < (0.5*egamina).or.ene > (0.5*egaminr)) then
!
!  Now the modification of electrostatic interactions
!
!  Electrostatics soft core
        ch=cho
        if(ch > 0) then
        emin=egaminr
        else
        emin=egamina
        endif
                  IF (LCONS) THEN
                     R1 = SQRT(TR2)
! cdie original shift
                     IF (CSHIFT) THEN
! not supported
! cdie force switch
!
! the soft core acts only for inner switching distance
!
                     ELSE IF (CFSWIT) THEN
                        IF (.not.LOUTER) THEN
!c                          alfa=-ch*(emin/ch-eadd)**2.0
!c                          ENE = alfa*s1+2.0*emin-EADD*ch
!c                          TFELEC = alfa*r1
                        ecut=0.5*emin
                        rc2=ch/(ecut-ch*eadd)
                        rc2o=rc2*rc2
                    if(qelexp) then
                        alfa=-ch/rc2o
                        beta=alfa*rc2/(ecut-emin)
                        alfa=(ecut-emin)/rc2**beta
                        ene=emin+alfa*s1**beta
                        tfelec=(enn-emin)*beta/s2
                    else
                       alfa=-ch/rc2o
                       ENE = alfa*s1+2.0*ecut-EADD*ch
                       TFELEC = alfa*r1
                    endif
                        ENDIF
! cdie switch
                     ELSE IF (CSWIT) THEN
                        IF (.NOT.LOUTER) THEN
!                          ENE=CH*R1
!                          TFELEC = -ENE*TR2
                 ECUT=0.5*emin
                 rc2=ch/ecut
                 rc2o=rc2*rc2
!              if(qelexp) then
!                        alfa=-ch/rc2o
!                        beta=alfa*rc2/(ecut-emin)
!                        alfa=(ecut-emin)/rc2**beta
!                        ene=emin+alfa*s1**beta
!                        tfelec=(ene-emin)*beta/s2
!              else
!
! linear soft core
!
                        alfa=-ecut*ecut/ch
                        ene=2.0*ecut+alfa*s1
                        tfelec=alfa*R1
!              endif
                        ENDIF
                     ENDIF
                  ELSE
                     R1 = SQRT(TR2)

! rdie original shift
                     IF (RSHFT) THEN
!                        CH=CH*TR2*FSH
!                        ENE=CH*FSH
!                        TFELEC= -TWO*ENE*TR2+C4ROF2*CH
                     Ecut=emin*0.5
                     beta=2.0*CTROF2-ecut/ch
                     ct2=CTROF2**2.0
                     x1=sqrt((beta)**2.0-4.0*CT2)
                     x1=0.5*(-beta-x1)/CT2
                     rc2=sqrt(x1)
                     rc2o=x1
                     FSH=ONE+x1*CTROF2
              if(qelexp) then
                        alfa=-2.0*ecut/rc2
                        alfa=alfa+4.0*FSH*ch*CTROF2/rc2
                        beta=alfa*rc2/(ecut-emin)
                        alfa=(ecut-emin)/rc2**beta
                        ene=emin+alfa*s1**beta
                        tfelec=(ene-emin)*beta/s2
              else
!
! linear soft core
!
                     alfa=-2.0*ch*FSH*FSH/rc2/x1
                     alfa=alfa+4.0*FSH*ch*CTROF2/rc2
                     ene=ecut+alfa*(s1-rc2)
                     tfelec=alfa*r1
              endif
! rdie switch
! rdie switch
                     ELSE IF (RSWIT) THEN
                        IF (.NOT.LOUTER) THEN
                        Ecut=0.5*emin
                        rc2o=ch/ecut
                        rc2=sqrt(rc2o)
            if(qelexp) then
                        alfa=-2.0*ecut/rc2
                        beta=alfa*rc2/(ecut-emin)
                        alfa=(ecut-emin)/rc2**beta
                        ene=emin+alfa*s1**beta
                        tfelec=(ene-emin)*beta/s2
            else
!
! linear soft core
!
                        alfa=-2.0*ecut/rc2
                        ene=3.0*ecut+alfa*s1
                        tfelec=alfa*r1
            endif
                        ENDIF
! no elec
                     ENDIF
                  ENDIF
        endif
!
!------ End of Electrostatic soft core energies
!
        endif
#endif 
#if KEY_BLOCK==1 /*block_1*/
               IF (QBLOCK) THEN
!ldm
                if (qmld) then
                   call msld_nb_scale_enerforce(iblckp(i),iblckp(jvect),tfelec,ene,tfvdw,enn)
                   tf = tfelec + tfvdw
                else
!ldm
                  IBL=IBLOCK(I)
                  JBL=IBLOCK(JVECT)
#if KEY_DOCK==1
!                 get asymmetric matrix coefficient
                  DOCFI = 1.0
                  DOCFJ = 1.0
                  IF(QDOCK) THEN
                    KDOC  = (IBL - 1)*NBLOCK + JBL
                    DOCFI = BLDOCP(KDOC)
                    KDOC  = (JBL - 1)*NBLOCK + IBL
                    DOCFJ = BLDOCP(KDOC)
                  ENDIF
#endif /*  DOCK*/
                  KK=MAX(IBL,JBL)
                  KK=KK*(KK-1)/2+MIN(IBL,JBL)
!
                  IF (QPRNTV) THEN
                    IF (IBL == 1 .OR. JBL == 1 .OR. IBL == JBL) THEN
                      VBELEC(MAX(IBL,JBL)) = VBELEC(MAX(IBL,JBL)) + ENE
                      VBVDW(MAX(IBL,JBL)) = VBVDW(MAX(IBL,JBL)) + ENN
                    ENDIF
                  ENDIF
!ldm
                  IF(QLDM .or. QLMC) THEN 
                    JLDM = MAX(IBL,JBL)
                    ILDM = MIN(IBL,JBL)
                    IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                       (ILDM == 1.AND.JLDM >= LSTRT)) THEN
                      FALPHA = (ENE + ENN)
                      LAGMUL = LAGMUL + FALPHA
                      BIFLAM(JLDM) = BIFLAM(JLDM) + FALPHA
                    ENDIF
                    IF(NRST == 2)THEN
                      IF((ILDM >= LSTRT.AND.ILDM == JLDM).OR. &
                         (ILDM == 2.AND.JLDM >= LSTRT)) THEN
                        FALPHA=ENE+ENN
                        BFRST(JLDM) = BFRST(JLDM) + FALPHA
                      ENDIF
                    ENDIF
                  ENDIF
                 TF=TFELEC+TFVDW
                 IF(NRST == 1)THEN
!  BETWEEN PROTEIN(I) AND LIGAND(JVECT)
                  IF(IBL == 1.AND.JBL >= LSTRT) THEN
! ADD FORCE TO I(PROTEIN)
                     ENVDX((JBL-2)*NATOMX+I) = ENVDX((JBL-2)*NATOMX+I) + TX * TF
                     ENVDY((JBL-2)*NATOMX+I) = ENVDY((JBL-2)*NATOMX+I) + TY * TF
                     ENVDZ((JBL-2)*NATOMX+I) = ENVDZ((JBL-2)*NATOMX+I) + TZ * TF
! ADD FORCE TO JVECT(LIGAND)
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
!  BETWEEN LIGAND(I) AND PROTEIN(JVECT)
                  ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
! ADD FORCE TO JVECT(PROTEIN)
                     ENVDX((IBL-2)*NATOMX+JVECT) = ENVDX((IBL-2)*NATOMX+JVECT) - TX * TF
                     ENVDY((IBL-2)*NATOMX+JVECT) = ENVDY((IBL-2)*NATOMX+JVECT) - TY * TF
                     ENVDZ((IBL-2)*NATOMX+JVECT) = ENVDZ((IBL-2)*NATOMX+JVECT) - TZ * TF
! ADD FORCE TO I(LIGAND)
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
!  WITHIN LIGAND
                  ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
                  ENDIF
                 ELSE IF(NRST == 2)THEN
                   IF(IBL >= LSTRT.AND.JBL == 2)THEN
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
                   ELSE IF(JBL >= LSTRT.AND.IBL == 2)THEN
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
                   ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
                   ENDIF
                 ELSE IF(NRST == 3)THEN
! Add force only to ligand
! Protein(I) & Ligand(JVECT)
                   IF(IBL == 1.AND.JBL >= LSTRT) THEN
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
! Protein(JVECT) & Ligand(I)
                   ELSE IF(JBL == 1.AND.IBL >= LSTRT) THEN
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
! Within ligand(I)
                   ELSE IF(IBL >= LSTRT.AND.IBL == JBL) THEN
                     ENVDX(I) = ENVDX(I) + TX * TF
                     ENVDY(I) = ENVDY(I) + TY * TF
                     ENVDZ(I) = ENVDZ(I) + TZ * TF
                     ENVDX(JVECT) = ENVDX(JVECT) - TX * TF
                     ENVDY(JVECT) = ENVDY(JVECT) - TY * TF
                     ENVDZ(JVECT) = ENVDZ(JVECT) - TZ * TF
                   ENDIF
                    TF=TFELEC*BLCOE(KK)+TFVDW*BLCOV(KK)
                 ENDIF
!ldm
!
#if KEY_DOCK==1
                  IF(QDOCK) THEN
                    ENE=ENE*BLCOE(KK)*0.5*(DOCFI + DOCFJ)
                    ENN=ENN*BLCOV(KK)*0.5*(DOCFI + DOCFJ)
                    TF=TFELEC*BLCOE(KK)+TFVDW*BLCOV(KK)
                  ELSE
#endif 
                    IF(QBVSPLT) THEN
                      CA=CCNBA(IVECT)*TR6*TR6
! Scaled-VDW = (VDW - repulsive)*Scale-attractive + Repulsive*Scale-repulsive
                      ENN = (ENN - CA)*BLCOVA(KK) + CA*BLCOVR(KK)
!%%AvdV-070721 bugfix
! fixed issue that Eelec was not scaled properly when using vdwa or vdwr
! e.g.: coef 1 2 1.0 vdwa 0.0 elec 0.0  !# old code: results in elec value for lambda=1
                                        !# new code: results in properly scaled elec
                      ENE=ENE*BLCOE(KK)
!%%AvdV-070721--end
                      TF = MINSIX*TR2*(CA+CA)
                      TFVDW = TF*BLCOVR(KK) + (TFVDW - TF)*BLCOVA(KK)
                      TF = TFELEC*BLCOE(KK)+TFVDW
                    ELSE
                      ENE=ENE*BLCOE(KK)
                      ENN=ENN*BLCOV(KK)
                      TF=TFELEC*BLCOE(KK)+TFVDW*BLCOV(KK)
                    ENDIF
#if KEY_DOCK==1
                  ENDIF                                      
#endif
                endif                                        !ldm
               ELSE
                  TF=TFELEC+TFVDW
               ENDIF
!.ab.QHYBH code.
               IF (QHYBH) THEN
                  IF (KK /= 1) THEN
                     IF (KK < 4) THEN
                        DNER=DNER+ENE*DFR
                        DNNR=DNNR+ENN*DFR
                        IF (QOUT) THEN
                           TF=TF*FR
                        ELSE
                           DNER=DNER+FR*TFELEC*R02*DL112R
                           DNNR=DNNR+FR*TFVDW *R02*DL112R
                           TF=ZERO
                        ENDIF
                        ENE=ENE*FR
                        ENN=ENN*FR
                     ELSE
                        DNEP=DNEP+ENE*DFP
                        DNNP=DNNP+ENN*DFP
                        IF (QOUT) THEN
                           TF=TF*FP
                        ELSE
                           DNEP=DNEP+FP*TFELEC*R02*DL112P
                           DNNP=DNNP+FP*TFVDW *R02*DL112P
                           TF=ZERO
                        ENDIF
                        ENE=ENE*FP
                        ENN=ENN*FP
                     ENDIF
                  ENDIF
!.ab. KK=1, do nothing.
               ENDIF
!.ab.
!
               IF (.NOT. NOFORC) THEN
#else /* (block_1)*/
               TF=TFELEC+TFVDW
#endif /* (block_1)*/
!
               TX=TX*TF
               TY=TY*TF
               TZ=TZ*TF
#if KEY_BLOCK==1 /*block_2*/
#if KEY_DOCK==1
               IF(QDOCK) THEN
                 DTX=DTX+TX*DOCFI
                 DTY=DTY+TY*DOCFI
                 DTZ=DTZ+TZ*DOCFI
                 if (Lfma .and. (.not. lvdwx)) then
                   DX(JVECT)=DX(JVECT)+TX*DOCFJ
                   DY(JVECT)=DY(JVECT)+TY*DOCFJ
                   DZ(JVECT)=DZ(JVECT)+TZ*DOCFJ
                 else
                   DX(JVECT)=DX(JVECT)-TX*DOCFJ
                   DY(JVECT)=DY(JVECT)-TY*DOCFJ
                   DZ(JVECT)=DZ(JVECT)-TZ*DOCFJ
                 endif
               ELSE
#endif /* dock*/
#endif /* (block_2)*/
#if KEY_CHEQ==1
            IF (.NOT.FQINV) THEN              
#endif
               DTX=DTX+TX
               DTY=DTY+TY
               DTZ=DTZ+TZ
               if (Lfma .and. (.not. lvdwx)) then
                  DX(JVECT)=DX(JVECT)+TX
                  DY(JVECT)=DY(JVECT)+TY
                  DZ(JVECT)=DZ(JVECT)+TZ
               else
                  DX(JVECT)=DX(JVECT)-TX
                  DY(JVECT)=DY(JVECT)-TY
                  DZ(JVECT)=DZ(JVECT)-TZ
               endif
#if KEY_CHEQ==1
            ENDIF                             
#endif
#if KEY_DOCK==1
           ENDIF                              
#endif
#if KEY_BLOCK==1
         ENDIF                                
#endif
!
#if KEY_FLUCQ==1
! If FlucQ is active, add the electrostatic energy to the relevant FlucQ
! charge arrays
               IF (QFLUC) THEN
                  FQCFOR(I)=FQCFOR(I)+ENE
                  FQCFOR(JVECT)=FQCFOR(JVECT)+ENE
               ENDIF
#endif 
               ENB=ENB+ENN
               EEL=EEL+ENE
#if KEY_MTS==1
999      CONTINUE
#endif 
            ENDIF  ! IF (S2 < C2OFNB)
 30      CONTINUE
!
!     RESTORE i-TH COMPONENT OF FORCE IN THE ARRAY
#if KEY_BLOCK==1
         IF (.NOT. NOFORC) THEN                    
#endif
#if KEY_CHEQ==1
            IF (.NOT.FQINV) THEN                   
#endif
               if (Lfma .and. (.not. lvdwx)) then
                  DX(I)=DX(I)-DTX
                  DY(I)=DY(I)-DTY
                  DZ(I)=DZ(I)-DTZ
               else
                  DX(I)=DX(I)+DTX
                  DY(I)=DY(I)+DTY
                  DZ(I)=DZ(I)+DTZ
               endif
#if KEY_CHEQ==1
            ENDIF                                   
#endif
#if KEY_BLOCK==1
         ENDIF                                      
#endif
         ENDIF  ! IF (NPR > 0)
         ITEMP=INBL(I)
      ENDDO
!.ab.Save and put back qblock.
#if KEY_BLOCK==1
      IF (QHYBH) THEN
         CALL SUMHYB(IHYBH,DNNR,DNNP)
         CALL SUMHYB(IHYBH+1,DNER,DNEP)
         QBLOCK=.TRUE.
      ENDIF
#endif 
!.ab.
!
      call timer_stop(T_ELECVDW)        

      RETURN
   END SUBROUTINE ENBFS8
