module ace_module
  !-----------------------------------------------------------------------
  !     The ACE water model
  !
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_ACE==1 /*ace_fcm*/
  !
  !  ESELF     INT   (NATOM)  - electrostatic self energies
  !  BSOLV     INT   (NATOM)  - solvation radii
  !  DBDE      INT   (NATOM)  - derivative d(BSOLV)/d(ESELF)
  !  DISUM     INT   (NATOM)  - D_i = sum_j D_ij (eq 47 in S&K, JPC 100 (1996))
  !  DISTM     INT   (NPAIR)  - distance matrix ( i < j )
  !  SA14P     INT   (NNB14)  - distance matrix for nonbond exclusion pairs
  !  XFREA     INT   (NATOM)  - reaction force (eq 42) on charge i
  !  YFREA     INT   (NATOM)
  !  ZFREA     INT   (NATOM)
  !  CG2       INT   (NATOM)  - square of charges
  !  XFSDI     INT   (2*NPAIR)- self-dielectric force (eq 42) on volume k due
  !  YFSDI     INT   (2*NPAIR)      to charge i ( index ik, i< k );
  !  ZFSDI     INT   (2*NPAIR)      NPAIR=INBL(NATOM)
  !  SWIT      INT   (NPAIR)  - switch function
  !  SWA14P    INT   (NNB14)  - switch function nonbond exclusion pair
  !  DSWIT     INT   (NPAIR)  - derivative of switching function
  !  NATACE    INT   1        - size of allocated atom arrays
  !  NPAIR     INT   1        - size of allocated nonbonded pair arrays
  !  NPA14     INT   1        - size of allocated bonded pair arrays
  !  CES1H     INT   (MAXATC**2) - parameters used to evaluate Eq. (22) in
  !  CES2H     INT   (MAXATC)        Schaefer & Karplus (1996) J. Phys. Chem.
  !  SIG2IH    INT   (MAXATC**2)     vol.100, p. 1578-1599
  !  MUE4H     INT   (MAXATC**2)
  !  KHYDH     INT   (MAXATC) - factor going from self to hydrophobic energy
  !  ESIIH     INT   (MAXATC) - contribution of volume i to Eself of charge i
  !  ATCACE    INT   1        - size of allocated atom type arrays/matrices
  !  ESFIX     INT   (NATOM)  - self energy contributions of fixed atoms
  !  QFIACE    INT   1        - whether ESFIX array was calculated
  !  CGIACP    INT   (NATOM)  - Charge used for interaction energy
  !  CGSACP    INT   (NATOM)  - Charge used for self energy
  !  DESDBP    INT   (NATOM)  - derivative Eself wrt rescaled Born radius
  !
  !           LACE     use analytic continuum electrostatics (ACE)
  !           LIDEAL   use ideal geometry for bonded self energy terms
  !           LACE2    use ACE with cooperative self energy (ACE2)
  !           LACE3    use ACE with self energy correction (ACE3, experimental)
  !           LQGAUS   use Coulomb's law for Gaussian charges
  !
  !         = EPSI     dielectric constant of interior (solute)
  !         = EPSS     dielectric constant of solvent
  !         = ACEAL    smoothing parameter for effective volumes
  !         = SIGHYD   parameter (kcal/mol/A/A) relating hydrophobic energy
  !                       to solvent-accessible surface area; this global
  !                       parameter (same for all atom types) is kept
  !                       for backward compatibility with ACE1; it is
  !                       not employed when set to a negative value
  !                       (SIGHYD=-1 is default set in iniall.src)
  !         = RSOLV    radius of solvent probe used to calculate surface
  !         = MXBSOL   maximum Born radius (def 14.0; negative == inactive)
  !         = TBSOLV   turning point of the formula for the Born
  !                    solvation radius (def 8.4)
  !         = TBSOLH   same as TBSOLV, for hydrogens (def 3.85)
  !         = SCTON    ESelf switching function turn on dist. (def 10.0)
  !         = SCTOF    ESelf switching function turn off dist. (def 14.0)
  !         = FISCAL   scaling of ionic-ionic charge interaction
  !         = FSSCAL   scaling of ionic charge self energy
  !         = FVSCAL   scaling of atomic volumes
  !
  real(chm_real),allocatable,dimension(:,:) :: CES1,sig2i,mue4
  real(chm_real),allocatable,dimension(:) :: CES2,khyd,esii,bsolv,cgiacp,cgsacp, &
       sa14p,swa14p,eself,esfix,swit,dswit,distm,xfsdi,yfsdi,zfsdi, &
       disum,xfrea,yfrea,zfrea,cg2,dbde,desdbp
  INTEGER       ATCACE, natace, NPAIR,NPA14,QFIACE
  !!
  real(chm_real)        EPSI,EPSS,ACEAL,SIGHYD,RSOLV, &
       MXBSOL,TBSOLV,TBSOLH,SCTON,SCTOF, &
       FISCAL,FSSCAL,FVSCAL
  !!
  LOGICAL       LACE,LIDEAL,LACE2,LACE3,LQGAUS
#if KEY_ENDIF==1
  
#endif
#else /* (ace_fcm)*/
  logical,parameter :: LACE = .false.
#endif /* (ace_fcm)*/

#if KEY_ACE==1 /*ace*/
contains
  subroutine ace_init()
    use number,only:one
    lace   = .false.
    lace2  = .false.
    lideal = .true.
    epsi = 1.0_chm_real
    epss = 80.0_chm_real
    aceal= 1.3_chm_real
    natace = 0
    npair  = 0
    npa14  = 0
    atcace = 0
    !     negative hydrophobic sigma indicates it's not assigned a value
    !     via the nonbonded specifications (use atom type sigma from
    !     ACE2 parameter file):
    sighyd = -0.001_chm_real
    rsolv  = 1.4_chm_real
    qfiace = 0
    mxbsol = 14.0_chm_real
    tbsolv = 8.4_chm_real
    tbsolh = 3.85_chm_real
    !     negative SCTON, SCTOF indicates self energy cutoff is not assigned
    !     values via the nonbonded specifications (use the "normal" cutoff
    !     values CTONNB, CTOFNB):
    scton  = -one
    sctof  = -one
    lqgaus = .false.
    fiscal = one
    fsscal = one
    fvscal = one
    return
  end subroutine ace_init

  !======================================================================
  !
  SUBROUTINE ACEINI()
    !
    !     INITIALIZATION OF ACE FORCE FIELD PARAMETERS
    !       o CHRAD = atom type charge radius
    !       o 1/OMEGA
    !       o 1/SIGMA^2
    !       o MUE^4
    !
  use exfunc
  use consta
  use inbnd
  use number
  use param
  use stream

    INTEGER I,K,IC,T1,T2
    real(chm_real),PARAMETER :: MINR=0.825_chm_real,VMIN=1.0E-8_chm_real
    real(chm_real)  FACT1,FACT3,FACT4,FACT5,FACT6
    real(chm_real)  AI,RQ,RV,VK,AL,AL3,BK,BK2,AI2BK2,QIK,AQIK
    real(chm_real)  QMAQ,OME,QUOT,SIG,MUE,MUE2,VQ
    !-----------------------------------------------------------------------
    !     FIRST PART, atom type charge radius (CHRAD):
    !
    !     set the charge radii equal to the vdW radii
    DO I=1,NATC
       IF(ITC(I) > 0) THEN
          CHRAD(I)=MAX(VDWR(ITC(I)),MINR)
       ENDIF
    ENDDO
    !
    ! set charge radii based on bond lengths and vdW radii of bonded atoms,
    ! Eqs. [25] & [39] of Schaefer & Karplus (1996) J. Phys. Chem. 100, 1578:
    DO I=1,NCB
       T1=INT(SQRT(TWO*KCB(I))+HALF)
       T2=KCB(I)-(T1*(T1-1))/2
       !       WRITE(OUTU,*) T1,T2,KCB(I),I,VDWR(ITC(T1)),VDWR(ITC(T2)),CBB(I)
       CHRAD(T1)=MAX(CHRAD(T1),VDWR(ITC(T2))-CBB(I))
       CHRAD(T2)=MAX(CHRAD(T2),VDWR(ITC(T1))-CBB(I))
    ENDDO
    !
    ! check that all charge radii are set and print them if PRNLEV>5:
    IF(PRNLEV > 5) WRITE(OUTU,10)
10  FORMAT(/9X,'CHARGE RADII PARAMETERS',/, &
         8X,'ITC   TYPE   CODE       RADIUS')
    DO I=1,NATC
!       IF(ITC(I) > 0.AND.CHRAD(I) /= VDWR(ITC(I)) &
       IF(ITC(I) > 0) THEN
        IF (CHRAD(I) /= VDWR(ITC(I)) &
            .AND.WRNLEV >= 6) THEN
          WRITE(OUTU,15) ATC(I),CHRAD(I),VDWR(ITC(I))
15        FORMAT(' ACEINI> ATOM TYPE ',A4, &
               ' CHARGE RADIUS ',F5.3,' INCREASED FROM RVDW ',F5.3)
        ENDIF
       ENDIF
       IF(PRNLEV > 5.AND.ITC(I) > 0) THEN
          WRITE(OUTU,20) ITC(I),ATC(I),I,CHRAD(I)
20        FORMAT(6X,I5,3X,A4,I7,F13.3)
       ENDIF
    ENDDO
    !.......................................................................
    !     SECOND PART, parameters of self energy potential (OME, SIG, MUE):
    !
    !     TAU=ONE/EPSI-ONE/EPSS  ! not needed, set tau=1 here
    FACT1=SQRT(PI/2.0)
    FACT3=SQRT(PI)/4.0
    FACT4=77.*PI*SQRT(2.)/512.
    !     FACT5=CCELEC*TAU/(8.0*PI) ! for hydrophobic energy, use tau=1
    FACT5=CCELEC/(8.0*PI)
    !
    !     test if atom volume defined for all types; set to zero
    !     and give warning for atom type with unassigned volume;
    !     volumes initially set to -ONE
    DO K=1,NATC
       IF((ITC(K) > 0).AND.(EFVOL(K) < ZERO)) THEN
          EFVOL(K)=ZERO
          IF(WRNLEV >= 2) WRITE(OUTU,50) ATC(K)
50        FORMAT(' ACEINI> WARNING: volume of ',A4, &
               ' missing in param.file -- set to zero.')
       ENDIF
    ENDDO
    !
    DO I=1,NATC
       IF(ITC(I) > 0) THEN
          RQ=CHRAD(I)
          AI=FACT1/RQ
          DO K=1,NATC
             IF(ITC(K) > 0) THEN
                !             option to scale atom volumes (Mar 2001):
                VK=FVSCAL*EFVOL(K)
                !             1997 version of ACE: volume radius = charge radius;
                !             equations of Schaefer, Bartels & Karplus, JMB 284 (1998), 835:
                RV=CHRAD(K)
                BK=ACEAL*RV
                BK=MAX(BK,RQ)
                BK2=BK*BK
                AI2BK2=AI*AI*BK2
                QIK=AI2BK2/SQRT(TWO*AI2BK2+ONE)
                AQIK=ATAN(QIK)
                QMAQ=QIK-AQIK
                IF(VK > VMIN) THEN
                   !               the OME here corresponds to omega/(8*PI), with omega
                   !               defined in eq (23) of the ACE paper:
                   OME=PI*BK2*BK2/(EIGHT*VK*QMAQ)
                   QUOT=(THREE*AI2BK2+ONE)/((AI2BK2+ONE)*(TWO*AI2BK2+ONE))
                   SIG2I(I,K)=((THREE+QUOT)*QIK-FOUR*AQIK)/(THREE*BK2*QMAQ)
                   SIG=SQRT(ONE/SIG2I(I,K))
                   !               FACT3 is FACT3/(8*PI) in the ACE paper, eq(27), because
                   !               the OME here is also scaled with 1/(8*PI) (see above)!
                   MUE=FACT4/((ONE/RQ) - FACT3*(SIG**3/VK)/OME)
                   MUE2=MUE*MUE
                   MUE4(I,K)=MUE2*MUE2
                   CES1(I,K)=FACT5/OME
                ELSE
                   CES1(I,K)=ZERO
                   SIG2I(I,K)=1.D-6
                   MUE4(I,K)=1.D+6
                ENDIF
                !             screening of charge i by volume i
                IF(I == K) THEN
                   VQ=(FOUR*PI/THREE)*RQ*RQ*RQ
                   ESII(I)=(VK-VQ)*QMAQ*EIGHT*FACT5/(PI*BK2*BK2)
                ENDIF
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    if (prnlev >= 2) WRITE(outu,*) 'NATC ',NATC
    DO I=1,NATC
       IF(ITC(I) > 0) THEN
          !         option to scale atom volumes (Mar 2001):
          CES2(I)=FVSCAL*EFVOL(I)*FACT5
          !         hydrophobic energy parameter, KHYD*DESELF = Chyd/b_i,
          !         conversion from self energy (with tau*q^2 = 1) to
          !         hydrophobic energy;
          !         CAVE: if SIGHYD is used as for ACE1 and assigned
          !         a value >= 0, then this supersedes the atom type
          !         dependent table that can be read via aceio.src from
          !         the acepar file; this option is kept for backward
          !         compatibility with ace1 and its input (e.g., testcase
          !         in ./test/c27test/ace1.inp):
          IF(SIGHYD >= ZERO) THEN
             FACT6=-EIGHT*PI*SIGHYD/CCELEC
          ELSE
             FACT6=-EIGHT*PI*ACEHYD(I)/CCELEC
          ENDIF
          RQ=CHRAD(I)
          KHYD(I)=FACT6*RQ*(RQ+RSOLV)**2
       ENDIF
    ENDDO
    IF(PRNLEV > 6) THEN
       WRITE(OUTU,70)
70     FORMAT('ACE PARAMETERS:',/, &
            'VOLUME/CHARGE ATOM',5X,'CES1',10X,'CES2',10X,'SIG2I',9X, &
            'MUE4',10X,'KHYD')
       DO K=1,NATC
          DO I=1,NATC
             IF(ITC(I) > 0.AND.ITC(K) > 0) THEN
                WRITE(OUTU,75) ATC(K),ATC(I),CES1(I,K),CES2(K), &
                     SIG2I(I,K),MUE4(I,K),KHYD(K)
75              FORMAT(2X,A4,5X,A4,3X,5(2X,G12.3))
             ENDIF
          ENDDO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE ACEINI
  !
  !
  !======================================================================
  !
  SUBROUTINE ENACE(ENB,EEL,LELECX,LVDWX,NATOMX, &
       JNBL,INBL,JNBL14,INBL14, &
#if KEY_BLOCK==1
       IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA, &         
#endif
       ESARR,ESFIXA, &
       CGIACE,CGSACE,DESDBS, &
       BSARR,DIARR,SARR,SA14,SWA14, &
       XFREAR,YFREAR,ZFREAR,XFSDAR,YFSDAR,ZFSDAR, &
       CG2ARR,DBDEAR,SWARR,DSWARR)
    !----------------------------------------------------------------------
    !     ACE electrostatic energy and non-polar solvation terms,
    !     for atom based cutoff scheme (ATOM, VATOM).
    !
    !     August 1996 -- July 1998, M. Schaefer & C. Bartels
    !            1999 -- Dec  2001, M. Schaefer
    !-----------------------------------------------------------------------
  use consta
  use exfunc
  use number
#if KEY_BLOCK==1
  use block_fcm         
#endif
  use coord
  use deriv
  use energym
  use inbnd
  use param
  use psf
  use stream
#if KEY_PARALLEL==1
  use parallel
#endif 

    real(chm_real)  ENB,EEL
    LOGICAL LELECX,LVDWX
    INTEGER NATOMX,JNBL(*),INBL(*),JNBL14(*),INBL14(*)
    !
    !     CES1,CES2,SIG2I,MUE4,KHYD,ESII = atom type dependent parameters
    !     of Eself and hydrophobic energy
    !
#if KEY_BLOCK==1
    INTEGER IBLOCK(*)
    real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#endif /*  BLOCK*/
    !
    !     ESARR = atomic solvation energy
    !     ESFIXA= self energy contributions of fixed atoms
    !     CGSACE= Charge used for interaction energy
    !     CGIACE= Charge used for self energy
    !     DESDBS= deriv. of self energy wrt (rescaled) Born radius
    !     BSARR = solvation radius
    !     DIARR = D_i array, where D_i = sum_j D_ij (see equation 47)
    !     SARR  = distance matrix, non-bonded atom pair ij
    !     SA14  = distance matrix, bonded atom pair ij
    !     FREAR = reaction force array (each atom)
    !     FSDAR = self-dielectric force matrix, atom pair ij
    !     CG2ARR= TAU * square of atomic charges (for self energy)
    !     SWARR = switching function, non-bonded atom pair ij
    !     SWA14 = switching function, bonded atom pair ij
    !     DSWARR= derivative of switching function
    !
    real(chm_real)  ESARR(*),BSARR(*),DIARR(*),SARR(*),CG2ARR(*), &
         DBDEAR(*)
    real(chm_real)  CGIACE(*),CGSACE(*),DESDBS(*)
    real(chm_real)  SA14(*),SWA14(*)
    real(chm_real)  XFREAR(*),YFREAR(*),ZFREAR(*)
    real(chm_real)  XFSDAR(*),YFSDAR(*),ZFSDAR(*)
    real(chm_real)  SWARR(*),DSWARR(*)
    real(chm_real)  ESFIXA(*)
    !
    !
    !     local variables:
    !
    INTEGER ITEMP,ITEMP14,NPR,NPRALL,I,J,KVECT,JVECT,IACI,IACJ
    INTEGER INDXIJ,INDXJI,INDX14
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real)  BLFACE
#endif /*  BLOCK*/
    real(chm_real)  S,S2
    real(chm_real)  TXIJ,TYIJ,TZIJ
    real(chm_real)  TAU,E,FACT1,FACT2,FACT3,FACT4,FACT5
    real(chm_real)  CFACT1,EC
    real(chm_real)  CFACT3,CBIJ2,CEXPO,CFEXP,CRIJ2,CRIJ
    real(chm_real)  BIJ2,FEXP,DIJ,DJI,RIJ,RIJ2,CGIJ
    real(chm_real)  FX,FY,FZ
    real(chm_real)  EEL0,ECOUL,EHYD
    real(chm_real)  EXPO,RSYS,ESMAX
    real(chm_real)  RUL3,RUL12,C2OFNB,C2ONNB,RIJL,RIJU,SW,DSW
    real(chm_real)  C2SOF,C2SON,SRUL3,SRUL12,OUTSID
    real(chm_real)  RTURN,ETURN,MEREF,DENOM,DELTB
    real(chm_real)  ETURNX,ETURNH,MEREFH,FACT2H,SMARAD
    !     SMARAD = small electrostatic radius, limit of using
    !              the max. RBorn TBSOLH instead of TBSOLV
    PARAMETER (SMARAD=1.39D0)
    real(chm_real)  ARGU,BSINCR,QUOT,LIMIT
    !     LIMIT = 17*LOG(10)+MAX(LOG(ACEA2(*))), used as the
    !             asymptotic limit of 1 for the self energy
    !             re-scaling function; ACEA2 is < 1 in general;
    !             LIMIT is used to prevent numerical overflow
    !             when exponentiating (EXP(LIMIT) still ok)
    PARAMETER (LIMIT=40.0D0)
    LOGICAL CSHIFT,CFSWIT,CSHFT,CSWIT,LOUTER,DIFFER
    !-----------------------------------------------------------------------
    ENB=ZERO
    EEL=ZERO
    IF(.NOT.(LELECX)) THEN
       ETERM(HYDR)=ZERO
       EPROP(SELF)=ZERO
       EPROP(SCREEN)=ZERO
       EPROP(COUL)=ZERO
       EPROP(SOLV)=ZERO
       EPROP(INTER)=ZERO
       RETURN
    ENDIF
    ECOUL=ZERO
    EHYD=ZERO
    !     check electrostatic switching function
    CSHIFT=      LSHFT .AND.      LFSWT
    CFSWIT= .NOT.LSHFT .AND.      LFSWT
    CSHFT =      LSHFT .AND. .NOT.LFSWT
    CSWIT = .NOT.LSHFT .AND. .NOT.LFSWT
    IF(CSHIFT.OR.CFSWIT.OR.CSHFT) THEN
       CALL WRNDIE(-2,'<ENACE>','Only electrostatic switch supported')
    ENDIF
    !     calculate factors:
    C2OFNB=CTOFNB*CTOFNB
    C2ONNB=CTONNB*CTONNB
    TAU=ONE/EPSI-ONE/EPSS
    !     self energy cutoff SCTOF must be <= CTOFNB (see nbutil)!
    C2SOF=SCTOF*SCTOF
    C2SON=SCTON*SCTON
    !     is self energy cutoff different from non-bonded cutoff?
    DIFFER=((SCTON /= CTONNB).OR.(SCTOF /= CTOFNB))
    !     calculate factors of switching functions:
    IF(CSWIT) THEN
       !       for charge interaction:
       IF(CTOFNB > CTONNB) THEN
          RUL3=ONE/(C2OFNB-C2ONNB)**3
          RUL12=TWELVE*RUL3
       ENDIF
       !       for self energy:
       IF(SCTOF > SCTON) THEN
          SRUL3=ONE/(C2SOF-C2SON)**3
          SRUL12=TWELVE*SRUL3
       ENDIF
    ENDIF
    !     initialize atom arrays (self energies, force buffers);
    !     self-dielectric force matrix not initialized because it is
    !     assigned in ESELFIK, and not incremented (unlike reaction
    !     force array FREAR); calculate system radius RSYS (relevant
    !     only if the ACE1 equation for avoiding divergence of BSOLV
    !     is used, as given in Schaefer et al., JMB 284, 835, eq A2);
    !     include factor TAU in q^2 array (taken out from self energy
    !     parameters CES1, CES2 to be able to calulate hydrophobic
    !     potential even for epsi=epss --> tau=0):
    RSYS=ZERO
    DO I=1,NATOMX
       ESARR(I)=ESFIXA(I)
       XFREAR(I)=ZERO
       YFREAR(I)=ZERO
       ZFREAR(I)=ZERO
       DIARR(I)=ZERO
       CG2ARR(I)=TAU*CGSACE(I)*CGSACE(I)
#if KEY_BLOCK==1
       IF(QBLOCK) THEN
          IBL=IBLOCK(I)
          KK=IBL*(IBL-1)/2+IBL
          RSYS=RSYS+FVSCAL*EFVOL(IAC(I))*BLCOE(KK)
       ELSE
#endif 
          RSYS=RSYS+FVSCAL*EFVOL(IAC(I))
#if KEY_BLOCK==1
       ENDIF  
#endif
    ENDDO
    RSYS=(THREE*RSYS/(FOUR*PI))**(THIRD)
    IF(RSYS <= ZERO) THEN
       RSYS=TENM5
       CALL WRNDIE(-2,'<ENACE>', &
            'Volume of system is ZERO, pointless to use ACE.')
    ENDIF
    !..............................................................................
    !     FIRST loop over atom pairs (non-bonded only, NO 1-2, 1-3 pairs):
    OUTSID=CTOFNB+ONE
    INDXIJ=0
    INDXJI=INBL(NATOMX)
    DO I=1,NATOMX
       IACI=IAC(I)
       IF(I > 1) THEN
          ITEMP=INBL(I-1)
       ELSE
          ITEMP=0
       ENDIF
#if KEY_IMCUBES==1
       IF(lbycbim) ITEMP=INBL(I+NATOMX)              
#endif
       NPR=INBL(I)-ITEMP
#if KEY_BLOCK==1
       IF(QBLOCK) IBL=IBLOCK(I)  
#endif
       DO J=1,NPR
          INDXIJ=INDXIJ+1
          INDXJI=INDXJI+1
          KVECT=JNBL(ITEMP+J) 
          !         fixed atom pairs do not show in nb-list, no need to check!
          JVECT=ABS(KVECT)
          CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
               X(I),Y(I),Z(I),X(JVECT),Y(JVECT),Z(JVECT))
          S2=MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
          IF(S2 >= C2OFNB) THEN
             !           atom pair outside non-bonded cutoff:
             SARR(INDXIJ)=OUTSID
          ELSE
             SARR(INDXIJ)=SQRT(S2)
             !           switching function for non-bonded cutoff:
             LOUTER=(S2 > C2ONNB)
             IF(CSWIT.AND.LOUTER) THEN
                RIJL=C2ONNB-S2
                RIJU=C2OFNB-S2
                SW =RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                DSW=RIJL*RIJU*RUL12
             ELSE
                SW=ONE
                DSW=ZERO
             ENDIF
             !           memorize data for the next non-bonded atom pair loop:
             SWARR(INDXIJ) =SW
             DSWARR(INDXIJ)=DSW
             !           switching function for self energy (if different):
             IF(DIFFER) THEN
                !             skip if outside self energy cutoff:
                IF(SARR(INDXIJ) >= SCTOF) GOTO 90
                LOUTER=(S2 > C2SON)
                IF(CSWIT.AND.LOUTER) THEN
                   RIJL=C2SON-S2
                   RIJU=C2SOF-S2
                   SW =RIJU*RIJU*(RIJU-THREE*RIJL)*SRUL3
                   DSW=RIJL*RIJU*SRUL12
                ELSE
                   SW=ONE
                   DSW=ZERO
                ENDIF
             ENDIF
#if KEY_BLOCK==1
             IF(QBLOCK) THEN
                JBL=IBLOCK(JVECT)
                KK=MAX(IBL,JBL)
                KK=KK*(KK-1)/2+MIN(IBL,JBL)
                !             scale electrostatic energy with BLCOE(KK)
                BLFACE=BLCOE(KK)
                !             scale van der Waals energy with BLCOV(KK)
             ELSE
                BLFACE=ONE
             ENDIF
#endif /*  BLOCK*/
             IACJ=IAC(JVECT)
             !           skip if atom JVECT has zero volume:
             IF(CES2(IACJ) > ZERO) THEN
                CALL ESELFIK(ESARR(I), &
                     XFREAR(I),YFREAR(I),ZFREAR(I), &
                     XFSDAR(INDXIJ),YFSDAR(INDXIJ),ZFSDAR(INDXIJ), &
                     TXIJ,TYIJ,TZIJ, &
#if KEY_BLOCK==1
                     BLFACE,                & 
#endif
                     CES1(IACI,IACJ),CES2(IACJ), &
                     SIG2I(IACI,IACJ),MUE4(IACI,IACJ), &
                     SARR(INDXIJ),S2,SW,DSW)
                !           ELSE ! never used
                !             XFSDAR(INDXIJ)=ZERO
                !             YFSDAR(INDXIJ)=ZERO
                !             ZFSDAR(INDXIJ)=ZERO
             ENDIF
             !           skip if atom I has zero volume:
             IF(CES2(IACI) > ZERO) THEN
                CALL ESELFIK(ESARR(JVECT), &
                     XFREAR(JVECT),YFREAR(JVECT),ZFREAR(JVECT), &
                     XFSDAR(INDXJI),YFSDAR(INDXJI),ZFSDAR(INDXJI), &
                     -TXIJ,-TYIJ,-TZIJ, &
#if KEY_BLOCK==1
                     BLFACE,                & 
#endif
                     CES1(IACJ,IACI),CES2(IACI), &
                     SIG2I(IACJ,IACI),MUE4(IACJ,IACI), &
                     SARR(INDXIJ),S2,SW,DSW)
                !           ELSE ! never used
                !             XFSDAR(INDXJI)=ZERO
                !             YFSDAR(INDXJI)=ZERO
                !             ZFSDAR(INDXJI)=ZERO
             ENDIF
             !           the reaction and self-dielectric forces will be
             !           assigned later, together with the interaction energy
             !           related forces (Coulomb + GB) and hydrophobic forces
          ENDIF  ! atom pair within non-bonded cutoff
90        CONTINUE
       ENDDO
    ENDDO
#if KEY_PARALLEL==1
    !     add up atom-wise quantities
    CALL GCOMB(ESARR,NATOMX)
#endif 
    !..............................................................................
    !     FACT1 without tau, which is at this point not included
    !     in the calculated self energies ESARR:
    FACT1=-CCELEC/TWO
    DO I=1,NATOMX
       !       include contribution of volume i in self energy of charge i,
       !       add Born solvation energy of charge i to obtain atomic solvation
       !       free energy:
#if KEY_BLOCK==1
       IF(QBLOCK) THEN
          IBL=IBLOCK(I)
          KK=IBL*(IBL-1)/2+IBL
          ESARR(I)=ESARR(I)+(FACT1/CHRAD(IAC(I))+ESII(IAC(I)))*BLCOE(KK)
       ELSE
#endif 
          ESARR(I)=ESARR(I)+FACT1/CHRAD(IAC(I))+ESII(IAC(I))
#if KEY_BLOCK==1
       ENDIF  
#endif
    ENDDO
    !
    !..............................................................................
    !     Calculate the Born radii (BSARR):
    !     cave: DBDEAR is the negative derivative d(BSARR)/d(ESARR)!
    IF(LACE2) THEN
       !       ACE2 version of calculating the Born radius BSARR from the self
       !       energy ESARR, which limits BSARR to values < MXBSOL:
       RTURN=TBSOLV
       ETURN=FACT1/RTURN
       MEREF=-ETURN*MXBSOL/TBSOLV
       FACT2=ONE-MXBSOL/TBSOLV
       FACT2=FACT1*FACT2*FACT2
       !       and for small atoms (hydrogens):
       RTURN=TBSOLH
       ETURNH=FACT1/RTURN
       MEREFH=-ETURNH*MXBSOL/TBSOLH
       FACT2H=ONE-MXBSOL/TBSOLH
       FACT2H=FACT1*FACT2H*FACT2H
       DO I=1,NATOMX
          IF(CHRAD(IAC(I)) > SMARAD) THEN
             ETURNX=ETURN
          ELSE
             ETURNX=ETURNH
          ENDIF
#if KEY_BLOCK==1
          IF(QBLOCK) THEN
             IBL=IBLOCK(I)
             KK=IBL*(IBL-1)/2+IBL
             IF(ESARR(I) <= (ETURNX*BLCOE(KK))) THEN
                BSARR(I)=BLCOE(KK)*FACT1/ESARR(I)
                DBDEAR(I)=BSARR(I)/ESARR(I)
             ELSEIF(CHRAD(IAC(I)) > SMARAD) THEN
                DENOM=ESARR(I)+BLCOE(KK)*MEREF
                DELTB=BLCOE(KK)*FACT2/DENOM
                BSARR(I)=MXBSOL+DELTB
                DBDEAR(I)=DELTB/DENOM
             ELSE
                !             atom with small electrostatic radius (hydrogen):
                DENOM=ESARR(I)+BLCOE(KK)*MEREFH
                DELTB=BLCOE(KK)*FACT2H/DENOM
                BSARR(I)=MXBSOL+DELTB
                DBDEAR(I)=DELTB/DENOM
             ENDIF
          ELSE
#endif 
             IF(ESARR(I) <= ETURNX) THEN
                BSARR(I)=FACT1/ESARR(I)
                DBDEAR(I)=BSARR(I)/ESARR(I)
             ELSEIF(CHRAD(IAC(I)) > SMARAD) THEN
                DENOM=ESARR(I)+MEREF
                DELTB=FACT2/DENOM
                BSARR(I)=MXBSOL+DELTB
                DBDEAR(I)=DELTB/DENOM
             ELSE
                !           atom with small electrostatic radius (hydrogen):
                DENOM=ESARR(I)+MEREFH
                DELTB=FACT2H/DENOM
                BSARR(I)=MXBSOL+DELTB
                DBDEAR(I)=DELTB/DENOM
             ENDIF
#if KEY_BLOCK==1
          ENDIF  
#endif
       ENDDO
    ELSE
       !       OLD (ACE1) version of calculating the Born radii BSARR from
       !       the self energy ESARR as given in JMB 284, 835--847 (eq A2);
       !       calculate derivative DBDEAR of BSARR with respect to
       !       the atomic solvation free energy;
       ESMAX=FACT1/RSYS
       FACT2=RSYS/ESMAX
       DO I=1,NATOMX
#if KEY_BLOCK==1
          IF(QBLOCK) THEN
             IBL=IBLOCK(I)
             KK=IBL*(IBL-1)/2+IBL
             IF(ESARR(I) <= (ESMAX*BLCOE(KK))) THEN
                BSARR(I)=BLCOE(KK)*FACT1/ESARR(I)
                DBDEAR(I)=BSARR(I)/ESARR(I)
             ELSE
                BSARR(I)=RSYS*(TWO-ESARR(I)/(ESMAX*BLCOE(KK)))
                DBDEAR(I)=FACT2/BLCOE(KK)
                IF(WRNLEV >= 6) THEN
                   WRITE(OUTU,100) I,IAC(I),BSARR(I),RSYS
100                FORMAT('ENACE> WARNING, RBORN>RSYS ', &
                        ' IATOM=',I6,' IAC=',I4,' BSARR=',F6.1,' RSYS=',F6.1)
                ENDIF
             ENDIF
          ELSE
#endif 
             IF(ESARR(I) <= ESMAX) THEN
                BSARR(I)=FACT1/ESARR(I)
                DBDEAR(I)=BSARR(I)/ESARR(I)
             ELSE
                BSARR(I)=RSYS*(TWO-ESARR(I)/ESMAX)
                DBDEAR(I)=FACT2
                IF(WRNLEV >= 6) THEN
                   WRITE(OUTU,110) I,IAC(I),BSARR(I),RSYS
110                FORMAT('ENACE> WARNING, RBORN>RSYS ', &
                        ' IATOM=',I6,' IAC=',I4,' BSARR=',F6.1,' RSYS=',F6.1)
                ENDIF
             ENDIF
#if KEY_BLOCK==1
          ENDIF  
#endif
       ENDDO
    ENDIF
    !..............................................................................
    !     if cooperative self energy correction (ACE3) is on,
    !     rescale the calculated Born solvation radii using ACEA0/1/2;
    !     change DBDEAR accordingly:
    IF(LACE3) THEN
       DO I=1,NATOMX
          BSINCR=BSARR(I)-CHRAD(IAC(I))
          ARGU=ACEA1(IAC(I))*BSINCR/CHRAD(IAC(I))
          !         to avoid numerical overflow for large ARGU, check for
          !         the limit where the rescaling approaches 1 asymptotically:
          IF((ARGU < LIMIT).AND.(ACEA2(IAC(I)) /= ZERO)) THEN
             FEXP=EXP(ARGU)
             QUOT=FEXP/(ACEA2(IAC(I))+FEXP)
             !           ACEA0 = ONE+a0 (see iniall.src and aceio.src):
             BSARR(I)=CHRAD(IAC(I))*ACEA0(IAC(I))+BSINCR*QUOT
             DBDEAR(I)=DBDEAR(I)*(QUOT+ARGU*(QUOT-QUOT*QUOT))
          ELSE
             !           ACEA0 = ONE+a0 (see iniall.src and aceio.src):
             BSARR(I)=CHRAD(IAC(I))*ACEA0(IAC(I))+BSINCR
             !           DBDEAR(I) remains unchanged (slope 1)
          ENDIF
       ENDDO
    ENDIF
    !..............................................................................
    !     loop over all atoms, sum up self energy, hydrophobic energy;
    !     calculate hydrophobic energy and forces from the ORIGINAL
    !     ACE1-type ESARR (this prevents the disappearance of hydrophobic
    !     forces on buried atoms, for which DESDBS tends to zero):
    DO I=1,NATOMX
       !       hydrophobic effect
#if KEY_PARALLEL==1
       !       ESARR(I) is already summed up; do not sum again
       IF(MYNOD == 0) THEN
#endif 
          EHYD=EHYD+KHYD(IAC(I))*ESARR(I)
#if KEY_PARALLEL==1
       ENDIF  
#endif
       IF(LACE2) THEN
          !         different from ACE1 of 1997, the self energy contribution
          !         is re-calculated from the rescaled Born solvation radius BSARR
          !         and thus always negative (as should be);
#if KEY_BLOCK==1
          IF(QBLOCK) THEN
             IBL=IBLOCK(I)
             KK=IBL*(IBL-1)/2+IBL
             ESARR(I)=BLCOE(KK)*FACT1/BSARR(I)
          ELSE
#endif /*  BLOCK*/
             ESARR(I)=FACT1/BSARR(I)
#if KEY_BLOCK==1
          ENDIF  
#endif
          !         need the derivative -d(ESARR)/d(BSARR) WITHOUT tau*q_i^2
          !         for dielectric and hydrophobic forces later on:
          DESDBS(I)=ESARR(I)/BSARR(I)
       ENDIF
       ESARR(I)=ESARR(I)*CG2ARR(I)
#if KEY_PARALLEL==1
       !       ESARR(I) is already summed up; do not sum again
       IF(MYNOD == 0) THEN
#endif 
          EEL=EEL+ESARR(I)
#if KEY_PARALLEL==1
       ENDIF  
#endif
    ENDDO
    ETERM(HYDR)=EHYD
    EPROP(SELF)=EEL
    EEL0=EEL
    !..............................................................................
    !     SECOND loop over all atom pairs (INCLUDING 1-2 and 1-3 interactions),
    !     calculate Coulomb energy and screening (generalized Born) energy:
    FACT1=-CCELEC*TAU
    CFACT1=CCELEC/EPSI
    INDXIJ=0
    !     INDXJI=INBL(NATOMX) ! not used in this loop
    DO I=1,NATOMX
       IF(I > 1) THEN
          ITEMP=INBL(I-1)
          ITEMP14=INBL14(I-1)
       ELSE
          ITEMP=0
          ITEMP14=0
       ENDIF
#if KEY_IMCUBES==1
       IF(lbycbim) ITEMP=INBL(I+NATOMX)          
#endif
       NPR=INBL(I)-ITEMP
       NPRALL=NPR+INBL14(I)-ITEMP14
#if KEY_PARALLEL==1
       !       bonded atom pair interactions should be calculated only once:
       IF(MOD(I,NUMNOD) /= MYNOD) THEN
          NPRALL=NPR
       ENDIF
#endif 
       ITEMP14=ITEMP14-NPR
#if KEY_BLOCK==1
       IF(QBLOCK) IBL=IBLOCK(I)  
#endif
       !
       !       everything that is calculated in this pair loop (the increment
       !       of EEL, DIJ, DJI, and forces) vanishes if Q_i or Q_j=0;
       !       skip if Q_i is zero:
       IF(ABS(CGIACE(I)) == ZERO) THEN
          INDXIJ = INDXIJ+NPR
          !         INDXJI = INDXJI+NPR ! not used in this loop
       ELSE
          DO J=1,NPRALL
             IF(J <= NPR) THEN
                !             nonbonded atom pair:
                KVECT=JNBL(ITEMP+J)
                JVECT=ABS(KVECT)
                !             fixed atom pairs do not show in nb-list, no need to check!
                INDXIJ=INDXIJ+1
                !             INDXJI=INDXJI+1 ! not used in this loop
                S=SARR(INDXIJ)
                SW=SWARR(INDXIJ)
                !             DSW=DSWARR(INDXIJ) ! assigned later
             ELSE
                !             bonded (1-2, 1-3) atom pair -- use known distance
                !             calculated in IESFIX (ideal, or current distance):
                INDX14=ITEMP14+J
                JVECT=JNBL14(INDX14)
                !             JVECT < 0 indicates 1-4 interaction, treated as non-bonded
                !             atom pair; don't calculate twice:
                IF(JVECT < 0) GOTO 200
                !             skip pairs of fixed atoms (treated in IESFIX):
                IF((IMOVE(JVECT) > 0).AND.(IMOVE(I) > 0)) GOTO 200
                S=SA14(INDX14)
                SW=SWA14(INDX14)
                !             DSW=DSWA14(INDX14) ! not needed (r_ij treated as fixed):
             ENDIF
             !
             IF(     (S < CTOFNB) &
                  .AND.(ABS(CGIACE(JVECT)) > ZERO) ) THEN
                S2    = S*S
                BIJ2  = BSARR(I)*BSARR(JVECT)
                EXPO  = S2/(FOUR*BIJ2)
                FEXP  = EXP(-EXPO)
                RIJ2  = S2 + BIJ2*FEXP
                RIJ   = SQRT(RIJ2)
                CGIJ  = CGIACE(I)*CGIACE(JVECT)
                E     = FACT1*CGIJ/RIJ
#if KEY_BLOCK==1
                IF(QBLOCK) THEN
                   !               scale electrostatic energy, forces with BLCOE(KK):
                   JBL=IBLOCK(JVECT)
                   KK=MAX(IBL,JBL)
                   KK=KK*(KK-1)/2+MIN(IBL,JBL)
                   E=E*BLCOE(KK)
                ENDIF
#endif /*  BLOCK*/
                EEL   = EEL + E*SW
                FACT3 = E/RIJ2
                FACT4 = HALF*FACT3*(ONE+EXPO)*FEXP*SW
                !
                !             FACT4*BSARR(JVECT) is the neg. derivative d(E)/d(BSARR(I))
                !             DBDEAR is tau*q_i^2* the negative derivative d(BSARR)/d(ESARR)
                DIJ   = FACT4*BSARR(JVECT)*DBDEAR(I)
                DJI   = FACT4*BSARR(I)*DBDEAR(JVECT)
                DIARR(I)     = DIARR(I)     + DIJ
                DIARR(JVECT) = DIARR(JVECT) + DJI
                !
                IF(J <= NPR) THEN
                   !               field force (with Coulomb) for non-bonded pairs:
                   IF(LQGAUS) THEN
                      !                 Gaussian Coulomb option:
                      CBIJ2 = CHRAD(IAC(I))*CHRAD(IAC(JVECT))
                      CEXPO = S2/(FOUR*CBIJ2)
                      CFEXP = EXP(-CEXPO)
                      CRIJ2 = S2 + CBIJ2*CFEXP
                      CRIJ  = SQRT(CRIJ2)
                      EC    = CFACT1*CGIJ/CRIJ
                      IF(KVECT < 0) EC=EC*E14FAC
#if KEY_BLOCK==1
                      IF(QBLOCK) EC=EC*BLCOE(KK)  
#endif
                      CFACT3= EC/CRIJ2
                      FACT5 = PT25*(FACT3*(FEXP-FOUR)+CFACT3*(CFEXP-FOUR))
                   ELSE
                      !                 Point charge Coulomb option:
                      EC=CFACT1*CGIJ/S
                      IF(KVECT < 0) EC=EC*E14FAC
#if KEY_BLOCK==1
                      IF(QBLOCK) EC=EC*BLCOE(KK)    
#endif
                      FACT5 = FACT3*(PT25*FEXP-ONE)-EC/S2
                   ENDIF
                   !
                   !               switching:
                   ECOUL = ECOUL + EC*SW
                   FACT5 = SW*FACT5 + (E+EC)*DSWARR(INDXIJ)
                   !
                   !               symmetric (first) part of field-force from derivative
                   !               wrt r_ij -- only for nonbonded pairs, since r_ij is
                   !               kept fixed for bonded (1-2, 1-3) atom pairs;
                   !               distance already calculated, but the vector TXIJ needs
                   !               to be re-established; call DXYZPB again:
                   CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                        X(I),Y(I),Z(I),X(JVECT),Y(JVECT),Z(JVECT))
                   FX    = FACT5*TXIJ
                   FY    = FACT5*TYIJ
                   FZ    = FACT5*TZIJ
                   DX(I) = DX(I)+FX
                   DY(I) = DY(I)+FY
                   DZ(I) = DZ(I)+FZ
                   DX(JVECT) = DX(JVECT)-FX
                   DY(JVECT) = DY(JVECT)-FY
                   DZ(JVECT) = DZ(JVECT)-FZ
                ENDIF
             ENDIF
200          CONTINUE
          ENDDO
       ENDIF
    ENDDO
    EPROP(SCREEN)=EEL-EEL0
    EPROP(COUL)  =ECOUL
    !     EEL=EEL+ECOUL+EHYD ! separate EEL (electrostatic) and EHYD (hydrophobic)
    !                        ! energy terms (MS 12 Aug 1999)
    EEL=EEL+ECOUL
#if KEY_PARALLEL==1
    !     sum up atom wise properties and energy properties
    CALL GCOMB(DIARR,NATOMX)
    CALL GCOMB(EPROP(SELF),1)
    CALL GCOMB(EPROP(SCREEN),1)
    CALL GCOMB(EPROP(COUL),1)
#endif 
    EPROP(SOLV)=EPROP(SELF)+EPROP(SCREEN)
    EPROP(INTER)=EPROP(COUL)+EPROP(SCREEN)
    !..............................................................................
    !     loop over all atoms, assign reaction force plus focussing
    !     part of the field force on atom I, (1+D_i)*F^reac_i, and
    !     hydrophobic force:
    DO I=1,NATOMX
       !       reaction and sdiel forces are still for tau*q^2=1,
       !       whereas DIARR(I) already includes the factor tau*q_i^2;
       !       hydrophobic factor KHYD requires F^reac_i to be for tau*q^2=1;
       !       for ACE2, need to include the derivative
       !       d(ESARRx)/d(BSARRx)*d(BSARRx)/d(BSARR)*d(BSARR)/d(ESARR)
       !                = DESDBS(I)*DBDEAR(I)
       !              --- suffix "x" refers to RBorn=BSARR after scaling ---
       !       which is irrelevant if the self energy and hydrophobic energy
       !       are taken directly from the sum over ESELIK contributions (this
       !       is the case if LACE2=.FALSE.):
       IF(LACE2) THEN
          DIARR(I)=DIARR(I)+CG2ARR(I)*DESDBS(I)*DBDEAR(I)+KHYD(IAC(I))
       ELSE
          DIARR(I)=DIARR(I)+CG2ARR(I)+KHYD(IAC(I))
       ENDIF
       DX(I)=DX(I)+DIARR(I)*XFREAR(I)
       DY(I)=DY(I)+DIARR(I)*YFREAR(I)
       DZ(I)=DZ(I)+DIARR(I)*ZFREAR(I)
    ENDDO
    !..............................................................................
    !     THIRD loop over atom pairs (non-bonded only, NO 1-2, 1-3 pairs)
    !     for the assignment of the dielectric (self+int) forces:
    INDXIJ=0
    INDXJI=INBL(NATOMX)
    DO I=1,NATOMX
       IACI=IAC(I)
       IF(I > 1) THEN
          ITEMP=INBL(I-1)
       ELSE
          ITEMP=0
       ENDIF
#if KEY_IMCUBES==1
       IF(lbycbim) ITEMP=INBL(I+NATOMX)              
#endif
       NPR=INBL(I)-ITEMP
       DO J=1,NPR
          INDXIJ=INDXIJ+1
          INDXJI=INDXJI+1
          KVECT=JNBL(ITEMP+J) 
          JVECT=ABS(KVECT)
          IF(SARR(INDXIJ) < SCTOF) THEN
             !           atom pair within self energy cutoff:
             IACJ=IAC(JVECT)
             !           skip if atom JVECT has zero volume:
             IF(CES2(IACJ) > ZERO) THEN
                DX(JVECT) = DX(JVECT)+DIARR(I)*XFSDAR(INDXIJ)
                DY(JVECT) = DY(JVECT)+DIARR(I)*YFSDAR(INDXIJ)
                DZ(JVECT) = DZ(JVECT)+DIARR(I)*ZFSDAR(INDXIJ)
             ENDIF
             !           skip if atom I has zero volume:
             IF(CES2(IACI) > ZERO) THEN
                DX(I)     = DX(I)+DIARR(JVECT)*XFSDAR(INDXJI)
                DY(I)     = DY(I)+DIARR(JVECT)*YFSDAR(INDXJI)
                DZ(I)     = DZ(I)+DIARR(JVECT)*ZFSDAR(INDXJI)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE ENACE
  !
  !
  !======================================================================
  !
  SUBROUTINE ESELFIK(ESI,XFRI,YFRI,ZFRI,XFDK,YFDK,ZFDK, &
       DXIK,DYIK,DZIK, &
#if KEY_BLOCK==1
       BLFACE,               & 
#endif
       CES1,CES2,SIG2I,MUE4, &
       R,R2,SW,DSW)
    !----------------------------------------------------------------------
    !     o increment self energy ESI of UNIT charge I by contr. from volume K,
    !     o increment reaction force XFRI of UNIT charge I,
    !     o return self-dielectric force XFDK on atom (volume) K
    !           due to UNIT charge I
    !-----------------------------------------------------------------------
  use number

    real(chm_real)  ESI,XFRI,YFRI,ZFRI,XFDK,YFDK,ZFDK,DXIK,DYIK,DZIK
#if KEY_BLOCK==1
    real(chm_real)  BLFACE                          
#endif
    real(chm_real)  CES1,CES2,SIG2I,MUE4,R,R2,SW,DSW
    !
    real(chm_real) RHO,RHO2,FEXP,DENO,R4,FFK,FAC1,FAC2,FX,FY,FZ
    real(chm_real) EXPO,E
    !-----------------------------------------------------------------------
    R4   = R2*R2
    DENO = R4 + MUE4
    RHO  = R2*R/DENO
    RHO2 = RHO*RHO
    EXPO = R2*SIG2I
    FEXP = EXP(-EXPO)
    !
    FAC1 = CES1*FEXP
    FAC2 = CES2*RHO2*RHO
    !     ESI  = ESI + CES1*FEXP + CES2*RHO2*RHO2
    E    = FAC1 + FAC2*RHO
    !     ... * SW accounts for switching function
#if KEY_BLOCK==0
    ESI  = ESI + E * SW               
#endif
#if KEY_BLOCK==1
    ESI  = ESI + E * SW * BLFACE      
#endif
    !
    FFK  = -FOUR*FAC2*R*(THREE*MUE4-R4)/(DENO*DENO) &
         +TWO*FAC1*SIG2I 
    !     switching
#if KEY_BLOCK==0
    FFK  = FFK*SW - E*DSW             
#endif
#if KEY_BLOCK==1
    FFK  = (FFK*SW - E*DSW) * BLFACE  
#endif
    !
    XFDK = FFK*DXIK
    YFDK = FFK*DYIK
    ZFDK = FFK*DZIK
    XFRI = XFRI - XFDK
    YFRI = YFRI - YFDK
    ZFRI = ZFRI - ZFDK
    !
    RETURN
  END SUBROUTINE ESELFIK
  !      
  !
  !======================================================================
  !
  SUBROUTINE IESFIX(LELECX,NATOMX, &
#if KEY_BLOCK==1
       IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA, &        
#endif
       IOFF,CGIACE,CGSACE, &
       JNBL14,INBL14,SA14,SWA14,ESFIXA)
    !----------------------------------------------------------------------
    !     o calculate self energy contributions of nonbonded exclusion list
    !     o memorize distance, sw-function of nonbonded exclusion list
    !     o calculate self energy contributions of fixed atoms (w/ cutoff)
    !     o scale the net charge of ionic groups
    !----------------------------------------------------------------------
  use number
#if KEY_BLOCK==1
  use block_fcm            
#endif
  use coord
  use inbnd
  use param
  use stream
  use psf
  use code
#if KEY_PARALLEL==1
  use parallel
#endif 
#if KEY_MMFF==1 || KEY_CFF==1
  use ffieldm
#endif 
  use machutil,only:die

    !     parameters
    LOGICAL LELECX
    INTEGER NATOMX
    INTEGER JNBL14(*),INBL14(*)
    real(chm_real)  SA14(*),SWA14(*)
    real(chm_real)  CGIACE(*),CGSACE(*)
    !
    !     CES1,CES2,SIG2I,MUE4 = atom type dependent parameters of Eself
    !
    !  real(chm_real)  CES1(MAXATC,MAXATC),CES2(MAXATC)
    !  real(chm_real)  SIG2I(MAXATC,MAXATC),MUE4(MAXATC,MAXATC)
#if KEY_BLOCK==1
    INTEGER IBLOCK(*)
    real(chm_real)  BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#endif /*  BLOCK*/
    INTEGER IOFF(*)
    real(chm_real)  ESFIXA(*)
    !
    !
    !     local variables
    INTEGER I,J,JJ,IACI,IACJ,I1,I2,IK,JK
    real(chm_real)  RUL3,RUL12,C2OFNB,C2ONNB,RIJL,RIJU,SW,DSW,TAU,S,S2
    real(chm_real)  C2SOF,C2SON,SRUL3,SRUL12,OUTSID
    real(chm_real)  TXIJ,TYIJ,TZIJ,L1,L2
    LOGICAL CSHIFT,CFSWIT,CSHFT,CSWIT,LOUTER,FAILED,DIFFER
    !     dummy parameters for ESELFIK (not saved)
    real(chm_real)  XFREAR,YFREAR,ZFREAR,XFSDAR,YFSDAR,ZFSDAR
#if KEY_BLOCK==1
    INTEGER IBL,JBL,KK
    real(chm_real)  BLFACE
#endif /*  BLOCK*/
    INTEGER IS,IQ,NAT
    real(chm_real)  QGRP,XQ,YQ,ZQ,sumf
    real(chm_real)  SCAI,SCAS,SCAIM1,SCASM1
    real(chm_real),PARAMETER :: QSMALL=1.0E-3_chm_real
    INTEGER INDX14,ITEMP14,NPR,JVECT,NFIX,NFAIL
    INTEGER ICB1,ICB2,ICT2
    !     functions:
    !-----------------------------------------------------------------------
    IF(.NOT.(LELECX)) RETURN
    !     check electrostatic switching function
    CSHIFT=      LSHFT .AND.      LFSWT
    CFSWIT= .NOT.LSHFT .AND.      LFSWT
    CSHFT =      LSHFT .AND. .NOT.LFSWT
    CSWIT = .NOT.LSHFT .AND. .NOT.LFSWT
    IF(CSHIFT.OR.CFSWIT.OR.CSHFT) THEN
       CALL WRNDIE(-2,'<ENACE>','Only electrostatic switch supported')
    ENDIF
    !     calculate factors:
    C2OFNB=CTOFNB*CTOFNB
    C2ONNB=CTONNB*CTONNB
    TAU=ONE/EPSI-ONE/EPSS
    !     self energy cutoff SCTOF must be <= CTOFNB (see nbutil)!
    C2SOF=SCTOF*SCTOF
    C2SON=SCTON*SCTON
    !     is self energy cutoff different from non-bonded cutoff?
    DIFFER=((SCTON /= CTONNB).OR.(SCTOF /= CTOFNB))
    !     calculate factors of switching functions:
    IF(CSWIT) THEN
       !       for charge interaction:
       IF(CTOFNB > CTONNB) THEN
          RUL3=ONE/(C2OFNB-C2ONNB)**3
          RUL12=TWELVE*RUL3
       ENDIF
       !       for self energy:
       IF(SCTOF > SCTON) THEN
          SRUL3=ONE/(C2SOF-C2SON)**3
          SRUL12=TWELVE*SRUL3
       ENDIF
    ENDIF
    !     initialize array:
    DO I=1,NATOMX
       ESFIXA(I)=ZERO
    ENDDO
    !     loop over all atom pairs with both atoms fixed, except atom pairs
    !     in the nonbonded exclusion list (with JVECT>0):
    NFIX=0
    DO I=1,NATOMX
       IF(IMOVE(I) > 0) THEN
          NFIX=NFIX+1
#if KEY_PARALLEL==1
          !         calculate fixed atom pair interaction only once for all nodes:
          IF(MOD(I,NUMNOD) /= MYNOD) GOTO 610
#endif 
          IACI=IAC(I)
#if KEY_BLOCK==1
          IF(QBLOCK) IBL=IBLOCK(I)  
#endif
          !         setup loop over nonbonded exclusion partners of atom I:
          IF(I > 1) THEN
             ITEMP14=INBL14(I-1)
          ELSE
             ITEMP14=0
          ENDIF
          NPR=INBL14(I)-ITEMP14
          DO J=I+1,NATOMX
             IF(IMOVE(J) > 0) THEN
                !             calculate self energy contribution of fixed pair i-j;
                !             first check whether this is a pair in the nonbonded
                !             exclusion list, which is taken care of below and
                !             must, therefore, be skipped here:
                DO JJ=1,NPR
                   INDX14=ITEMP14+JJ
                   JVECT=JNBL14(INDX14)
                   !               JVECT < 0 indicates 1-4 interaction (i.e., non-bonded)
                   !               which should be treated here:
                   IF(JVECT == J) GOTO 600 
                ENDDO
                CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                     X(I),Y(I),Z(I),X(J),Y(J),Z(J))
                S2=MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
                IF(S2 <= C2SOF) THEN
                   S=SQRT(S2)
                   !               calc switching function
                   LOUTER=(S2 > C2SON)
                   IF(CSWIT.AND.LOUTER) THEN
                      RIJL=C2SON-S2
                      RIJU=C2SOF-S2
                      SW =RIJU*RIJU*(RIJU-THREE*RIJL)*SRUL3
                      DSW=RIJL*RIJU*SRUL12
                   ELSE
                      SW=ONE
                      DSW=ZERO
                   ENDIF
#if KEY_BLOCK==1
                   IF(QBLOCK) THEN
                      JBL=IBLOCK(J)
                      KK=MAX(IBL,JBL)
                      KK=KK*(KK-1)/2+MIN(IBL,JBL)
                      !                 scale electrostatic energy with BLCOE(KK)
                      BLFACE=BLCOE(KK)
                      !                 scale van der Waals energy with BLCOV(KK)
                   ELSE
                      BLFACE=ONE
                   ENDIF
#endif /*  BLOCK*/
                   IACJ=IAC(J)
                   !               skip if atom J has zero volume:
                   IF(CES2(IACJ) > ZERO) &
                        CALL ESELFIK(ESFIXA(I), &
                        XFREAR,YFREAR,ZFREAR, &
                        XFSDAR,YFSDAR,ZFSDAR, &
                        TXIJ,TYIJ,TZIJ, &
#if KEY_BLOCK==1
                        BLFACE,                & 
#endif
                        CES1(IACI,IACJ),CES2(IACJ), &
                        SIG2I(IACI,IACJ),MUE4(IACI,IACJ), &
                        S,S2,SW,DSW)
                   !               skip if atom I has zero volume:
                   IF(CES2(IACI) > ZERO) &
                        CALL ESELFIK(ESFIXA(J), &
                        XFREAR,YFREAR,ZFREAR, &
                        XFSDAR,YFSDAR,ZFSDAR, &
                        -TXIJ,-TYIJ,-TZIJ, &
#if KEY_BLOCK==1
                        BLFACE,                & 
#endif
                        CES1(IACJ,IACI),CES2(IACI), &
                        SIG2I(IACJ,IACI),MUE4(IACJ,IACI), &
                        S,S2,SW,DSW)
                ENDIF
             ENDIF
600          CONTINUE
          ENDDO
       ENDIF
610    CONTINUE
    ENDDO
    !.......................................................................
    !     calculate self energy contribution, distance (stored in SA14)
    !     and switching function ("normal cutoff", stored in SWA14) for
    !     the nonbonded exclusion list (1-2, 1-3, and 1-4 in aromatic groups);
    !     store ESelf contribution by incrementing ESFIXA;
    !     for non-bonded exclusion list, forget the forces and DSW
    !     because the distances between these atom pairs are treated
    !     as invariant:
    NFAIL=0
    OUTSID=CTOFNB+ONE
    !
    IF(LIDEAL) THEN
       !       some elements in the ICB, ICT arrays may be undefined (set to 0),
       !       e.g., when the participating atoms are fixed; need to set up
       !       auxiliary array to search for bond lengths, angles in param.f90:
#if KEY_MMFF==1
       IF(FFIELD == MMFF) &
            CALL WRNDIE(1,'<IESFIX>', &
            'ACE with LIDEAL may be incompatible with MMFF.')
#endif 
#if KEY_CFF==1
       !       In the cff forcefield there are wildcard parameters designated with
       !       atom types of 0.  To allow for atom types of 0 the offset array has
       !       to be stretched out.
       IF(FFIELD == CFF) THEN
          I2=1
          DO I=1,NATC
             IOFF(I)=I2
             I2=I2+I+1
          ENDDO
       ELSE
#endif 
          DO I=1,NATC
             IOFF(I)=(I*(I-1))/2
          ENDDO
#if KEY_CFF==1
       ENDIF  
#endif
    ENDIF
    !
    DO I=1,NATOMX
#if KEY_PARALLEL==1
       !       calculate bonded atom pair interaction only once for all nodes:
       IF(MOD(I,NUMNOD) /= MYNOD) GOTO 810
#endif 
       IACI=IAC(I)
       IF(I > 1) THEN
          ITEMP14=INBL14(I-1)
       ELSE
          ITEMP14=0
       ENDIF
       NPR=INBL14(I)-ITEMP14
#if KEY_BLOCK==1
       IF(QBLOCK) IBL=IBLOCK(I)  
#endif
       DO J=1,NPR
          INDX14=ITEMP14+J
          JVECT=JNBL14(INDX14)
          !         JVECT < 0 indicates 1-4 interaction, treated as non-bonded
          !         atom pair; don't calculate here:
          IF(JVECT < 0) GOTO 800 
          !         do NOT skip pairs of fixed atoms because these were excluded
          !         in the loop above; this ensures that the nonbonded exclusion
          !         list atom pairs are treated with ideal distance if LIDEAL is
          !         invoked, even if both atom are fixed:
          FAILED=.TRUE.
          IF(LIDEAL) THEN
             !           calculate the distance according to ideal bond length
             !           and angles; no need to determine vector TXIJ,TYIJ,TZIJ
             !           since forces returned by ESELFIK, below, are ignored:
             CALL ACEIDX(I,JVECT,IB,JB,1,NBOND,I1)
             CALL ACEIDX(I,JVECT,IT,KT,1,NTHETA,I2)
             IF((I1 > 0).AND.(I2 > 0)) THEN
                IF(WRNLEV >= 2) THEN
                   WRITE(OUTU,700) I,JVECT,I1,I2
700                FORMAT(' IESFIX> WARNING: atom pair ',2I6, &
                        ' matches bond No ',I6,' and angle No ',I6)
                   WRITE(OUTU,710)
710                FORMAT(' IESFIX> using bond to set distance.')
                ENDIF
             ENDIF
             IF(I1 > 0) THEN
                !             I,JVECT is a 1-2 pair:
                ICB1=ICB(I1)
                IF(ICB1 == 0) THEN
                   !               ICB(I1) undefined (e.g., when I, JVECT fixed),
                   !               must search in param.f90:
                   ICB1=PRMICB(I,JVECT,IOFF)
                ENDIF
                IF(ICB1 > 0) THEN
                   S=CBB(ICB1)
                   S2=S*S
                   FAILED=.FALSE.
                ENDIF
                !             WRITE(OUTU,'(A,3I6,F9.4)') ' IESFIX> BOND ',I,JVECT,I1,S
             ELSEIF(I2 > 0) THEN
                !             I,JVECT is a 1-3 pair:
                CALL ACEIDX(I,JT(I2),IB,JB,1,NBOND,IK)
                IF(IK <= 0) CALL WRNDIE(-2,'<IESFIX>', &
                     'Did not find bond for angle I-J atom pair.')
                CALL ACEIDX(JT(I2),JVECT,IB,JB,1,NBOND,JK)
                IF(JK <= 0) CALL WRNDIE(-2,'<IESFIX>', &
                     'Did not find bond for angle J-K atom pair.')
                IF((IK > 0).AND.(JK > 0)) THEN
                   ICB1=ICB(IK)
                   IF(ICB1 == 0) THEN
                      !                 ICB(IK) undefined, search in param.f90:
                      ICB1=PRMICB(I,JT(I2),IOFF)
                   ENDIF
                   ICB2=ICB(JK)
                   IF(ICB2 == 0) THEN
                      !                 ICB(JK) undefined, search in param.f90:
                      ICB2=PRMICB(JT(I2),JVECT,IOFF)
                   ENDIF
                   ICT2=ICT(I2)
                   IF(ICT2 == 0) THEN
                      !                 ICT(I2) undefined, search in param.f90:
                      ICT2=PRMICT(I,JT(I2),JVECT,IOFF)
                   ENDIF
                   IF((ICB1 > 0).AND.(ICB2 > 0).AND.(ICT2.GT.0)) THEN
                      L1=CBB(ICB1)
                      L2=CBB(ICB2)
                      S2=L1*L1+L2*L2-TWO*L1*L2*COS(CTB(ICT2))
                      S=SQRT(S2)
                      FAILED=.FALSE.
                   ENDIF
                   !               WRITE(OUTU,'(A,3I6,F9.4)') ' IESFIX> THET ',I,JVECT,I2,S
                ENDIF
             ENDIF
          ENDIF
          IF(FAILED) THEN
             !           use the current distance as a last resort (normal
             !           approach if LIDEAL=.FALSE., and currently also normal
             !           for 1-4 atom pairs in aromatic systems, which have
             !           a positive KVECT=JVECT entry in the nb-exclusion list):
             NFAIL=NFAIL+1
             CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                  X(I),Y(I),Z(I),X(JVECT),Y(JVECT),Z(JVECT))
             S2=MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
             S=SQRT(S2)
             !           IF(LIDEAL) WRITE(OUTU,'(A,3I6,F9.4)')
             !    &                 ' IESFIX> FAIL ',I,JVECT,0,S
          ENDIF
          IF(S2 >= C2OFNB) THEN
             !           atom pair outside non-bonded cutoff:
             SA14(INDX14)=OUTSID
             SWA14(INDX14)=ZERO  ! just to be sure
          ELSE
             SA14(INDX14)=S
             !           switching function for non-bonded cutoff (used in ENACE):
             LOUTER=(S2 > C2ONNB)
             IF(CSWIT.AND.LOUTER) THEN
                RIJL=C2ONNB-S2
                RIJU=C2OFNB-S2
                SW =RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                DSW=RIJL*RIJU*RUL12
             ELSE
                SW=ONE
                DSW=ZERO
             ENDIF
             SWA14(INDX14)=SW
             !
             !           no longer skip pairs of fixed atoms, because the
             !           pairs of atoms in the non-bonded exlusion list
             !           are excluded in the pair loop above:
             !           IF((IMOVE(JVECT) > 0).AND.(IMOVE(I) > 0)) GOTO 800
             !
             !           switching function for self energy (if different):
             IF(DIFFER) THEN
                !             skip if outside self energy cutoff:
                IF(S >= SCTOF) GOTO 800
                LOUTER=(S2 > C2SON)
                IF(CSWIT.AND.LOUTER) THEN
                   RIJL=C2SON-S2
                   RIJU=C2SOF-S2
                   SW =RIJU*RIJU*(RIJU-THREE*RIJL)*SRUL3
                   DSW=RIJL*RIJU*SRUL12
                ELSE
                   SW=ONE
                   DSW=ZERO
                ENDIF
             ENDIF
#if KEY_BLOCK==1
             IF(QBLOCK) THEN
                JBL=IBLOCK(JVECT)
                KK=MAX(IBL,JBL)
                KK=KK*(KK-1)/2+MIN(IBL,JBL)
                !             scale electrostatic energy with BLCOE(KK)
                BLFACE=BLCOE(KK)
                !             scale van der Waals energy with BLCOV(KK)
             ELSE
                BLFACE=ONE
             ENDIF
#endif /*  BLOCK*/
             IACJ=IAC(JVECT)
             !           skip if atom JVECT has zero volume:
             IF(CES2(IACJ) > ZERO) &
                  CALL ESELFIK(ESFIXA(I), &
                  XFREAR,YFREAR,ZFREAR, &
                  XFSDAR,YFSDAR,ZFSDAR, &
                  TXIJ,TYIJ,TZIJ, &
#if KEY_BLOCK==1
                  BLFACE,                & 
#endif
                  CES1(IACI,IACJ),CES2(IACJ), &
                  SIG2I(IACI,IACJ),MUE4(IACI,IACJ), &
                  S,S2,SW,DSW)
             !           skip if atom I has zero volume:
             IF(CES2(IACI) > ZERO) &
                  CALL ESELFIK(ESFIXA(JVECT), &
                  XFREAR,YFREAR,ZFREAR, &
                  XFSDAR,YFSDAR,ZFSDAR, &
                  -TXIJ,-TYIJ,-TZIJ, &
#if KEY_BLOCK==1
                  BLFACE,                & 
#endif
                  CES1(IACJ,IACI),CES2(IACI), &
                  SIG2I(IACJ,IACI),MUE4(IACJ,IACI), &
                  S,S2,SW,DSW)
             !           the reaction and self-dielectric forces for this
             !           bonded atom pair are ZERO (distance kept fixed),
             !           so the dummy vectors FREAR and FSDAR can be ignored
          ENDIF  ! atom pair within non-bonded cutoff
800       CONTINUE
       ENDDO
810    CONTINUE
    ENDDO
    IF((LIDEAL).AND.(WRNLEV >= 2).AND.(NFAIL > 0)) &
         WRITE(OUTU,820) NFAIL
820 FORMAT(' IESFIX> WARNING: distance between ',I8, &
         ' nb-exclusion pairs',/, &
         '         calculated from current coordinates')
    !.......................................................................
    !     scale the charges of ionic atom groups:
    IF((FISCAL == ONE).AND.(FSSCAL == ONE)) THEN
       !       there is nothing to scale:
       DO J=1,NATOM
          CGIACE(J)=CG(J)
          CGSACE(J)=CG(J)
       ENDDO
       GOTO 890
    ENDIF
    SCAI=SQRT(FISCAL)
    SCAS=SQRT(FSSCAL)
    SCAIM1=SCAI-ONE
    SCASM1=SCAS-ONE
    IF(PRNLEV > 5) WRITE(OUTU,870)
870 FORMAT(/9X,'SCALING OF IONIC GROUP CHARGES',/, &
         10X,'#      ATOM-RANGE        QGRP')
    DO I=1,NGRP
       IS=IGPBS(I)+1
       IQ=IGPBS(I+1)
       NAT=IQ-IS+1
       IF(NAT <= 0) CALL DIE
       QGRP=ZERO
       DO J=IS,IQ
          QGRP=QGRP+CG(J)
       ENDDO
       IF(ABS(QGRP) > QSMALL) THEN
          IF(PRNLEV > 5) WRITE(OUTU,880) I,IS,IQ,QGRP
880       FORMAT(I11,2(2X,I6),2X,F10.3)
          IF(NAT > 1) THEN
             !           determine the center of charge (use first atom in group
             !           as the spatial reference to deal with periodic boundary):
             XQ=CG(IS)*X(IS)
             YQ=CG(IS)*Y(IS)
             ZQ=CG(IS)*Z(IS)
             DO J=IS+1,IQ
                CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                     X(J),Y(J),Z(J),X(IS),Y(IS),Z(IS))
                XQ=XQ+CG(J)*(X(IS)+TXIJ)
                YQ=YQ+CG(J)*(Y(IS)+TYIJ)
                ZQ=ZQ+CG(J)*(Z(IS)+TZIJ)
             ENDDO
             XQ=XQ/QGRP
             YQ=YQ/QGRP
             ZQ=ZQ/QGRP
             !           determine 1/distance scaling factors for atoms (begin with
             !           their sum to avoid the need to store yet another atom array):
             sumf=ZERO
             DO J=IS,IQ
                CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                     X(J),Y(J),Z(J),XQ,YQ,ZQ)
                S2=MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
                sumf=sumf+ONE/SQRT(S2)
             ENDDO
             sumf=ONE/sumf
             DO J=IS,IQ
                CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
                     X(J),Y(J),Z(J),XQ,YQ,ZQ)
                S2=MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
                !             info: QGRP*((ONE/SQRT(S2))/sumf) = ionic part of CG(J)
                CGIACE(J)=CG(J)+SCAIM1*QGRP*(sumf/SQRT(S2))
                CGSACE(J)=CG(J)+SCASM1*QGRP*(sumf/SQRT(S2))
             ENDDO
          ELSE
             !           scale charge of single-atom ionic "group":
             CGIACE(IS)=SCAI*CG(IS)
             CGSACE(IS)=SCAS*CG(IS)
          ENDIF
       ELSE
          !         copy charges of polar atom group:
          DO J=IS,IQ
             CGIACE(J)=CG(J)
             CGSACE(J)=CG(J)
          ENDDO
       ENDIF
    ENDDO
890 CONTINUE
    !.......................................................................
    !
    RETURN
  END SUBROUTINE IESFIX
  !
  !
  !======================================================================
  !
  SUBROUTINE DXYZPB(TX,TY,TZ,X1,Y1,Z1,X2,Y2,Z2)
    !----------------------------------------------------------------------
  use number
#if KEY_PBOUND==1
  use pbound         
#endif

    real(chm_real)  X1,Y1,Z1,X2,Y2,Z2,TX,TY,TZ
#if KEY_PBOUND==1 /*pbound*/
    real(chm_real)  CORR
#endif /*     (pbound)*/
    !-----------------------------------------------------------------------
    TX=X1-X2
    TY=Y1-Y2
    TZ=Z1-Z2
#if KEY_PBOUND==1 /*pbound*/
    If(qBoun) then
       If(qCUBoun.or.qTOBoun) then
          TX      = BOXINV * TX
          TY      = BOYINV * TY
          TZ      = BOZINV * TZ
          tx = tx - nint(tx)
          ty = ty - nint(ty)
          tz = tz - nint(tz)
!!$          IF(TX >  HALF) TX = TX - ONE
!!$          IF(TX <  -HALF) TX = TX + ONE
!!$          IF(TY >  HALF) TY = TY - ONE
!!$          IF(TY <  -HALF) TY = TY + ONE
!!$          IF(TZ >  HALF) TZ = TZ - ONE
!!$          IF(TZ <  -HALF) TZ = TZ + ONE
          If(qTOBoun) Then
             CORR = HALF * AINT ( R75 * ( ABS ( TX ) + &
                  ABS ( TY ) + &
                  ABS ( TZ ) ) )
             TX = TX - SIGN ( CORR, TX )
             TY = TY - SIGN ( CORR, TY )
             TZ = TZ - SIGN ( CORR, TZ )
          Endif
          TX      = XSIZE * TX
          TY      = YSIZE * TY
          TZ      = ZSIZE * TZ
       Else
          Call PBMove(TX, TY, TZ)
       Endif
    Endif
#endif /*      (pbound)*/
    RETURN
  END SUBROUTINE DXYZPB
  !
  !      
  !======================================================================
  !
  SUBROUTINE ACEIDX(I,J,IAR,JAR,N0,NAR,IDX)
    !
    !     extremely primitive function to search for the integer
    !     pair I,J in the index arrays IAR,JAR; no assumption on
    !     sorting of the arrays is made, and the match can be
    !         I,J == IAR(IDX),JAR(IDX) or
    !         J,I == IAR(IDX),JAR(IDX).
    !     if there is no match, the subroutine returns IDX=0.
    !----------------------------------------------------------------------

    INTEGER I,J,IAR(*),JAR(*),N0,NAR
    INTEGER IDX
    !     local variables:
    !----------------------------------------------------------------------
    DO IDX=N0,NAR
       IF((I == IAR(IDX)).AND.(J == JAR(IDX))) RETURN
       IF((J == IAR(IDX)).AND.(I == JAR(IDX))) RETURN
    ENDDO
    IDX=0
    !
    RETURN
  END SUBROUTINE ACEIDX
  !
  !
  !======================================================================
  !
  INTEGER FUNCTION PRMICB(I,J,IOFF)
    !----------------------------------------------------------------------
    !     Search for ICB pointer to bond parameters of atom pair I-J.
    !
    !     January 2002, M. Schaefer
    !-----------------------------------------------------------------------
  use param
  use psf
#if KEY_MMFF==1 || KEY_CFF==1
  use ffieldm
  use mmffm
  use cff_fcm
#endif 

    INTEGER I,J,IOFF(*)
    INTEGER I2
    INTEGER I1,J1,NUMBER
    !     functions:
    INTEGER NINDX
    !----------------------------------------------------------------------
    PRMICB=0
    I1=IAC(I)
    J1=IAC(J)
#if KEY_CFF==1
    ! for cff use the equivalence types:
    IF(FFIELD == CFF) THEN
       I1=IBE(I1)
       J1=IBE(J1)
    ENDIF
#endif 
    IF(I1 <= 0.OR.J1 <= 0) RETURN
    IF(I1 > J1) THEN
       NUMBER=IOFF(I1)+J1
    ELSE
       NUMBER=IOFF(J1)+I1
    ENDIF
    PRMICB=NINDX(NUMBER,KCB,NCB)
#if KEY_CFF==1
    IF(FFIELD == CFF) THEN
       ! if explicit bond type not found then first look for bond order:
       IF(PRMICB == 0) THEN
          DO I2=NCB+1,NCBO
             IF(I1 == BID1(I2).AND.J1 == BID2(I2) &
                  .AND.BONDTYPE_cff(I) == BORD(I2)) PRMICB=I2
          ENDDO
          ! if bond type still not found try automatic equivalence types:
          IF(PRMICB == 0) THEN
             I1=IBAE(IAC(I))
             J1=IBAE(IAC(J))
             IF(I1 > J1) THEN
                NUMBER=IOFF(I1)+J1
             ELSE
                NUMBER=IOFF(J1)+I1
             ENDIF
             PRMICB=NINDX(NUMBER,KCB,NCB)
          ENDIF
       ENDIF
    ENDIF
#endif 
    !
    RETURN
  END FUNCTION PRMICB
  !
  !
  !======================================================================
  !
  INTEGER FUNCTION PRMICT(I,J,K,IOFF)
    !----------------------------------------------------------------------
    !     Search for ICT pointer to angle parameters of atom triple I-J-K.
    !
    !     January 2002, M. Schaefer
    !-----------------------------------------------------------------------
  use param
  use psf
#if KEY_MMFF==1 || KEY_CFF==1
  use ffieldm
  use mmffm
  use cff_fcm
#endif 

    INTEGER I,J,K,IOFF(*)
    INTEGER I2
    INTEGER I1,J1,K1,NO,NUMBER,ICTPT2
    !     functions:
    INTEGER NINDX
    !----------------------------------------------------------------------
    PRMICT=0
    I1=IAC(I)
    J1=IAC(J)
    K1=IAC(K)
#if KEY_CFF==1
    ! for cff use the equivalence atom types:
    IF(FFIELD == CFF) THEN
       I1=ITE(I1)
       J1=ITE(J1)
       K1=ITE(K1)
    ENDIF
#endif 
    IF(I1 <= 0.OR.J1 <= 0.OR.K1.LE.0) RETURN
    IF(I1 > K1) THEN
       NO=IOFF(I1)+K1
    ELSE
       NO=IOFF(K1)+I1
    ENDIF
    NUMBER=NATC*NO+J1
    PRMICT=NINDX(NUMBER,KCT,NCT)

#if KEY_CFF==1
    !     If a parameter is selected using automatic parameters it is assigned
    !     a -ve pointer.  In codes_cff another search will be made to see if
    !     a bond-ordered angle parameter should be used.
    IF(FFIELD == CFF) THEN
       IF(PRMICT == 0) THEN
          !         explicit angle type not found, try automatic equivalence types:
          I1=ITEAE(IAC(I))
          J1=ITAAE(IAC(J))
          K1=ITEAE(IAC(K))
          IF(I1 > K1) THEN
             NO=IOFF(I1)+K1
          ELSE
             NO=IOFF(K1)+I1
          ENDIF
          NUMBER=NATC*NO+J1
          PRMICT=-NINDX(NUMBER,KCT,NCT)
          IF(PRMICT == 0) THEN
             !           if angle type still not found, try wild card for atom i or k;
             !           if two wildcard types are found, select the one with
             !           the higher priority:
             NUMBER=NATC*IOFF(I1)+J1
             PRMICT=-NINDX(NUMBER,KCT,NCT)
             IF(I1 /= K1) THEN
                NUMBER=NATC*IOFF(K1)+J1
                ICTPT2=-NINDX(NUMBER,KCT,NCT)
                IF((PRMICT /= 0).AND.(ICTPT2 /= 0)) THEN
                   IF(TID1(-ICTPT2) > TID1(-PRMICT)) PRMICT=ICTPT2
                ELSEIF(PRMICT == 0) THEN
                   PRMICT=ICTPT2
                ENDIF
             ENDIF
             ! if angle type still not found try wild card for both atoms i and k:
             IF(PRMICT == 0) THEN
                NUMBER=J1
                PRMICT=-NINDX(NUMBER,KCT,NCT)
             ENDIF
          ENDIF
       ENDIF
    ENDIF
#endif 
    !
    RETURN
  END FUNCTION PRMICT
  !
  !
  !======================================================================
  !
  INTEGER FUNCTION GETNNB(L,N)

    INTEGER L(*),N
    !----------------------------------------------------------------------
    GETNNB=L(N)
    !
    RETURN
  END FUNCTION GETNNB
  !
  !
  !======================================================================
  !
  SUBROUTINE INIB(NATOMX,BSARR)
    !----------------------------------------------------------------------
    !     ace electrostatic energy terms
    !
    !     August 1996, M. Schaefer & C. Bartels
    !-----------------------------------------------------------------------
  use consta
  use psf
  use exfunc
  use number
  use param

    INTEGER NATOMX
    !
    !     BSARR = solvation radius,
    !
    real(chm_real)  BSARR(*)
    !
    !
    INTEGER I
    !-----------------------------------------------------------------------
    DO I=1,NATOMX
       BSARR(I)=CHRAD(IAC(I))
    ENDDO
    !
    return
  end SUBROUTINE INIB
#endif /* (ace)*/

end module ace_module

