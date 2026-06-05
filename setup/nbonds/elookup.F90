module lookup
  use chm_kinds
  private
  ! public routines
  public :: ELOOKUP,WWSETUP,WWSPLTNB,MKWWFLG
  ! public variables (some of the lists may be made internal, with resizing done locally)
  public :: QLOOKUP, QVU,QUU,QVV,IWWENR,EWWENBI,NWW,NWWO,NWWOIM,NNBBYCB
  public :: NNBIBYCB,MXJWWOO,MXJWWOOI,IWWO,IWWFLG,IWOONBL,JWOONBL,IVUNBL,JVUNBL
  public :: IWOONBLI,JWOONBLI,IVUNBLI,JVUNBLI,IUUNBL,JUUNBL,EWWEEL,EWWENB,EWWEELI
  public :: JUUNBLI,IUUNBLI,CTDDW
  !
  ! Lookup tables using R**2 indexing for non-bonded interactions based on 
  ! standard functions that are available in ENBAEXP.
  ! Should work with those methods that use ENBAEXP without changing anything
  ! (except the main coordinate arrays) between calls. 
  ! Works with BYGRoup and BYCBim list generators. Works with IMAGES, CRYSTAL, PME,
  ! also in parallel.
  ! 
  ! Lennart Nilsson, Karolinska Institutet, 2004-2007.
  !
  ! Currently supports solvent-solvent (three-site) and solvent-solute
  ! interactions using same cutoffs etc as for the rest (also real space PME)
  ! Support for TIP4P is in the works
  !     
  ! Data structures for the lookup table implementation of CHARMM's non-bonded energy routines
  ! Lennart Nilsson, Karolinska Institutet, 2004-2007. 
  !
  ! QLOOKUP      TRUE if this facility is to be used
  ! QVU        TRUE if also the solVent-solUte tables are to be used
  ! QUU        TRUE for solUte-solUte tables
  ! QVV        TRUE for solVent-solVent
  ! QVHLJ      TRUE if hydrogens (in CHARMM TIP3P) have L-J params
  ! QWLKUPINT  TRUE if the linear interpolation scheme is to be used. Default FALSE
  ! TBSIZ      Scalefactor for lookup table indexing: Tableindex=R2*TBSIZ
  ! RNDING     Index rounding for straight lookup (0.5) or interpolation (0.0)
  ! MXTB1      Size of each lookup table (=TBSIZ*CTOFNB*CTOFNB + 1)
  ! MXTB       MXTB1-1, index of last actual entry in table. Entry at MXTB1 is 0.0 and
  !                    is used to handle cutoff via MIN statement instead of IF
  ! NVVTAB     Number of VV tables needed for particular watermodel
  ! NVUTAB     Number of solVent-solUte LJ tables for each solvent LJ particle + 2Coulomb tables
  ! EVV,FVV    solVent-solVent Energy and Force tables
  ! E/FVULJ    solVent-solUte  Lennard-Jones Energy and Force tables
  ! E/FVUC     solVent-solUte  Coulomb Energy and Force tables
  ! VDWOFFS, ELEOFFS  Offsets for solVent-solUte LJ/Coulombtables
  ! WATMODEL   Type of watermodel being used. Three-site or TIP4P
  ! IWWENR     Controls use of energy lookup in addition to force lookup
  !            0 Never look up energy; 2 always do
  !            +1 ONLY AT UPDATES AND NOW
  !            -1 ONLY AT UPDATES AND WE ARE INBETWEEN NOW
  ! EWEEL,EWWWENB,EWEELI,EWWENBI  Stores energy values inbetween energy evalutations
  ! NWW       Number of solvent atoms involved in the lookup
  ! NWWO      Number of water molecules for the solvent-solvent lookup
  ! NWWOIM    Number of  water molecules for the solvent-solvent lookup, including IMAGE waters
  ! NNBBYCB,NNBIBYCB Nummber of primary and image non-bonded interactions found by NBNDGCM 
  ! MXJWWOO   Dimension of JWOONBL
  ! MXJWWOOI  Dimension of JWOONBLI
  ! MXJVU     Dimension of JVUNBL
  ! MXJVUI    Dimension of JVUNBLI
  ! IWWO      List of oxygens in the solvent
  ! IWWFLG    Flags selected solvent atoms
  ! IWOONBL, JWOONBL  Pair list (oxygen-oxygen for all solvent molecule pairs with any atom
  !           pair within CUTNB)
  ! IVUNBL, JVUNBL solVent-solUte atom-based pair-list
  ! IWOONBLI, JWOONBLI  Pair list (oxygen-oxygen for all solvent molecule pairs with any atom
  !           pair within CUTNB)  for Image interactions
  ! IVUNBLI, JVUNBLI solVent-solUte atom-based pair-list for Image interactions
  ! CTDDW     Distance margin used in energy routine when excluding molecule-pair from lookup
  !
  LOGICAL :: QLOOKUP,QVU,QVV,QUU,QVHLJ,QWLKUPINT
  INTEGER :: MXTB,MXTB1,NVVTAB,NVUTAB,WATMODEL,IWWENR
  INTEGER, PARAMETER :: W3SITE=1, WTIP4P=2
  ! Index into VU tables    
  INTEGER, PARAMETER :: IOX=1,ILP=1,IHYD=2
  INTEGER  :: NWW,NWWO,NWWOIM,MXJWWOO,MXJWWOOI
  INTEGER  :: MXJVU,MXJVUI,MXJUU,MXJUUI
  INTEGER, ALLOCATABLE :: IWOONBL(:),JWOONBL(:),IWWFLG(:),LKUPSEL(:),IWWO(:)
  INTEGER, ALLOCATABLE :: IVUNBL(:),JVUNBL(:)
  INTEGER, ALLOCATABLE :: IUUNBL(:),JUUNBL(:)
  INTEGER, ALLOCATABLE :: IWOONBLI(:),JWOONBLI(:)
  INTEGER, ALLOCATABLE :: IVUNBLI(:),JVUNBLI(:)
  INTEGER, ALLOCATABLE :: IUUNBLI(:),JUUNBLI(:)
  INTEGER, ALLOCATABLE :: VDWOFFS(:),ELEOFFS(:)
  ! Temporary arrays in wwspltnb; we don't want to allocate/reallocate all the time
  INTEGER MXVU
  INTEGER, ALLOCATABLE :: VUTMP(:,:),FRSTIDX(:),LASTIDX(:)
  INTEGER, ALLOCATABLE :: LNKLST(:,:)
  ! 
  INTEGER :: NNBBYCB,NNBIBYCB 
  REAL(chm_real4)    :: TBSIZ
  REAL(chm_real4)    :: EWWEEL,EWWENB,EWWEELI,EWWENBI
  REAL(chm_real8)    :: RNDING,CTDDW
  REAL(chm_real4), ALLOCATABLE :: EVV(:,:),FVV(:,:)
  REAL(chm_real4), ALLOCATABLE :: FVULJ(:,:,:),FVUC(:,:)
  REAL(chm_real4), ALLOCATABLE :: EVULJ(:,:,:),EVUC(:,:)
  REAL(chm_real4), ALLOCATABLE :: EUULJ(:,:,:),FUULJ(:,:,:),EUUC(:),FUUC(:)   

  !
contains

#if KEY_LOOKUP==0 /*lookup_main*/
  SUBROUTINE ELOOKUP(ENB8,EEL8,NWWOX,IWWOX,IWOONBLX,JWOONBLX, &
       NATOMX,IVUNBLX,JVUNBLX,IUUNBLX,JUUNBLX)
    real(chm_real)  ENB8,EEL8
    INTEGER  NWWOX,NATOMX,IWWOX(*),IWOONBLX(*),JWOONBLX(*)
    INTEGER  IVUNBLX(*),JVUNBLX(*),IUUNBLX(*),JUUNBLX(*)
    CALL WRNDIE(0,'<LOOKUP>','OPTION NOT COMPILED')
    RETURN
  END SUBROUTINE ELOOKUP
#else /* (lookup_main)*/
  SUBROUTINE ELOOKUP(ENB8,EEL8,NWWOX,IWWOX,IWOONBLX,JWOONBLX, &
       NATOMX,IVUNBLX,JVUNBLX,IUUNBLX,JUUNBLX)
    !
    ! Energy call dispatcher
    !
    use chm_kinds
    !  use LOOKUP
    use dimens_fcm
    use inbnd
    use parallel
    IMPLICIT NONE  
    real(chm_real)  ENB8,EEL8,ENB,EEL
    INTEGER  NWWOX,NATOMX,IWWOX(*),IWOONBLX(*),JWOONBLX(*)
    INTEGER  IVUNBLX(*),JVUNBLX(*),IUUNBLX(*),JUUNBLX(*)
    !
    IF(QVV)THEN
       SELECT CASE(WATMODEL)
       CASE(W3SITE)  
          CALL EVVTB3(ENB8,EEL8,NWWOX,IWWOX,IWOONBLX,JWOONBLX)
       CASE(WTIP4P)
          CALL EVVTB4
       END SELECT
    ENDIF
    IF(QVU)THEN
       ENB=0.0
       EEL=0.0
       CALL EVUTB(ENB,EEL,NATOMX,IVUNBLX,JVUNBLX)
       ENB8=ENB8+ENB
       EEL8=EEL8+EEL
    ENDIF
    IF(QUU)THEN
       ENB=0.0
       EEL=0.0
       CALL EUUTB(ENB,EEL,NATOMX,IUUNBLX,JUUNBLX)
       ENB8=ENB8+ENB
       EEL8=EEL8+EEL
    ENDIF
    RETURN
  END SUBROUTINE ELOOKUP
  !
  SUBROUTINE EVVTB4
    CALL WRNDIE(-1,'<EVVTB4>','No TIP4P support yet')
    RETURN
  END SUBROUTINE EVVTB4
  !
  SUBROUTINE EVVTB3(ENB8,EEL8,NWWOX,IWWOX,IWOONBLX,JWOONBLX)
    !
    !  SolVetn-solVent lookup table non-bonded routine for three site (O,H,H)
    !  water models with all three pairwise interaction types present.
    !
    use chm_kinds
    !  use LOOKUP 
    use consta
    use dimens_fcm
    use number
    use coord
    use deriv
    use inbnd
    use stream
    IMPLICIT NONE
    real(chm_real)  ENB8,EEL8
    INTEGER NWWOX,IWWOX(*),IWOONBLX(*),JWOONBLX(*)
    !
    ! Local variables
    real(chm_real) EEL
    real(chm_real4) :: S2,ENN,ENB,C2OFXT
    real(chm_real4) :: TX(9),TY(9),TZ(9),F(9),ALPHA(9),EE(9)
    real(chm_real4) :: DTXO,DTXH1,DTXH2,DTYO,DTYH1,DTYH2,DTZO,DTZH1,DTZH2
    real(chm_real4) :: XO,XH1,XH2,YO,YH1,YH2,ZO,ZH1,ZH2
    INTEGER :: I,J,JJ,KK,II,JVECT,J1,J2,K(9),ITYP(9)=(/1,2,2,2,2,3,3,3,3/)
    INTEGER NPR,ITEMPO,ITEMP
    !
    IF(WATMODEL /= W3SITE) CALL WRNDIE(-3,'<EVVTB3>', &
         'Only 3-site water support')
    IF(PRNLEV >= 7)THEN
       WRITE(OUTU,*) '%%%%  ENTERING  EVVTB3. EEL8&ENB8=',EEL8,ENB8 
    ENDIF
    !
    C2OFXT=(CTOFNB+CTDDW)**2
    EEL=0.0
    ENB=0.0
    IF(IWWENR <= 0)THEN
       !
       ! DO NOT EVALUATE ENERGIES
       !
       !CCCC
       IF(.NOT.QWLKUPINT)THEN
          ! Atom based table lookup energy/force evaluation
          ITEMPO=0
          DO KK=1,NWWOX
             I=IWWOX(KK)
             XO=X(I)   
             XH1=X(I+1)
             XH2=X(I+2)
             YO=Y(I)   
             YH1=Y(I+1)
             YH2=Y(I+2)
             ZO=Z(I)
             ZH1=Z(I+1)
             ZH2=Z(I+2)

             DTXO=0.0    
             DTYO=0.0    
             DTZO=0.0 
             DTXH1=0.0    
             DTYH1=0.0    
             DTZH1=0.0 
             DTXH2=0.0    
             DTYH2=0.0    
             DTZH2=0.0 
             NPR=IWOONBLX(I)-ITEMPO
             DO JJ=1,NPR
                J=JWOONBLX(ITEMPO+JJ)
                TZ(1)=ZO-Z(J)
                TY(1)=YO-Y(J)
                TX(1)=XO-X(J)

                S2=TX(1)*TX(1)+TY(1)*TY(1)+TZ(1)*TZ(1)
                IF(S2 < C2OFXT)THEN
                   ! OK, so there is at least a chance that some of the interactions 
                   !     are within range
                   ! this typically saves 20-25% of the loops from being executed 
                   J1=J+1
                   J2=J+2 
                   TX(4)=XH1-X(J)
                   TX(5)=XH2-X(J)
                   TX(2)=XO-X(J1)
                   TX(6)=XH1-X(J1)
                   TX(7)=XH2-X(J1)
                   TX(3)=XO-X(J2)
                   TX(8)=XH2-X(J2)
                   TX(9)=XH1-X(J2)

                   TY(4)=YH1-Y(J)
                   TY(5)=YH2-Y(J)
                   TY(2)=YO-Y(J1)
                   TY(6)=YH1-Y(J1)
                   TY(7)=YH2-Y(J1)
                   TY(3)=YO-Y(J2)
                   TY(8)=YH2-Y(J2)
                   TY(9)=YH1-Y(J2)

                   TZ(4)=ZH1-Z(J)
                   TZ(5)=ZH2-Z(J)
                   TZ(2)=ZO-Z(J1)
                   TZ(6)=ZH1-Z(J1)
                   TZ(7)=ZH2-Z(J1)
                   TZ(3)=ZO-Z(J2)
                   TZ(8)=ZH2-Z(J2)
                   TZ(9)=ZH1-Z(J2)
                   !
                   DO II=1,9
                      K(II)=MIN(MXTB1, &
                           INT(TBSIZ*(TX(II)*TX(II)+TY(II)*TY(II)+TZ(II)*TZ(II))))
                      F(II)=FVV(K(II),ITYP(II))
                      TX(II)=TX(II)*F(II) 
                      TY(II)=TY(II)*F(II) 
                      TZ(II)=TZ(II)*F(II)
                   ENDDO
                   ! manual  unrolling of above makes no difference

                   DTXO=DTXO+TX(1)+TX(2)+TX(3)
                   DTXH1=DTXH1+TX(4)+TX(6)+TX(9)
                   DTXH2=DTXH2+TX(5)+TX(7)+TX(8)

                   DTYO=DTYO+TY(1)+TY(2)+TY(3)
                   DTYH1=DTYH1+TY(4)+TY(6)+TY(9)
                   DTYH2=DTYH2+TY(5)+TY(7)+TY(8)

                   DTZO=DTZO+TZ(1)+TZ(2)+TZ(3)
                   DTZH1=DTZH1+TZ(4)+TZ(6)+TZ(9)
                   DTZH2=DTZH2+TZ(5)+TZ(7)+TZ(8)
                   !
                   DX(J)=DX(J)-TX(1)-TX(4)-TX(5)
                   DX(J1)=DX(J1)-TX(2)-TX(6)-TX(7)
                   DX(J2)=DX(J2)-TX(3)-TX(8)-TX(9)

                   DY(J)=DY(J)-TY(1)-TY(4)-TY(5)
                   DY(J1)=DY(J1)-TY(2)-TY(6)-TY(7)
                   DY(J2)=DY(J2)-TY(3)-TY(8)-TY(9)

                   DZ(J)=DZ(J)-TZ(1)-TZ(4)-TZ(5)
                   DZ(J1)=DZ(J1)-TZ(2)-TZ(6)-TZ(7)
                   DZ(J2)=DZ(J2)-TZ(3)-TZ(8)-TZ(9)

                ENDIF
             ENDDO

             ITEMPO=IWOONBLX(I)
             DX(I)=DX(I)+DTXO
             DX(I+1)=DX(I+1)+DTXH1
             DX(I+2)=DX(I+2)+DTXH2

             DY(I)=DY(I)+DTYO
             DY(I+1)=DY(I+1)+DTYH1
             DY(I+2)=DY(I+2)+DTYH2

             DZ(I)=DZ(I)+DTZO
             DZ(I+1)=DZ(I+1)+DTZH1
             DZ(I+2)=DZ(I+2)+DTZH2
          ENDDO

          !
          ! Atom based lookup table with linear interpolation
       ELSEIF(QWLKUPINT)THEN
          IF(PRNLEV >= 7)THEN
             WRITE(OUTU,*) ' %%% EVVTB3 LIN. INTERPOLATION LOOKUP'
          ENDIF
          !
          ITEMPO=0
          DO KK=1,NWWOX
             I=IWWOX(KK)
             XO=X(I)   
             XH1=X(I+1)
             XH2=X(I+2)
             YO=Y(I)   
             YH1=Y(I+1)
             YH2=Y(I+2)
             ZO=Z(I)
             ZH1=Z(I+1)
             ZH2=Z(I+2)

             DTXO=0.0    
             DTYO=0.0    
             DTZO=0.0 
             DTXH1=0.0    
             DTYH1=0.0    
             DTZH1=0.0 
             DTXH2=0.0    
             DTYH2=0.0    
             DTZH2=0.0 

             NPR=IWOONBLX(I)-ITEMPO
             DO JJ=1,NPR
                J=JWOONBLX(ITEMPO+JJ)
                TZ(1)=ZO-Z(J)
                TY(1)=YO-Y(J)
                TX(1)=XO-X(J)
                S2=TX(1)*TX(1)+TY(1)*TY(1)+TZ(1)*TZ(1)
                IF(S2 < C2OFXT)THEN
                   ! OK, so there is at least a chance that some of the interactions
                   !      are within range
                   ! this typically saves 20-25% of the loops from being executed 
                   J1=J+1
                   J2=J+2 
                   TX(4)=XH1-X(J)
                   TX(5)=XH2-X(J)
                   TX(2)=XO-X(J1)
                   TX(6)=XH1-X(J1)
                   TX(7)=XH2-X(J1)
                   TX(3)=XO-X(J2)
                   TX(8)=XH2-X(J2)
                   TX(9)=XH1-X(J2)

                   TY(4)=YH1-Y(J)
                   TY(5)=YH2-Y(J)
                   TY(2)=YO-Y(J1)
                   TY(6)=YH1-Y(J1)
                   TY(7)=YH2-Y(J1)
                   TY(3)=YO-Y(J2)
                   TY(8)=YH2-Y(J2)
                   TY(9)=YH1-Y(J2)

                   TZ(4)=ZH1-Z(J)
                   TZ(5)=ZH2-Z(J)
                   TZ(2)=ZO-Z(J1)
                   TZ(6)=ZH1-Z(J1)
                   TZ(7)=ZH2-Z(J1)
                   TZ(3)=ZO-Z(J2)
                   TZ(8)=ZH2-Z(J2)
                   TZ(9)=ZH1-Z(J2)
                   !
                   DO II=1,9
                      S2=TBSIZ*(TX(II)*TX(II)+TY(II)*TY(II)+TZ(II)*TZ(II))
                      K(II)=MIN(MXTB1,INT(S2))
                      ALPHA(II)=S2-K(II)
                      F(II)=FVV(K(II),ITYP(II))+ &
                           ALPHA(II)*FVV(K(II),3+ITYP(II))
                      TX(II)=TX(II)*F(II) 
                      TY(II)=TY(II)*F(II) 
                      TZ(II)=TZ(II)*F(II) 
                   ENDDO
                   !
                   DTXO=DTXO+TX(1)+TX(2)+TX(3)
                   DTXH1=DTXH1+TX(4)+TX(6)+TX(9)
                   DTXH2=DTXH2+TX(5)+TX(7)+TX(8)

                   DTYO=DTYO+TY(1)+TY(2)+TY(3)
                   DTYH1=DTYH1+TY(4)+TY(6)+TY(9)
                   DTYH2=DTYH2+TY(5)+TY(7)+TY(8)

                   DTZO=DTZO+TZ(1)+TZ(2)+TZ(3)
                   DTZH1=DTZH1+TZ(4)+TZ(6)+TZ(9)
                   DTZH2=DTZH2+TZ(5)+TZ(7)+TZ(8)
                   !
                   DX(J)=DX(J)-TX(1)-TX(4)-TX(5)
                   DX(J1)=DX(J1)-TX(2)-TX(6)-TX(7)
                   DX(J2)=DX(J2)-TX(3)-TX(8)-TX(9)

                   DY(J)=DY(J)-TY(1)-TY(4)-TY(5)
                   DY(J1)=DY(J1)-TY(2)-TY(6)-TY(7)
                   DY(J2)=DY(J2)-TY(3)-TY(8)-TY(9)

                   DZ(J)=DZ(J)-TZ(1)-TZ(4)-TZ(5)
                   DZ(J1)=DZ(J1)-TZ(2)-TZ(6)-TZ(7)
                   DZ(J2)=DZ(J2)-TZ(3)-TZ(8)-TZ(9)

                ENDIF
             ENDDO

             ITEMPO=IWOONBLX(I)
             DX(I)=DX(I)+DTXO
             DX(I+1)=DX(I+1)+DTXH1
             DX(I+2)=DX(I+2)+DTXH2

             DY(I)=DY(I)+DTYO
             DY(I+1)=DY(I+1)+DTYH1
             DY(I+2)=DY(I+2)+DTYH2

             DZ(I)=DZ(I)+DTZO
             DZ(I+1)=DZ(I+1)+DTZH1
             DZ(I+2)=DZ(I+2)+DTZH2
          ENDDO
          !
       ENDIF
    ELSE
       ! Do energy evaluation
       IF(PRNLEV >= 7)THEN
          WRITE(OUTU,'(/A,I5/)') &
               ' EVVTB3 DOING ENERGY EVALUATIONS, IWWENR=',IWWENR
       ENDIF
       IF(.NOT.QWLKUPINT)THEN
          IF(PRNLEV >= 7)THEN
             WRITE(OUTU,'(/A,F10.2,I10,/)') &
                  ' EVVTB3 ATOM-ATOM WTAB: TBSIZ,MXTB=',TBSIZ,MXTB
          ENDIF
          !CCCCCCCCCCCCCCCCCCCCCCCC
          ! Atom based table lookup energy/force evaluation
          ITEMPO=0
          DO KK=1,NWWOX
             I=IWWOX(KK)
             XO=X(I)   
             XH1=X(I+1)
             XH2=X(I+2)
             YO=Y(I)   
             YH1=Y(I+1)
             YH2=Y(I+2)
             ZO=Z(I)
             ZH1=Z(I+1)
             ZH2=Z(I+2)

             DTXO=0.0    
             DTYO=0.0    
             DTZO=0.0 
             DTXH1=0.0    
             DTYH1=0.0    
             DTZH1=0.0 
             DTXH2=0.0    
             DTYH2=0.0    
             DTZH2=0.0 

             NPR=IWOONBLX(I)-ITEMPO
             DO JJ=1,NPR
                J=JWOONBLX(ITEMPO+JJ)
                TZ(1)=ZO-Z(J)
                TY(1)=YO-Y(J)
                TX(1)=XO-X(J)
                S2=TX(1)*TX(1)+TY(1)*TY(1)+TZ(1)*TZ(1)
                IF(S2 < C2OFXT)THEN
                   ! OK, so there is at least a chance that some of the interactions
                   !    are within range
                   J1=J+1
                   J2=J+2 
                   TX(4)=XH1-X(J)
                   TX(5)=XH2-X(J)
                   TX(2)=XO-X(J1)
                   TX(6)=XH1-X(J1)
                   TX(7)=XH2-X(J1)
                   TX(3)=XO-X(J2)
                   TX(8)=XH2-X(J2)
                   TX(9)=XH1-X(J2)

                   TY(4)=YH1-Y(J)
                   TY(5)=YH2-Y(J)
                   TY(2)=YO-Y(J1)
                   TY(6)=YH1-Y(J1)
                   TY(7)=YH2-Y(J1)
                   TY(3)=YO-Y(J2)
                   TY(8)=YH2-Y(J2)
                   TY(9)=YH1-Y(J2)

                   TZ(4)=ZH1-Z(J)
                   TZ(5)=ZH2-Z(J)
                   TZ(2)=ZO-Z(J1)
                   TZ(6)=ZH1-Z(J1)
                   TZ(7)=ZH2-Z(J1)
                   TZ(3)=ZO-Z(J2)
                   TZ(8)=ZH2-Z(J2)
                   TZ(9)=ZH1-Z(J2)
                   !
                   K(1)=MIN(MXTB1,INT(TBSIZ*S2))
                   DO II=2,9
                      K(II)=MIN(MXTB1, &
                           INT(TBSIZ*(TX(II)*TX(II)+TY(II)*TY(II)+TZ(II)*TZ(II))))
                   ENDDO
                   !
                   DO II=1,9
                      F(II)=FVV(K(II),ITYP(II))
                      EE(II)=EVV(K(II),ITYP(II))
                      TX(II)=TX(II)*F(II) 
                      TY(II)=TY(II)*F(II) 
                      TZ(II)=TZ(II)*F(II) 
                   ENDDO
                   EEL=EEL+SUM(EE)
                   !
                   DTXO=DTXO+TX(1)+TX(2)+TX(3)
                   DTXH1=DTXH1+TX(4)+TX(6)+TX(9)
                   DTXH2=DTXH2+TX(5)+TX(7)+TX(8)

                   DTYO=DTYO+TY(1)+TY(2)+TY(3)
                   DTYH1=DTYH1+TY(4)+TY(6)+TY(9)
                   DTYH2=DTYH2+TY(5)+TY(7)+TY(8)

                   DTZO=DTZO+TZ(1)+TZ(2)+TZ(3)
                   DTZH1=DTZH1+TZ(4)+TZ(6)+TZ(9)
                   DTZH2=DTZH2+TZ(5)+TZ(7)+TZ(8)
                   !
                   DX(J)=DX(J)-TX(1)-TX(4)-TX(5)
                   DX(J1)=DX(J1)-TX(2)-TX(6)-TX(7)
                   DX(J2)=DX(J2)-TX(3)-TX(8)-TX(9)

                   DY(J)=DY(J)-TY(1)-TY(4)-TY(5)
                   DY(J1)=DY(J1)-TY(2)-TY(6)-TY(7)
                   DY(J2)=DY(J2)-TY(3)-TY(8)-TY(9)

                   DZ(J)=DZ(J)-TZ(1)-TZ(4)-TZ(5)
                   DZ(J1)=DZ(J1)-TZ(2)-TZ(6)-TZ(7)
                   DZ(J2)=DZ(J2)-TZ(3)-TZ(8)-TZ(9)

                ENDIF
             ENDDO

             ITEMPO=IWOONBLX(I)
             DX(I)=DX(I)+DTXO
             DX(I+1)=DX(I+1)+DTXH1
             DX(I+2)=DX(I+2)+DTXH2

             DY(I)=DY(I)+DTYO
             DY(I+1)=DY(I+1)+DTYH1
             DY(I+2)=DY(I+2)+DTYH2

             DZ(I)=DZ(I)+DTZO
             DZ(I+1)=DZ(I+1)+DTZH1
             DZ(I+2)=DZ(I+2)+DTZH2
          ENDDO
          !
          ! Atom based lookup table with linear interpolation
       ELSEIF(QWLKUPINT)THEN
          IF(PRNLEV >= 7)THEN
             WRITE(OUTU,*) '%%% EVVTB3 WAT-WAT LIN. INTERPOLATION LOOKUP'
          ENDIF
          ITEMPO=0
          DO KK=1,NWWOX
             I=IWWOX(KK)
             XO=X(I)   
             XH1=X(I+1)
             XH2=X(I+2)
             YO=Y(I)   
             YH1=Y(I+1)
             YH2=Y(I+2)
             ZO=Z(I)
             ZH1=Z(I+1)
             ZH2=Z(I+2)

             DTXO=0.0    
             DTYO=0.0    
             DTZO=0.0 
             DTXH1=0.0    
             DTYH1=0.0    
             DTZH1=0.0 
             DTXH2=0.0    
             DTYH2=0.0    
             DTZH2=0.0 

             NPR=IWOONBLX(I)-ITEMPO
             DO JJ=1,NPR
                J=JWOONBLX(ITEMPO+JJ)
                TZ(1)=ZO-Z(J)
                TY(1)=YO-Y(J)
                TX(1)=XO-X(J)
                S2=TX(1)*TX(1)+TY(1)*TY(1)+TZ(1)*TZ(1)
                IF(S2 < C2OFXT)THEN
                   ! OK, so there is at least a chance that some of the interactions 
                   !     are within range
                   J1=J+1
                   J2=J+2 
                   TX(4)=XH1-X(J)
                   TX(5)=XH2-X(J)
                   TX(2)=XO-X(J1)
                   TX(6)=XH1-X(J1)
                   TX(7)=XH2-X(J1)
                   TX(3)=XO-X(J2)
                   TX(8)=XH2-X(J2)
                   TX(9)=XH1-X(J2)

                   TY(4)=YH1-Y(J)
                   TY(5)=YH2-Y(J)
                   TY(2)=YO-Y(J1)
                   TY(6)=YH1-Y(J1)
                   TY(7)=YH2-Y(J1)
                   TY(3)=YO-Y(J2)
                   TY(8)=YH2-Y(J2)
                   TY(9)=YH1-Y(J2)

                   TZ(4)=ZH1-Z(J)
                   TZ(5)=ZH2-Z(J)
                   TZ(2)=ZO-Z(J1)
                   TZ(6)=ZH1-Z(J1)
                   TZ(7)=ZH2-Z(J1)
                   TZ(3)=ZO-Z(J2)
                   TZ(8)=ZH2-Z(J2)
                   TZ(9)=ZH1-Z(J2)
                   !
                   S2=S2*TBSIZ
                   K(1)=MIN(MXTB1,INT(S2))
                   ALPHA(1)=S2-K(1)  
                   DO II=2,9
                      S2=TBSIZ*(TX(II)*TX(II)+TY(II)*TY(II)+TZ(II)*TZ(II))
                      K(II)=MIN(MXTB1,INT(S2))
                      ALPHA(II)=S2-K(II)
                   ENDDO
                   !
                   DO II=1,9
                      F(II)=FVV(K(II),ITYP(II))+ &
                           ALPHA(II)*FVV(K(II),3+ITYP(II))
                      EE(II)=EVV(K(II),ITYP(II))+ &
                           ALPHA(II)*EVV(K(II),3+ITYP(II))
                      TX(II)=TX(II)*F(II) 
                      TY(II)=TY(II)*F(II) 
                      TZ(II)=TZ(II)*F(II) 
                   ENDDO
                   EEL=EEL+SUM(EE)
                   !
                   DTXO=DTXO+TX(1)+TX(2)+TX(3)
                   DTXH1=DTXH1+TX(4)+TX(6)+TX(9)
                   DTXH2=DTXH2+TX(5)+TX(7)+TX(8)

                   DTYO=DTYO+TY(1)+TY(2)+TY(3)
                   DTYH1=DTYH1+TY(4)+TY(6)+TY(9)
                   DTYH2=DTYH2+TY(5)+TY(7)+TY(8)

                   DTZO=DTZO+TZ(1)+TZ(2)+TZ(3)
                   DTZH1=DTZH1+TZ(4)+TZ(6)+TZ(9)
                   DTZH2=DTZH2+TZ(5)+TZ(7)+TZ(8)
                   !
                   DX(J)=DX(J)-TX(1)-TX(4)-TX(5)
                   DX(J1)=DX(J1)-TX(2)-TX(6)-TX(7)
                   DX(J2)=DX(J2)-TX(3)-TX(8)-TX(9)

                   DY(J)=DY(J)-TY(1)-TY(4)-TY(5)
                   DY(J1)=DY(J1)-TY(2)-TY(6)-TY(7)
                   DY(J2)=DY(J2)-TY(3)-TY(8)-TY(9)

                   DZ(J)=DZ(J)-TZ(1)-TZ(4)-TZ(5)
                   DZ(J1)=DZ(J1)-TZ(2)-TZ(6)-TZ(7)
                   DZ(J2)=DZ(J2)-TZ(3)-TZ(8)-TZ(9)

                ENDIF
             ENDDO

             ITEMPO=IWOONBLX(I)
             DX(I)=DX(I)+DTXO
             DX(I+1)=DX(I+1)+DTXH1
             DX(I+2)=DX(I+2)+DTXH2

             DY(I)=DY(I)+DTYO
             DY(I+1)=DY(I+1)+DTYH1
             DY(I+2)=DY(I+2)+DTYH2

             DZ(I)=DZ(I)+DTZO
             DZ(I+1)=DZ(I+1)+DTZH1
             DZ(I+2)=DZ(I+2)+DTZH2
          ENDDO

       ENDIF
    ENDIF
    !
    EEL8=EEL
    ENB8=ENB
    IF(PRNLEV >= 7)THEN
       WRITE(OUTU,*) '%%% EVVTB3 EEL8,ENB8=',EEL8,ENB8
    ENDIF

    RETURN
  END SUBROUTINE EVVTB3
  SUBROUTINE EVUTB(ENB,ELE,NATOMX,INBL,JNBL)
    !
    ! Outerloop, dispatching innerloop of  solVent-solUte interaction calculation
    !
    use chm_kinds
    use dimens_fcm
    use coord
    use deriv
    use fast
    use inbnd
    use number
    use psf
    use stream
    !  use LOOKUP
    IMPLICIT NONE  
    real(chm_real) ENB,ELE
    INTEGER INBL(*),JNBL(*),NATOMX
    ! Local variables
    real(chm_real) DTX,DTY,DTZ,ENN,TABSCALE
    INTEGER ITEMP,I,NPR
    LOGICAL QVDW,QELE
    !
    TABSCALE=TBSIZ   ! real8 vs real4 issue
    ITEMP=0
    ENB=ZERO
    ELE=ZERO
    DO I=1,NATOMX
       QVDW=VDWOFFS(I) > 0
       QELE=ELEOFFS(I) > 0
       IF(.NOT. (QVDW.OR.QELE) ) CYCLE
       DTX=ZERO
       DTY=ZERO
       DTZ=ZERO
       IF(I > 1) ITEMP=INBL(I-1)
       NPR=INBL(I)-ITEMP
       IF(.NOT.QWLKUPINT)THEN
          IF(IWWENR <= 0)THEN
             ! No interpolation, no energies returned
             IF(QELE.AND.QVDW)THEN
                CALL FTBVUCLJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVULJ(1,1,VDWOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FTBVUCLJ'
             ELSEIF(QELE)THEN
                CALL FTBVUC(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FTBVUC'
             ELSE  !IF(QVDW)THEN
                CALL FTBVULJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVULJ(1,1,VDWOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FTBVULJ'
             ENDIF
          ELSE  !IWWENR
             ! No interpolation, but energies returned
             IF(QELE.AND.QVDW)THEN
                CALL ETBVUCLJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVULJ(1,1,VDWOFFS(I)),EVUC(1,ELEOFFS(I)), &
                     FVULJ(1,1,VDWOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called ETBVUCLJ'
             ELSEIF(QELE)THEN
                CALL ETBVUC(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVUC(1,ELEOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called ETBVUC'
             ELSE  !IF(QVDW)THEN
                CALL ETBVULJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVULJ(1,1,VDWOFFS(I)),FVULJ(1,1,VDWOFFS(I)), &
                     TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called ETBVULJ'
             ENDIF
             ELE=ENN+ELE
          ENDIF !IWWENR
       ELSE !QWLKUPINT
          !
          ! Linear interpolation in the same tables
          IF(IWWENR <= 0)THEN
             ! Interpolation, no energies returned
             IF(QELE.AND.QVDW)THEN
                CALL FITBVUCLJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVULJ(1,1,VDWOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FITBVUCLJ'
             ELSEIF(QELE)THEN
                CALL FITBVUC(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FITBVUC'
             ELSE  !IF(QVDW)THEN
                CALL FITBVULJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     FVULJ(1,1,VDWOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called FITBVULJ'
             ENDIF
          ELSE  !IWWENR
             ! Interpolation, energies returned
             IF(QELE.AND.QVDW)THEN
                CALL EITBVUCLJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVULJ(1,1,VDWOFFS(I)),EVUC(1,ELEOFFS(I)), &
                     FVULJ(1,1,VDWOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called EITBVUCLJ'
             ELSEIF(QELE)THEN
                CALL EITBVUC(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVUC(1,ELEOFFS(I)),FVUC(1,ELEOFFS(I)),TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called EITBVUC'
             ELSE  !IF(QVDW)THEN
                CALL EITBVULJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z, &
                     DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                     EVULJ(1,1,VDWOFFS(I)),FVULJ(1,1,VDWOFFS(I)), &
                     TABSCALE)
                IF(PRNLEV > 6) WRITE(OUTU,*) ' EVUTB called EITBVULJ'
             ENDIF
             ELE=ENN+ELE
          ENDIF !IWWENR
       ENDIF !QLKUPINT
       ! 
       DX(I)=DX(I) + DTX
       DY(I)=DY(I) + DTY
       DZ(I)=DZ(I) + DTZ
    ENDDO
    RETURN
  END SUBROUTINE EVUTB
  !
  ! Linear interpolation variants
  ! Force only routines   
  SUBROUTINE FITBVUCLJ(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVULJ,FVUC,TABSC8)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: FVULJ(MXTAB1,*),FVUC(*)
    real(chm_real) TABSC8
    ! Local variables
    INTEGER J,K,L,LJTYPE,MXTAB,LP1
    real(chm_real) TABSCALE,ALPHA,S2
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,TF1
    !
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    TABSCALE=TABSC8
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ! No check that alpha <=1 (when S2 is large), since if it is not TF1=TF(=0)
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       TF= FVULJ(L,LJTYPE)   + CG(J)*FVUC(L)
       TF1=FVULJ(LP1,LJTYPE) + CG(J)*FVUC(LP1)
       TF=TF + ALPHA * (TF1 - TF)
       !
       !         TF=FVULJ(L,LJTYPE) + CG(J)*FVUC(L) +
       !     &      ALPHA*( FVULJ(LP1,LJTYPE)-FVULJ(L,LJTYPE) +
       !     &             CG(J)*(FVUC(LP1)-FVUC(L)) )
       !
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FITBVUCLJ
  !
  SUBROUTINE FITBVULJ(NPR,JNBL,IACNB,I,X,Y,Z,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVULJ,TABSCALE)
    !
    ! Computes vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*)
    real(chm_real4) :: FVULJ(MXTAB1,*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,S2,ALPHA 
    !
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE)+ALPHA*(FVULJ(LP1,LJTYPE)-FVULJ(L,LJTYPE))
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FITBVULJ
  !
  SUBROUTINE FITBVUC(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVUC,TABSCALE)
    !
    ! Computes electrostatic interactions between solVent and solUte atoms
    ! Linear interpolation table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real) TABSCALE
    real(chm_real4) :: FVUC(*)
    ! Local variables
    INTEGER J,K,L,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,ALPHA,S2 
    !
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       TF=CG(J)*(FVUC(L)+ALPHA*(FVUC(LP1)-FVUC(L)))
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FITBVUC
  !
  ! Energy and  force routines   
  SUBROUTINE EITBVUCLJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVULJ,EVUC,FVULJ,FVUC,TABSCALE)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: EVULJ(MXTAB1,*),EVUC(*),FVULJ(MXTAB1,*),FVUC(*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,S2,ALPHA,ETMP
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE) + CG(J)*FVUC(L)
       TF=TF + ALPHA * (FVULJ(LP1,LJTYPE) + CG(J)*FVUC(LP1) -TF)
       ETMP=EVULJ(L,LJTYPE) + CG(J)*EVUC(L)
       ENB=ENB+ETMP+ ALPHA *(EVULJ(LP1,LJTYPE)+CG(J)*EVUC(LP1)-ETMP)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE EITBVUCLJ
  !
  SUBROUTINE EITBVULJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVULJ,FVULJ,TABSCALE)
    !
    ! Computes vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*)
    real(chm_real4) :: EVULJ(MXTAB1,*),FVULJ(MXTAB1,*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,ETMP,ALPHA,S2 
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    ENB=0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE)+ALPHA*(FVULJ(LP1,LJTYPE)-FVULJ(L,LJTYPE))
       ETMP=EVULJ(L,LJTYPE)
       ENB=ENB+ETMP+ALPHA*(EVULJ(LP1,LJTYPE)-ETMP)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE EITBVULJ
  !
  SUBROUTINE EITBVUC(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVUC,FVUC,TABSCALE)
    !
    ! Computes electrostatic interactions between solVent and solUte atoms
    ! Linear interpolation table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real) TABSCALE
    real(chm_real4) :: EVUC(*),FVUC(*)
    ! Local variables
    INTEGER J,K,L,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,S2,ALPHA
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       TF=CG(J)*(FVUC(L)+ALPHA*(FVUC(LP1)-FVUC(L)))
       ENB=ENB+CG(J)*(EVUC(L)+ALPHA*(EVUC(LP1)-EVUC(L)))
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE EITBVUC
  !
  !
  ! NO interpolation
  ! Force only routines   
  SUBROUTINE FTBVUCLJ(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVULJ,FVUC,TABSC8)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Straight table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: FVULJ(MXTAB1,*),FVUC(*)
    real(chm_real) TABSC8
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) TABSCALE
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF 
    !
    IF(NPR <= 0) RETURN
    TABSCALE=TABSC8
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE) + CG(J)*FVUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FTBVUCLJ
  !
  SUBROUTINE FTBVULJ(NPR,JNBL,IACNB,I,X,Y,Z,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVULJ,TABSCALE)
    !
    ! Computes vdW interactions between solVent and solUte atoms
    ! Straight table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*)
    real(chm_real4) :: FVULJ(MXTAB1,*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF 
    !
    IF(NPR <= 0) RETURN
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FTBVULJ
  !
  SUBROUTINE FTBVUC(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FVUC,TABSCALE)
    !
    ! Computes electrostatic interactions between solVent and solUte atoms
    ! Straight table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real) TABSCALE
    real(chm_real4) :: FVUC(*)
    ! Local variables
    INTEGER J,K,L
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF 
    !
    IF(NPR <= 0) RETURN
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       TF=CG(J)*FVUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FTBVUC
  !
  ! Energy and  force routines   
  SUBROUTINE ETBVUCLJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVULJ,EVUC,FVULJ,FVUC,TABSCALE)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Straight table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: EVULJ(MXTAB1,*),EVUC(*),FVULJ(MXTAB1,*),FVUC(*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE) + CG(J)*FVUC(L)
       ENB=ENB + EVULJ(L,LJTYPE) + CG(J)*EVUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE ETBVUCLJ
  !
  SUBROUTINE ETBVULJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVULJ,FVULJ,TABSCALE)
    !
    ! Computes vdW interactions between solVent and solUte atoms
    ! Straight table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*)
    real(chm_real4) :: EVULJ(MXTAB1,*),FVULJ(MXTAB1,*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF 
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    ENB=0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       TF=FVULJ(L,LJTYPE)
       ENB=ENB+EVULJ(L,LJTYPE)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE ETBVULJ
  !
  SUBROUTINE ETBVUC(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EVUC,FVUC,TABSCALE)
    !
    ! Computes electrostatic interactions between solVent and solUte atoms
    ! Straight table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real) TABSCALE
    real(chm_real4) :: EVUC(*),FVUC(*)
    ! Local variables
    INTEGER J,K,L
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF 
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       TF=CG(J)*FVUC(L)
       ENB=ENB + CG(J)*EVUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE ETBVUC
  !
  ! SolUte-solUte routines
  !
  SUBROUTINE EUUTB(ENB,ELE,NATOMX,INBL,JNBL)
    !
    ! Outerloop, dispatching innerloop of  solUte-solUte interaction calculation
    !
    use chm_kinds
    !  use LOOKUP
    use dimens_fcm
    use coord
    use deriv
    use fast
    use inbnd
    use number
    use psf
    use stream
    IMPLICIT NONE  
    real(chm_real) ENB,ELE
    INTEGER INBL(*),JNBL(*),NATOMX
    ! Local variables
    real(chm_real) DTX,DTY,DTZ,ENN,TABSCALE
    INTEGER ITEMP,I,NPR,ILJTYPE
    LOGICAL QVDW,QELE
    !
    TABSCALE=TBSIZ   ! real8 vs real4 issue
    ITEMP=0
    ENB=ZERO
    ELE=ZERO
    DO I=1,NATOMX
       DTX=ZERO
       DTY=ZERO
       DTZ=ZERO
       IF(I > 1) ITEMP=INBL(I-1)
       NPR=INBL(I)-ITEMP
       ILJTYPE=IACNB(I)
       IF(.NOT.QWLKUPINT)THEN
          IF(IWWENR <= 0)THEN
             ! No interpolation, no energies returned
             CALL FTBUUCLJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                  DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                  FUULJ(1,1,ILJTYPE),FUUC,TABSCALE)
             IF(PRNLEV > 6) WRITE(OUTU,*) ' EUUTB called FTBUUCLJ'
          ELSE  !IWWENR
             ! No interpolation, but energies returned
             CALL ETBUUCLJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                  DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                  EUULJ(1,1,ILJTYPE),EUUC, &
                  FUULJ(1,1,ILJTYPE),FUUC,TABSCALE)
             IF(PRNLEV > 6) WRITE(OUTU,*) ' EUUTB called ETBUUCLJ'
             ELE=ENN+ELE
          ENDIF !IWWENR
       ELSE !QWLKUPINT
          !
          ! Linear interpolation in the same tables
          IF(IWWENR <= 0)THEN
             ! Interpolation, no energies returned
             CALL FITBUUCLJ(NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                  DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                  FUULJ(1,1,ILJTYPE),FUUC,TABSCALE)
             IF(PRNLEV > 6) WRITE(OUTU,*) ' EUUTB called FITBUUCLJ'
          ELSE  !IWWENR
             ! Interpolation, energies returned
             CALL EITBUUCLJ(ENN,NPR,JNBL(ITEMP+1),IACNB,I,X,Y,Z,CG, &
                  DTX,DTY,DTZ,DX,DY,DZ,MXTB1, &
                  EUULJ(1,1,ILJTYPE),EUUC, &
                  FUULJ(1,1,ILJTYPE),FUUC,TABSCALE)
             IF(PRNLEV > 6) WRITE(OUTU,*) ' EUUTB called EITBUUCLJ'
             ELE=ENN+ELE
          ENDIF !IWWENR
       ENDIF !QLKUPINT
       ! 
       DX(I)=DX(I) + DTX
       DY(I)=DY(I) + DTY
       DZ(I)=DZ(I) + DTZ
    ENDDO
    RETURN
  END SUBROUTINE EUUTB
  !
  ! Linear interpolation variants
  SUBROUTINE FITBUUCLJ(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FUULJ,FUUC,TABSC8)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: FUULJ(MXTAB1,*),FUUC(*)
    real(chm_real) TABSC8
    ! Local variables
    INTEGER J,K,L,LJTYPE,MXTAB,LP1
    real(chm_real) TABSCALE,ALPHA,S2,CGI,CHG
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,TF1
    !
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    TABSCALE=TABSC8
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    CGI=CG(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       CHG=CGI*CG(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ! No check that alpha <=1 (when S2 is large), since if it is not TF1=TF(=0)
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       TF= FUULJ(L,LJTYPE)   + CHG*FUUC(L)
       TF1=FUULJ(LP1,LJTYPE) + CHG*FUUC(LP1)
       TF=TF + ALPHA * (TF1 - TF)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END  SUBROUTINE FITBUUCLJ
  !
  SUBROUTINE EITBUUCLJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EUULJ,EUUC,FUULJ,FUUC,TABSCALE)
    !
    ! Computes electrostatic and vdW interactions between solVent and solUte atoms
    ! Linear interpolation table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: EUULJ(MXTAB1,*),EUUC(*),FUULJ(MXTAB1,*),FUUC(*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE,LP1,MXTAB
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,S2,ALPHA,ETMP,CGI,CHG
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    MXTAB=MXTAB1-1
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    CGI=CG(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       S2=(TX*TX + TY*TY + TZ*TZ)*TABSCALE
       ! Allow for using L+1 within MXTAB1 in the tables
       L=MIN(MXTAB,INT(S2))
       ! Interpolation coefficient
       ALPHA=S2-L
       LP1=MIN(MXTAB1,L+1)
       LJTYPE=IACNB(J)
       CHG=CGI*CG(J)
       TF=FUULJ(L,LJTYPE) + CHG*FUUC(L)
       TF=TF + ALPHA * (FUULJ(LP1,LJTYPE) +CHG*FUUC(LP1) -TF)
       ETMP=EUULJ(L,LJTYPE) + CHG*EUUC(L)
       ENB=ENB+ETMP+ ALPHA *(EUULJ(LP1,LJTYPE)+CHG*EUUC(LP1)-ETMP)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE EITBUUCLJ
  !
  ! No interpolation variants
  !
  SUBROUTINE FTBUUCLJ(NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,FUULJ,FUUC,TABSC8)
    !
    ! Computes electrostatic and vdW interactions between solUte and solUte atoms
    ! Straight table lookup, no energy tables used
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) X(*),Y(*),Z(*),DTX,DTY,DTZ,DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: FUULJ(MXTAB1,*),FUUC(*)
    real(chm_real) TABSC8
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) TABSCALE,CGI
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,TF1
    !
    IF(NPR <= 0) RETURN
    TABSCALE=TABSC8
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    CGI=CG(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       TF= FUULJ(L,LJTYPE) + CGI*CG(J)*FUUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    RETURN
  END SUBROUTINE FTBUUCLJ
  !
  SUBROUTINE ETBUUCLJ(ENB8,NPR,JNBL,IACNB,I,X,Y,Z,CG,DTX,DTY,DTZ, &
       DX,DY,DZ,MXTAB1,EUULJ,EUUC,FUULJ,FUUC,TABSCALE)
    !
    ! Computes electrostatic and vdW interactions between solUte and solUte atoms
    ! No interpolation table lookup energy and force
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER NPR,JNBL(*),IACNB(*),MXTAB1,I
    real(chm_real) ENB8,X(*),Y(*),Z(*),DTX,DTY,DTZ, &
         DX(*),DY(*),DZ(*),CG(*)
    real(chm_real4) :: EUULJ(MXTAB1,*),EUUC(*),FUULJ(MXTAB1,*),FUUC(*)
    real(chm_real) TABSCALE
    ! Local variables
    INTEGER J,K,L,LJTYPE
    real(chm_real) XI,YI,ZI,TX,TY,TZ,TF,CGI,CHG
    REAL ENB
    !
    ENB8=0.0
    IF(NPR <= 0) RETURN
    ENB=0.0
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)
    CGI=CG(I)
    DO K=1,NPR
       J=JNBL(K)
       TX=XI-X(J)
       TY=YI-Y(J)
       TZ=ZI-Z(J)
       L=MIN(INT((TX*TX + TY*TY + TZ*TZ)*TABSCALE),MXTAB1)
       LJTYPE=IACNB(J)
       CHG=CGI*CG(J)
       TF=FUULJ(L,LJTYPE) + CHG*FUUC(L)
       ENB=ENB+EUULJ(L,LJTYPE) + CHG*EUUC(L)
       TX=TF*TX
       TY=TF*TY
       TZ=TF*TZ
       DTX=DTX + TX
       DTY=DTY + TY
       DTZ=DTZ + TZ
       DX(J)=DX(J) - TX
       DY(J)=DY(J) - TY
       DZ(J)=DZ(J) - TZ
    ENDDO
    ENB8=ENB
    RETURN
  END SUBROUTINE ETBUUCLJ
  !
  !   LIST ROUTINES
  !
  SUBROUTINE WWSPLTNB(QCOMPLET,QIMG,IFRSTA,NATOMX,INBL,JNB, &
       MXOOX,IWOONBX,JWOONBX, &
#if KEY_IMCUBES==1
       LBYCBIM,                   & 
#endif
       NBR,NBOO,NBVU,NBUU)
    ! 
    ! PBUCBES NOT SUPPORTED HERE
    !
    !  Split the non-bond list INBL(IFRSTA:NATOMX) by extracting separate
    !  OO based lists for those waters that are to be handled by
    !  the water-water routine. In similar fashion pick the solVent-solUte part. 
    !  This means that only a possibly small fraction of the original list
    !  ends up being used which could be a waste of memory. 
    !  INBL is processed in two passes. First the other lists are extracted,
    !  and only if this is successful (no dimension overruns) we remove
    !  things from INBL/JNB. This allows us to reuse INBL/JNB in the next
    !  try with resized arrays.  
    !
    use chm_kinds
    !  use LOOKUP
    use dimens_fcm
    use stream
    IMPLICIT NONE
    INTEGER IFRSTA,MXOOX,NATOMX,INBL(*),JNB(*)
    INTEGER IWOONBX(*),JWOONBX(MXOOX)
    INTEGER NBR,NBOO,NBVU,NBUU
    LOGICAL QCOMPLET,QIMG
#if KEY_IMCUBES==1
    INTEGER NBR1  
    LOGICAL LBYCBIM
#endif 
    !
    INTEGER I,J,JJ,NNNB,MNB,ITEMP,ITEMP1,JEND,IH1,IH2,JTMP1
    INTEGER JO,JH1,JH2,JOEND,JH1END,JH2END,JA
    INTEGER JCURR,JMO,JMH1,JMH2,NV,JTMP,MBVU,K,L,SOLVENT,SOLUTE
    INTEGER, PARAMETER :: IVAL=1,IDX=2
    LOGICAL QIO,DONE
    !
    !      INTEGER DBGNWOO,IDBG1
    !         DBGNWOO=0
    !         DO K=IFRSTA,NATOMX
    !            IF(IWWFLG(K) > 0) DBGNWOO=DBGNWOO+1
    !         ENDDO
    !
    QCOMPLET=.TRUE.
    NBOO=0 
    ITEMP=0 
    JEND=MAXAIM+1
    DO I=IFRSTA,NATOMX
       IF(IWWFLG(I) == I)THEN
          ! OK, this is the oxygen, now get index for the two "hydrogens"  as well
          IH1=I+1
          IH2=I+2
          ! Validity check should not be necessary 
          IF(I == 1)THEN                                
             JO=1
          ELSE
             JO=INBL(I-1)+1
          ENDIF
          JH1=INBL(IH1-1)+1
          JH2=INBL(IH2-1)+1
          JOEND=INBL(I)
          JH1END=INBL(IH1)
          JH2END=INBL(IH2)
#if KEY_IMCUBES==1
          IF(LBYCBIM)THEN
             JO=INBL(I+NATOMX)+1     
             JH1=INBL(IH1+NATOMX)+1     
             JH2=INBL(IH2+NATOMX)+1     
          ENDIF
#endif 
          JCURR=0
          DONE=.FALSE.
          DO WHILE(.NOT.DONE)
             !
             ! For each I-atom find the next neighbor with higher index than most
             ! recently added (JCURR), then pick the smallest of these to add to list
             !
             JMO=NEXTJ(JCURR,JO,JNB,IWWFLG,JOEND,JEND)
             JMH1=NEXTJ(JCURR,JH1,JNB,IWWFLG,JH1END,JEND) 
             JMH2=NEXTJ(JCURR,JH2,JNB,IWWFLG,JH2END,JEND)
             JCURR=MIN(JMO,JMH1,JMH2)
             IF(JCURR  ==  JEND)THEN
                DONE=.TRUE.
             ELSE
                NBOO=NBOO+1
                IF(NBOO > MXOOX)THEN
                   QCOMPLET=.FALSE.
                ELSE
                   JWOONBX(NBOO)=JCURR 
                ENDIF
             ENDIF
          ENDDO
          IWOONBX(I)=NBOO     
       ENDIF
    ENDDO
    ! 
    IF(.NOT.QCOMPLET) RETURN
    !
    ! All well so far, now trim down JNB
    ! NBR returns total number nonbond interactions left on regular list 
    !   (possibly only the 1-4s)
    !
    IF(QVU)THEN
#if KEY_IMCUBES==1
       IF(LBYCBIM)THEN
          IF(QIMG)THEN
             MNB=NNBIBYCB
          ELSE
             MNB=NNBBYCB
          ENDIF
       ELSE
#endif 
          MNB=INBL(NATOMX) 
#if KEY_IMCUBES==1
       ENDIF                
#endif
       IF(.NOT.ALLOCATED(LNKLST))THEN
          MXVU=MAX(NBOO,MNB-5*NBOO)
          ALLOCATE(LNKLST(2,MXVU))
          IF(PRNLEV > 5)THEN
             IF(QIMG)THEN
                WRITE(OUTU,100) '   Image',MXVU
             ELSE
                WRITE(OUTU,100) ' Primary',MXVU
             ENDIF
100          FORMAT(' WWSPLTNB>',A,'. Allocated space for', &
                  I10,' pairs on linked-list')
          ENDIF
       ENDIF
       JTMP=MIN(MAXAIM,INT(1.1*NATOMX))
       IF(.NOT.ALLOCATED(FRSTIDX))THEN
          ALLOCATE(FRSTIDX(JTMP))
          IF(PRNLEV > 5) WRITE(OUTU,110) JTMP,'FRSTIDX'
110       FORMAT(' Allocated space for',I10,' atoms on ',A)
       ELSEIF(SIZE(FRSTIDX) < NATOMX)THEN
          DEALLOCATE(FRSTIDX) 
          ALLOCATE(FRSTIDX(JTMP))
          IF(PRNLEV > 5) WRITE(OUTU,110) JTMP,'FRSTIDX/resize'
       ENDIF
       IF(.NOT.ALLOCATED(LASTIDX))THEN
          ALLOCATE(LASTIDX(JTMP)) 
          IF(PRNLEV > 5) WRITE(OUTU,110) JTMP,'LASTIDX'
       ELSEIF(SIZE(LASTIDX) < NATOMX)THEN 
          DEALLOCATE(LASTIDX) 
          ALLOCATE(LASTIDX(JTMP))
          IF(PRNLEV > 5) WRITE(OUTU,110) JTMP,'LASTIDX/resize'
       ENDIF
       FRSTIDX=-1
       LASTIDX=-1
       LNKLST(IDX,1:MXVU)=-1
    ENDIF !QVU
    !
    NBR=0
    ITEMP=0      
    NBVU=0
    NV=0
    DO I=IFRSTA,NATOMX
       QIO= (IWWFLG(I) > 0)
       ! PBCUBE also needs similar treatment as IMCUBES
#if KEY_IMCUBES==1
       IF(LBYCBIM) ITEMP=INBL(NATOMX+I)              
#endif
#if KEY_IMCUBES==1
       NBR1=0                                        
#endif
       IF(QIO) NV=NV+1 
       NNNB=INBL(I)-ITEMP
       DO JJ=1,NNNB         
          J=JNB(ITEMP+JJ)
          JA=ABS(J)
          IF(.NOT. (QIO.AND.IWWFLG(JA) > 0) )THEN
             !
             ! Not solvent-solvent, but is it a solVent-solUte interaction?
             ! NB The list need not be ordered with eg all solvents first, so either 
             ! I or J can be a solvent molecule (as long as not the other is one too,
             !  but that we just checked. Our new VU list should be arranged so that 
             ! for each solVent atom its solUte non-bonded interactions are listed.
             ! First make a linked list, then unpack it.
             !
             IF(QVU .AND. (QIO.OR.IWWFLG(JA) > 0))THEN
                NBVU=NBVU+1
                IF(NBVU > MXVU)THEN
                   ! Use a small margin to reduce number of deallocation/allocation events
                   JTMP=1.1*MXVU
                   IF(.NOT.ALLOCATED(VUTMP)) THEN
                      ALLOCATE(VUTMP(2,JTMP))
                      IF(PRNLEV > 5) WRITE(OUTU,120) 'Allocated',JTMP
120                   FORMAT(1X,A,' space for',I10,' VUTMP entries')
                   ELSEIF(SIZE(VUTMP) < 2*JTMP) THEN
                      DEALLOCATE(VUTMP)
                      ALLOCATE(VUTMP(2,JTMP))
                      IF(PRNLEV > 5)WRITE(OUTU,120)'Reallocated',JTMP
                   ENDIF
                   VUTMP(1,1:MXVU)=LNKLST(1,1:MXVU)
                   VUTMP(2,1:MXVU)=LNKLST(2,1:MXVU)
                   JTMP=MXVU
                   JTMP1=MNB*FLOAT(NWW)/FLOAT(MAX(NV,1))
                   JTMP1=MAX(JTMP1,INT(1.2*MXVU))
                   MXVU=MIN(MNB,JTMP1)
                   !
                   ! The actual resize of the linked-list, and copy back old data
                   !
                   DEALLOCATE(LNKLST)
                   ALLOCATE(LNKLST(2,MXVU))
                   LNKLST(1,1:JTMP)=VUTMP(1,1:JTMP)
                   LNKLST(2,1:JTMP)=VUTMP(2,1:JTMP)
                   LNKLST(IDX,JTMP+1:MXVU)=-1
                   IF(QIMG)THEN
                      IF(PRNLEV > 5) WRITE(OUTU,100) &
                           'Image/resize',MXVU
                   ELSE
                      IF(PRNLEV > 5) WRITE(OUTU,100) &
                           'Primary/resize',MXVU
                   ENDIF
                ENDIF !NBVU
                !                                        
                IF(QIO)THEN
                   SOLVENT=I
                   SOLUTE=JA
                ELSE
                   SOLVENT=JA
                   SOLUTE=I
                ENDIF
                !
                L=LASTIDX(SOLVENT)
                IF(L <= 0) THEN
                   FRSTIDX(SOLVENT)=NBVU
                ELSE
                   LNKLST(IDX,L)=NBVU
                ENDIF
                LASTIDX(SOLVENT)=NBVU
                LNKLST(IVAL,NBVU)=SOLUTE
             ELSE
                !
                ! Put it back onto  regular list,  which at this stage cannot overflow
                NBR=NBR+1
#if KEY_IMCUBES==1
                IF(LBYCBIM)THEN  
                   NBR1=NBR1+1  
                   JNB(ITEMP+NBR1)=J
                ELSE            
#endif 
                   JNB(NBR)=J
#if KEY_IMCUBES==1
                ENDIF                        
#endif
             ENDIF  !QVU
          ENDIF  !.NOT.QIO .AND. 
       ENDDO
       ! put back new value; for LBYCBIM start index in INBL(I+NATOMX) has not changed
#if KEY_IMCUBES==1
       IF(LBYCBIM)THEN
          INBL(I)=ITEMP+NBR1
       ELSE
#endif 
          ITEMP=INBL(I)
          INBL(I)=NBR
#if KEY_IMCUBES==1
       ENDIF                                  
#endif
    ENDDO
    !
    NBUU=0
    IF(.NOT.(QVU.OR.QUU)) RETURN
    !
    IF(QVU)THEN
       ! Unpack the linked list into the appropriate VU-nonbonded list
       IF(QIMG)THEN
          IF(.NOT. ALLOCATED(JVUNBLI))THEN
             MXJVUI=1.1*NBVU
             ALLOCATE(JVUNBLI(MXJVUI))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJVUI,' image'
130          FORMAT(' Allocated space for',I10,A,' VU pairs')  
          ELSEIF(SIZE(JVUNBLI) < NBVU)THEN
             MXJVUI=1.1*NBVU
             DEALLOCATE(JVUNBLI)
             ALLOCATE(JVUNBLI(MXJVUI))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJVUI,' image/resize'
          ENDIF
          IVUNBLI=0
          MBVU=0
          DO I=IFRSTA,NATOMX
             IF(IWWFLG(I) > 0)THEN
                K=FRSTIDX(I)
                DO WHILE(K > 0)
                   J=LNKLST(IVAL,K)
                   MBVU=MBVU+1
                   JVUNBLI(MBVU)=J
                   K=LNKLST(IDX,K)
                ENDDO
             ENDIF
             IVUNBLI(I)=MBVU
          ENDDO
       ELSE
          IF(.NOT. ALLOCATED(JVUNBL))THEN
             MXJVU=1.1*NBVU
             ALLOCATE(JVUNBL(MXJVU))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJVU,' primary'
          ELSEIF(SIZE(JVUNBL) < NBVU)THEN
             MXJVU=1.1*NBVU
             DEALLOCATE(JVUNBL)
             ALLOCATE(JVUNBL(MXJVU))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJVU,' primary/resize'
          ENDIF
          IVUNBL=0
          MBVU=0
          DO I=IFRSTA,NATOMX
             IF(IWWFLG(I) > 0)THEN
                K=FRSTIDX(I)
                DO WHILE(K > 0)
                   J=LNKLST(IVAL,K)
                   MBVU=MBVU+1
                   JVUNBL(MBVU)=J
                   K=LNKLST(IDX,K)
                ENDDO
             ENDIF
             IVUNBL(I)=MBVU
          ENDDO
       ENDIF
       !
       IF(MBVU /= NBVU)THEN
          IF(PRNLEV >= 5)THEN
             !
             !         IDBG1=0
             !         DO K=IFRSTA,NATOMX
             !            IF(IWWFLG(K) > 0) IDBG1=IDBG1+1
             !         ENDDO
             !         WRITE(OUTU,*) ' IWWFLG non-zero entries:',IDBG1,DBGNWOO
             WRITE(OUTU,*) ' IFRSTA,NATOMX,QIMG:',IFRSTA,NATOMX,QIMG
             DO K=IFRSTA,NATOMX
                IF(FRSTIDX(K) == -1)THEN
                   IF(LASTIDX(K) /= -1)WRITE(OUTU,*)'K,LASTIDX:',K,LASTIDX(K)
                ELSE
                   IF(LASTIDX(K) == -1)THEN
                      WRITE(OUTU,*) &
                           'K,FRSTIDX,LASTIDX:',K,FRSTIDX(K),LASTIDX(K)
                   ELSE 
                      IF(LNKLST(IDX,LASTIDX(K)) /= -1)  &
                           WRITE(OUTU,*) 'K,LNKLST(IDX,K):',K,LNKLST(IDX,K)
                   ENDIF
                ENDIF
             ENDDO
             !         
             IF(QIMG)THEN
                WRITE(OUTU,150) 'image',MBVU,NBVU,MXJVUI,MXVU
150             FORMAT(' Error processing ',A,' linked list'/ &
                     '     MBVU:',I10/ &
                     '     NBVU:',I10/ &
                     ' MXJVU(I):',I10/ &
                     '     MXVU:',I10)
             ELSE
                WRITE(OUTU,150) 'primary',MBVU,NBVU,MXJVU,MXVU
             ENDIF
          ENDIF
          CALL WRNDIE(-4,'<WWSPLTNB>','Internal linked list error')
       ENDIF
    ENDIF !QVU
    !
    IF(QUU)THEN
       IF(QIMG)THEN
          IF(.NOT. ALLOCATED(JUUNBLI))THEN
             MXJUUI=1.1*NBR
             ALLOCATE(JUUNBLI(MXJUUI))
             IF(PRNLEV > 5) WRITE(OUTU,140) MXJUUI,' image'
140          FORMAT(' Allocated space for',I10,A,' UU pairs')  
          ELSEIF(SIZE(JUUNBLI) < NBR)THEN
             MXJUUI=1.1*NBR
             DEALLOCATE(JUUNBLI)
             ALLOCATE(JUUNBLI(MXJUUI))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJUUI,' image/resize'
          ENDIF
          IUUNBLI=0
          ITEMP=0
          NBR=0
          DO I=IFRSTA,NATOMX
#if KEY_IMCUBES==1
             IF(LBYCBIM) ITEMP=INBL(NATOMX+I)              
#endif
#if KEY_IMCUBES==1
             NBR1=0                                        
#endif
             NNNB=INBL(I)-ITEMP
             DO JJ=1,NNNB
                J=JNB(ITEMP+JJ)
                IF(J > 0)THEN
                   NBUU=NBUU+1
                   IF(NBUU > MXJUUI)THEN
                      ! resize
                      JTMP=1.1*MXJUUI
                      IF(.NOT.ALLOCATED(VUTMP)) THEN
                         ALLOCATE(VUTMP(1,JTMP))
                         IF(PRNLEV > 5) WRITE(OUTU,160)  &
                              'Allocated image',JTMP
160                      FORMAT(1X,A,' space for',I10,' UUTMP entries')
                      ELSEIF(SIZE(VUTMP) < MXJUUI) THEN
                         DEALLOCATE(VUTMP)
                         ALLOCATE(VUTMP(1,JTMP))
                         IF(PRNLEV > 5)WRITE(OUTU,160) &
                              'Reallocated image',JTMP
                      ENDIF
                      VUTMP(1,1:MXJUUI)=JUUNBLI(1:MXJUUI)
                      DEALLOCATE(JUUNBLI)
                      MXJUUI=JTMP
                      ALLOCATE(JUUNBLI(MXJUUI))
                   ENDIF
                   JUUNBLI(NBUU)=J
                ELSE
                   NBR=NBR+1
#if KEY_IMCUBES==1
                   IF(LBYCBIM)THEN
                      NBR1=NBR1+1
                      JNB(ITEMP+NBR1)=J
                   ELSE
#endif 
                      JNB(NBR)=J
#if KEY_IMCUBES==1
                   ENDIF                                   
#endif
                ENDIF
             ENDDO
             IUUNBLI(I)=NBUU
#if KEY_IMCUBES==1
             IF(LBYCBIM)THEN
                INBL(I)=ITEMP+NBR1
             ELSE
#endif 
                ITEMP=INBL(I)
                INBL(I)=NBR
#if KEY_IMCUBES==1
             ENDIF                                          
#endif
          ENDDO
       ELSE  !QIMG
          IF(.NOT. ALLOCATED(JUUNBL))THEN
             MXJUU=1.1*NBR
             ALLOCATE(JUUNBL(MXJUU))
             IF(PRNLEV > 5) WRITE(OUTU,140) MXJUU,' primary'
          ELSEIF(SIZE(JUUNBL) < NBR)THEN
             MXJUU=1.1*NBR
             DEALLOCATE(JUUNBL)
             ALLOCATE(JUUNBL(MXJUU))
             IF(PRNLEV > 5) WRITE(OUTU,130) MXJUU,' primary/resize'
          ENDIF
          IUUNBL=0
          ITEMP=0
          NBR=0
          DO I=IFRSTA,NATOMX
#if KEY_IMCUBES==1
             IF(LBYCBIM) ITEMP=INBL(NATOMX+I)              
#endif
#if KEY_IMCUBES==1
             NBR1=0                                        
#endif
             NNNB=INBL(I)-ITEMP
             DO JJ=1,NNNB
                J=JNB(ITEMP+JJ)
                IF(J > 0)THEN
                   NBUU=NBUU+1
                   IF(NBUU > MXJUU)THEN
                      ! resize
                      JTMP=1.1*MXJUU
                      IF(.NOT.ALLOCATED(VUTMP)) THEN
                         ALLOCATE(VUTMP(1,JTMP))
                         IF(PRNLEV > 5) WRITE(OUTU,150) &
                              'Allocated primary',JTMP
                      ELSEIF(SIZE(VUTMP) < MXJUU) THEN
                         DEALLOCATE(VUTMP)
                         ALLOCATE(VUTMP(1,JTMP))
                         IF(PRNLEV > 5)WRITE(OUTU,150) &
                              'Reallocated primary',JTMP
                      ENDIF
                      VUTMP(1,1:MXJUU)=JUUNBL(1:MXJUU)
                      DEALLOCATE(JUUNBL)
                      MXJUU=JTMP
                      ALLOCATE(JUUNBLI(MXJUU))
                   ENDIF
                   JUUNBL(NBUU)=J
                ELSE
                   NBR=NBR+1
#if KEY_IMCUBES==1
                   IF(LBYCBIM)THEN
                      NBR1=NBR1+1
                      JNB(ITEMP+NBR1)=J
                   ELSE
#endif 
                      JNB(NBR)=J
#if KEY_IMCUBES==1
                   ENDIF                                   
#endif
                ENDIF
             ENDDO
             IUUNBL(I)=NBUU
#if KEY_IMCUBES==1
             IF(LBYCBIM)THEN
                INBL(I)=ITEMP+NBR1
             ELSE
#endif 
                ITEMP=INBL(I)
                INBL(I)=NBR
#if KEY_IMCUBES==1
             ENDIF                                          
#endif
          ENDDO
       ENDIF !QIMG
    ENDIF !QUU

    !
    RETURN
  END SUBROUTINE WWSPLTNB
  !
  INTEGER FUNCTION NEXTJ(JCURR,J,JNB,IWWFLG,JMAX,JEND)
    ! 
    ! Return index of first atom in next water molecule in JNB if J points to 
    ! a water molecule <= JCURR.
    ! If list is exhausted returns JEND
    !
    use chm_kinds
    IMPLICIT NONE
    INTEGER  JCURR,J,JNB(*),JMAX,JEND,IWWFLG(*)
    INTEGER JMOL
    !
    DO WHILE(J <= JMAX)
       JMOL=IWWFLG(JNB(J))
       IF(JMOL  >  0)THEN
          NEXTJ=JMOL
          IF(JMOL > JCURR) RETURN
       ENDIF
       J=J+1
    ENDDO
    NEXTJ=JEND
    RETURN
  END FUNCTION NEXTJ
  !
  !   SETUP ROUTINES
  !
  SUBROUTINE WWSETUP(COMLYN,COMLEN)
    !
    ! Parse commandline for ww-routines and set everything up
    !
    use chm_kinds
    !  use LOOKUP
    use dimens_fcm
    use select
    use coord
    use number
    use image
    use inbnd
    use psf
    use param
    use fast
#if KEY_PERT == 1
    use pert,only:qpert,pertip
#endif
#if KEY_PARALLEL==1
    use parallel     
#endif
    use stream
    use string
    use chutil,only: hydrog
    IMPLICIT NONE 
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN  
    !
    real(chm_real) TBS8
    INTEGER I,I1,J,NUUTAB

    !
    ! Always clean up first
    IF(ALLOCATED(EVV)) DEALLOCATE(EVV)
    IF(ALLOCATED(FVV)) DEALLOCATE(FVV)
    IF(ALLOCATED(IWWO)) DEALLOCATE(IWWO)
    IF(ALLOCATED(IWWFLG)) DEALLOCATE(IWWFLG)
    IF(ALLOCATED(ELEOFFS)) DEALLOCATE(ELEOFFS)
    IF(ALLOCATED(VDWOFFS)) DEALLOCATE(VDWOFFS)
    IF(ALLOCATED(EVULJ)) DEALLOCATE(EVULJ)
    IF(ALLOCATED(FVULJ)) DEALLOCATE(FVULJ)
    IF(ALLOCATED(EVUC)) DEALLOCATE(EVUC)
    IF(ALLOCATED(FVUC)) DEALLOCATE(FVUC)
    IF(ALLOCATED(EUULJ)) DEALLOCATE(EUULJ)
    IF(ALLOCATED(FUULJ)) DEALLOCATE(FUULJ)
    IF(ALLOCATED(EUUC)) DEALLOCATE(EUUC)
    IF(ALLOCATED(FUUC)) DEALLOCATE(FUUC)

    IF(ALLOCATED(IWOONBL)) DEALLOCATE(IWOONBL)
    IF(ALLOCATED(JWOONBL)) DEALLOCATE(JWOONBL)
    IF(ALLOCATED(IWOONBLI)) DEALLOCATE(IWOONBLI)
    IF(ALLOCATED(JWOONBLI)) DEALLOCATE(JWOONBLI)
    IF(ALLOCATED(IVUNBL)) DEALLOCATE(IVUNBL)
    IF(ALLOCATED(JVUNBL)) DEALLOCATE(JVUNBL)
    IF(ALLOCATED(IVUNBLI)) DEALLOCATE(IVUNBLI)
    IF(ALLOCATED(JVUNBLI)) DEALLOCATE(JVUNBLI)

    IF(ALLOCATED(IUUNBL)) DEALLOCATE(IUUNBL)
    IF(ALLOCATED(JUUNBL)) DEALLOCATE(JUUNBL)
    IF(ALLOCATED(IUUNBLI)) DEALLOCATE(IUUNBLI)
    IF(ALLOCATED(JUUNBLI)) DEALLOCATE(JUUNBLI)


    IF(ALLOCATED(FRSTIDX)) DEALLOCATE(FRSTIDX)
    IF(ALLOCATED(LASTIDX)) DEALLOCATE(LASTIDX)
    IF(ALLOCATED(LNKLST)) DEALLOCATE(LNKLST)
    IF(ALLOCATED(VUTMP)) DEALLOCATE(VUTMP)

    MXVU=-1  
    MXJVU=-1
    MXJVUI=-1
    MXJUU=-1
    MXJUUI=-1
    MXJWWOO=-1
    MXJWWOOI=-1

    QWLKUPINT=.FALSE.
    QLOOKUP=.FALSE.
    QVU=.FALSE.
    QVV=.FALSE.
    QUU=.FALSE.

    EWWEEL=0.0
    EWWENB=0.0
    EWWEELI=0.0
    EWWENBI=0.0
    IF(INDXA(COMLYN,COMLEN,'RELEASE') > 0)RETURN
    IF(INDXA(COMLYN,COMLEN,'RESE') > 0)THEN
       IF(PRNLEV >= 2)  WRITE(OUTU,'(/A/)') &
            ' WWSETUP> RESET - standard routines will be used'
       RETURN
    ENDIF
    !
    ! Selection needs to be parsed before the rest

    IF(.NOT.ALLOCATED(IWWFLG)) ALLOCATE(IWWFLG(MAXAIM))
    ! Allow lookup for just UU. No selection here has to mean that NO atoms are selected, but
    ! SELCTA has the opposite interpretation
    IF(INDX(COMLYN,COMLEN,'SELE',4) > 0)THEN
       CALL SELCTA(COMLYN,COMLEN,IWWFLG,X,Y,Z,WMAIN,.TRUE.)
       NWW=NSELCT(NATOM,IWWFLG)
    ELSE
       NWW=0
       IWWFLG=0
    ENDIF
    NWWO=NWW
    QVU= (NWW < NATOM) .AND. (NWW > 0)
    QUU= (NWW < NATOM)
    QVV= (NWW > 0)
    ! Force turning off of specific tables
    QVU= QVU.AND. (INDXA(COMLYN,COMLEN,'NOVU') == 0)
    QVV= QVV.AND. (INDXA(COMLYN,COMLEN,'NOVV') == 0)
    !      QUU= QUU.AND. (INDXA(COMLYN,COMLEN,'UU') > 0)
    QUU= QUU.AND. (INDXA(COMLYN,COMLEN,'NOUU') == 0)
    !
    !
    ! We can at present only handle the combinations listed due to problems with standard list
    ! which would be used by standard routine AND elookup for solute-solute in the other cases:
    !    UU  UV   VV
    !    F  any   any
    !    T   T     T
    !    T   F     F
    ! This logic may need checking..
    !

    IF(QUU .AND. (QVU.NEQV.QVV))THEN
       QUU=.FALSE.
       IF(PRNLEV >= 2) WRITE(OUTU,*) &
            ' Can not use UU tables when VU and VV are treated differently'
    ENDIF
    !
    QLOOKUP=QVU.OR.QUU.OR.QVV
    IF(.NOT. QLOOKUP )THEN
       IF(ALLOCATED(IWWFLG)) DEALLOCATE(IWWFLG)
       IF(PRNLEV >= 2) WRITE(OUTU,*)  &
            ' All lookup tables disabled. Standard routines will be used.'
       RETURN
    ELSE
       ! Check that we are not conflicting with PERT
#if KEY_PERT == 1
      IF(QPERT)THEN
        IF(QUU .OR. QVU) CALL WRNDIE(-4, &
           '<WWSETUP>','LOOKUP with PERT MUST use NOUU and NOVU')
        IF(ANY(IWWFLG(1:NATOM) == 1 .AND. (IWWFLG(1:NATOM)==PERTIP(1:NATOM))))THEN
           CALL WRNDIE(-4, &
           '<WWSETUP>','LOOKUP and PERT selections MUST be disjoint')
        ENDIF


     ENDIF
#endif
    ENDIF

    ! 
    ! We are in business
    !
    IF(.NOT.ALLOCATED(IWWO)) ALLOCATE(IWWO(MAXAIM))
    !
    ! Offsets into tables
    IF(QVU)THEN
       IF(ALLOCATED(ELEOFFS)) DEALLOCATE(ELEOFFS)
       IF(ALLOCATED(VDWOFFS)) DEALLOCATE(VDWOFFS)
       ALLOCATE(ELEOFFS(MAXAIM))
       ALLOCATE(VDWOFFS(MAXAIM))
    ENDIF
    !
    ! Give same index to  all atoms in a given selected solvent molecule
    CALL MKWWGRP(NATOM,IWWFLG)
    ! Default to using interpolation
    QWLKUPINT=.TRUE.
    IF(INDXA(COMLYN,COMLEN,'INT')  > 0) QWLKUPINT=.TRUE.
    IF(INDXA(COMLYN,COMLEN,'NOIN') > 0) QWLKUPINT=.FALSE. 
    IF(QWLKUPINT)THEN
       ! No pre-roundoff correction for the interpolation table
       RNDING=GTRMF(COMLYN,COMLEN,'ROUN',ZERO)
    ELSE
       ! but for the direct table lookup we need it
       RNDING=GTRMF(COMLYN,COMLEN,'ROUN',HALF)
    ENDIF
    !
    ! IWWENR: 0 DO NOT EVALUATE ENERGY (not used in current code)
    !         2 ALWAYS DO
    !         +1 ONLY AT UPDATES AND NOW  (default)
    !         -1 ONLY AT UPDATES AND WE ARE INBETWEEN NOW
    ! 
    IWWENR=-1
    IF(INDXA(COMLYN,COMLEN,'NOEN') > 0) IWWENR=-1 
    IF(INDXA(COMLYN,COMLEN,'ENER') > 0) IWWENR=2
    
#if KEY_PERT == 1
    IF(QPERT .AND. IWWENR /= 2) THEN
       CALL WRNDIE(1, &
       '<WWSETUP>','LOOKUP with PERT requires "ENERGY" flag; this has been set')
       IWWENR=2
    ENDIF
#endif /* KEY_PERT */
    
    ! Set default value
    TBS8=20.0
    TBSIZ=GTRMF(COMLYN,COMLEN,'TABI',TBS8)
    !
    IF(QVV.OR.QVU)THEN  
       ! Three or four site model? For now three site is supported. Also need more than one 
       ! water molecule for water-water intarctions to make sense...
       WATMODEL=W3SITE
       SELECT CASE(WATMODEL)
       CASE (W3SITE)
          IF(NWWO <= 3 .OR. MOD(NWWO,3) /= 0)THEN
             IF(PRNLEV >= 2) WRITE(OUTU,'(/A,I9,/)')  &
                  '   NWWRES=', NWWO
             CALL WRNDIE(1,'<WWSETUP>', &
                  'Incorrect # (or < 3) of atoms selected. Using std routine.')
             QLOOKUP=.FALSE.
             QVV=.FALSE.
             QVU=.FALSE.
             QUU=.FALSE.
             NWWO=0
             RETURN
          ELSE
             !
             !
             !   The third atom is a hydrogen for three site and four site (TIP4P) models.
             DO J=1,NATOM
                IF(IWWFLG(J) /= 0)THEN
                   I=J+2
                   EXIT
                ENDIF
             ENDDO
             IF(.NOT.HYDROG(I)) CALL WRNDIE(-3,'<WWSETUP>', &
                  '3rd solvent atom is not a hydrogen') 
             QVHLJ=(ABS(EFF(ITC(IAC(I)))) >  0.000001)  !EFF should be  <=  0
             ! Force neglect of LJ on water hydrogens? 
             !  NB: For TIP3-TIP3 interactions the parameter file still rules.
             !      May need to be changed.
             QVHLJ=QVHLJ.AND. (INDXA(COMLYN,COMLEN,'NOVH') == 0)
             IF(QVHLJ.AND.WATMODEL == WTIP4P)THEN
                CALL WRNDIE(-1,'<WWSETUP>', &
                     'TIP4P with L-J on hydrogens not implemented in lookup tables')
                QVHLJ=.FALSE.
             ENDIF
             !
             IF(PRNLEV >= 2 .AND. QVHLJ .AND. .NOT. QVU) WRITE(OUTU,'(A)') &
                  ' Using L-J interactions for solvent hydrogens'
             !
             ! Setup additonal pointer and flag arrays
             ! Also required for VU
             CALL MKWWFLG(NATOM,NWWO)
             ! 3-site models must have same atomtype and charge in positions 2 and 3 
             I=IWWO(1)+1
             J=IWWO(1)+2
             IF( (CG(I) /= CG(J) ) .OR. (IACNB(I).NE.IACNB(J)))THEN
                CALL WRNDIE(1,'<WWSETUP>', &
                     'Incorrect 3-site model. Using standard routine.')
                QLOOKUP=.FALSE.
                QVV=.FALSE.
                QVU=.FALSE.
                QUU=.FALSE.
                NWWO=0
                RETURN
             ENDIF
             ! All OK so far
             !
             ! VV-list will use oxygen-oxygen distance. CTDDW is the necessary margin 
             ! to make sure no H-H interaction is thrown out in the only distance test
             ! in EVVTB3. Water model may not be rigid so use a default margin slightly
             ! larger than twice the O-H bond length, and hope the first water is normal..
             I=IWWO(1)
             J=IWWO(1)+1
             CTDDW=2.1*SQRT((X(I)-X(J))**2+(Y(I)-Y(J))**2+(Z(I)-Z(J))**2)
             CTDDW=GTRMF(COMLYN,COMLEN,'DD',CTDDW)
             IF(PRNLEV >= 2) WRITE(OUTU,'(/I9,A/) ') NWWO, &
                  ' residues selected for 3-site solvent-solvent lookup tables'
          ENDIF
       CASE(WTIP4P)
          IF(PRNLEV >= 2)  &
               WRITE(OUTU,'(A)') ' TIP4P water model is not supported yet'
          QLOOKUP=.FALSE.
          QVV=.FALSE.
          QVU=.FALSE.
          QUU=.FALSE.
          NWWO=0
          RETURN
       END SELECT
       !
       ! Allocate table space, and fill tables. Requires that ENERGY has been called 
       ! to have codes and coefficient arrays properly filled.
       MXTB=TBSIZ*CTOFNB*CTOFNB
       MXTB1=MXTB+1
       IF(ALLOCATED(EVV)) DEALLOCATE(EVV)
       IF(ALLOCATED(FVV)) DEALLOCATE(FVV)
       ! Three/four site water models, with extra space for interpolation 
       SELECT CASE (WATMODEL)
       CASE (W3SITE)
          NVVTAB=3        
       CASE (WTIP4P)
          ! Specific for TIP4P w/ no LJ on H/LP and no q on O (OO,HH,H-LP, LP-LP) 
          NVVTAB=4 
       END SELECT
       IF(QWLKUPINT) NVVTAB=NVVTAB*2
       IF(IWWENR /= 0) ALLOCATE(EVV(MXTB1,NVVTAB))
       ALLOCATE(FVV(MXTB1,NVVTAB))
    ENDIF ! QVV
    !
    ! Solvent-solute stuff
    IF(QVU)THEN
       !
       ! Nblist(s)
       IF(.NOT.ALLOCATED(IVUNBL)) ALLOCATE(IVUNBL(MAXAIM))
       IF(.NOT.ALLOCATED(IVUNBLI) .AND. NTRANS > 0) &
            ALLOCATE(IVUNBLI(MAXAIM))
       !
       IF(ALLOCATED(EVUC)) DEALLOCATE(EVUC)
       IF(ALLOCATED(FVUC)) DEALLOCATE(FVUC)
       IF(ALLOCATED(EVULJ)) DEALLOCATE(EVULJ)
       IF(ALLOCATED(FVULJ)) DEALLOCATE(FVULJ)
       ! 2 Coulomb tables suffice for VU interactions with TIP3,SPC,SPC/E or TIP4
       !   (EVU(*,1) is for O or LP, EVUC(*,2) is for H)
       ! For now if interpolation is used it has to be done within the tables, in contrast to VV
       IF(IWWENR /= 0) ALLOCATE(EVUC(MXTB1,2))
       ALLOCATE(FVUC(MXTB1,2))
       I=1
       IF(QVHLJ) I=2 
       NVUTAB=NITCC
       IF(IWWENR /= 0) ALLOCATE(EVULJ(MXTB1,NVUTAB,I))
       ALLOCATE(FVULJ(MXTB1,NVUTAB,I))
    ENDIF !QVU
    !
    IF(QUU)THEN
       !
       ! SolUte-solUte tables use standard NBlist so we only need the force and energy tables
       IF(ALLOCATED(EUUC)) DEALLOCATE(EUUC)
       IF(ALLOCATED(FUUC)) DEALLOCATE(FUUC)
       IF(ALLOCATED(EUULJ)) DEALLOCATE(EUULJ)
       IF(ALLOCATED(FUULJ)) DEALLOCATE(FUULJ)
       IF(.NOT.ALLOCATED(IUUNBL)) ALLOCATE(IUUNBL(MAXAIM))
       ! Hm - requires IMAGES to be set up before LOOKUP, but this may not be good or needed???
       IF(.NOT.ALLOCATED(IUUNBLI) .AND. NTRANS > 0) &
            ALLOCATE(IUUNBLI(MAXAIM))

       ! 1 Coulomb table suffices for UU 
       ! For now if interpolation is used it has to be done within the tables, in contrast to VV
       IF(IWWENR /= 0) ALLOCATE(EUUC(MXTB1))
       ALLOCATE(FUUC(MXTB1))
       NUUTAB=NITCC*NITCC
       IF(IWWENR /= 0) ALLOCATE(EUULJ(MXTB1,NITCC,NITCC))
       ALLOCATE(FUULJ(MXTB1,NITCC,NITCC))
    ENDIF !QUU
    !
    ! Tell user what to expect
    IF(PRNLEV >= 2)THEN
       SELECT CASE(IWWENR)
       CASE(2)
          WRITE(OUTU,'(A)') ' Including table energies'
       CASE(0)
          WRITE(OUTU,'(A)') ' Not including table energies'
       CASE DEFAULT
          WRITE(OUTU,'(A)') &
               ' Including table energies at non-bond updates'
       END SELECT

       IF(QWLKUPINT)  &
            WRITE(OUTU,'(A)') ' Using linear interpolation'
       I=2
       IF(IWWENR == 0) I=1
       WRITE(OUTU,100) 1.0/TBSIZ,MXTB1
100    FORMAT(/'        Table R**2 increment (A**2): ',F8.5, &
            /'                       Points/table: ',I8)
       IF(QVV)THEN
          WRITE(OUTU,'(//A,I8)')  &
               ' Solvent-solvent table size (bytes): ',I*NVVTAB*4*MXTB1
          WRITE(OUTU,'(A,F8.2)')  &
               '          Molecular size buffer (A):',CTDDW 
       ELSE
          WRITE(OUTU,'(//A)') 'Not using solvent-solvent lookup'
       ENDIF
       !
       IF(QVU)THEN
          IF(QVHLJ)THEN
             WRITE(OUTU,'(/A,A)') &
                  ' Using solvent-solute tables', &
                  ' with solvent hydrogen L-J interactions'
          ELSE
             WRITE(OUTU,'(/A,A)') &
                  ' Using solvent-solute tables', &
                  ' without solvent hydrogen L-J interactions'
          ENDIF
          J=1
          IF(IWWENR /= 0) J=2 
          I=1
          IF(QVHLJ) I=2
          WRITE(OUTU,'(A,I10)')  &
               '              Number of L-J tables:',NVUTAB*I, &
               '          Number of Coulomb tables:',2, &
               ' Solvent-solute table size (bytes):', &
               J*MXTB1*4*(NVUTAB*I+2)
       ELSE  !QVU
          ! Issue message in case it would have been possible to have VU tables
          IF(NWWO < NRES) WRITE(OUTU,'(/A)') &
               ' Not using lookup tables for solvent-solute'
       ENDIF !QVU
       !
       IF(QUU)THEN
          WRITE(OUTU,'(/A)') &
               ' Using solute-solute tables'
          J=1 
          IF(IWWENR /= 0) J=2 
          WRITE(OUTU,'(A,I10)')  &
               '                 Number of  tables:',NUUTAB+1, &
               '  Solute-solute table size (bytes):', &
               J*MXTB1*4*(NUUTAB+1)
       ELSE  !QUU
          ! Issue message in case it would have been possible to have UU tables
          IF(NWWO < NRES) WRITE(OUTU,'(/A)') &
               ' Not using lookup tables for solute-solute'
       ENDIF !QUU     
    ENDIF ! PRNLEV
    !
    ! Now fill tables
    ! NB! AT PRESENT THIS REQUIRES THAT ENERGY HAS BEEN CALLED BEFORE,
    ! SO THAT COEFFICIENT ARRAYS ARE PROPERLY FILLED
    !
    CALL TABSETUP(NATOM)
    !
    RETURN
  END SUBROUTINE WWSETUP
  !
  SUBROUTINE MKWWFLG(NATOMX,NWWOX)
    ! 
    ! Setup flags and index arrays for waters to be handled by lookup
    ! Just picking the oxygens
    !
    use chm_kinds
    !  use LOOKUP
    use chutil,only:hydrog,lone
    IMPLICIT NONE
    INTEGER NATOMX,NWWOX
    INTEGER I 
    !
    NWWOX=0
    DO I=1,NATOMX         
       IF(QVU)THEN
          ELEOFFS(I)=0
          VDWOFFS(I)=0
          IF(IWWFLG(I) /= 0)THEN
             IF(HYDROG(I))THEN
                ELEOFFS(I)=IHYD
                IF(QVHLJ) VDWOFFS(I)=IHYD
             ELSE IF(LONE(I))THEN
                ELEOFFS(I)=ILP
             ELSE 
                VDWOFFS(I)=IOX
                IF(WATMODEL == W3SITE) ELEOFFS(I)=IOX
             ENDIF
          ENDIF
       ENDIF
       IF(IWWFLG(I) == I)THEN
          NWWOX=NWWOX+1
          IWWO(NWWOX)=I
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE MKWWFLG
  !
  !
  SUBROUTINE MKWWGRP(NATOM,IWWFLG)
    !
    ! Change all non-zero entries to index of first atom in molecule to which 
    ! I belongs, assuming it is (water) ordered as: (O, H ,H)
    !CCC Will need revision when TIP4P support is added
    ! NO ERROR CHECKS!
    !
    use chm_kinds
    use chutil,only:hydrog
    IMPLICIT NONE
    INTEGER NATOM,IWWFLG(*)
    !
    INTEGER I
    DO I=1,NATOM
       IF(IWWFLG(I) > 0)THEN
          IWWFLG(I)=I
          IF(HYDROG(I) )THEN
             IWWFLG(I)=I-1
             IF(HYDROG(I-1))IWWFLG(I)=I-2
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE MKWWGRP
  !
  SUBROUTINE TABSETUP(NATOMX)
    !
    ! Allocate space and fill tables
    !
    use chm_kinds
    !  use LOOKUP
    use dimens_fcm
    use exfunc
    use number
    use fast
    use param
    use psf
    use stream
    IMPLICIT NONE
    INTEGER NATOMX
    !
    INTEGER IAT,JAT,I,J,K,L
    LOGICAL QENE,QELE,QVDW
    real(chm_real) DR2
    ! Temporary local allocation
    INTEGER, ALLOCATABLE :: INBL(:),LJIDX(:)
    !
    IF(.NOT. ALLOCATED(INBL)) ALLOCATE(INBL(NATOMX))
    IF(.NOT. ALLOCATED(LJIDX)) ALLOCATE(LJIDX(NITCC))
    INBL=0
    !
    ! Solvent-solvent; three or four site models allowed assuming this structure
    ! Three site: O,H,H     (TIP3P,SPC,SPC/E) - O-O,O-H,H-H
    ! Four site:  O,LP,H,H  (TIP4P) - O-O (vdW),H-H (Coul),LP-LP (coul),LP-H (coul) 
    !                                 and NO INTERACTIONS BETWEEN O and H/LP?
    QELE=.TRUE.
    QVDW=.TRUE.
    QENE=(IWWENR /= 0)
    DR2=ONE/TBSIZ
    IF(QVV)THEN
       !
       ! OO tables
       ! NB: Two distinct atoms have to be used to setup these tables
       IAT=IWWO(1)
       JAT=IWWO(2)
       !
       CALL FILLTAB(EVV(1,1),FVV(1,1),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
            QENE,QELE,QVDW)
       SELECT CASE (WATMODEL)
       CASE (W3SITE) 
          ! Three site solvent model
          ! OH tables
          JAT=JAT+1 
          CALL FILLTAB(EVV(1,2),FVV(1,2),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
               QENE,QELE,QVDW)
          ! HH tables
          IAT=IAT+1 
          CALL FILLTAB(EVV(1,3),FVV(1,3),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
               QENE,QELE,QVDW)
          IF(PRNLEV >= 7) WRITE(OUTU,*) 'THREE-SITE MODEL. MXTB1=',MXTB1
       CASE(WTIP4P)
          ! Four site model (TIP4P)
          ! HH tables
          IAT=IAT+2
          JAT=JAT+2 
          CALL FILLTAB(EVV(1,2),FVV(1,2),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
               QENE,QELE,QVDW)
          ! LP-LP tables
          IAT=IAT-1       
          JAT=JAT-1
          CALL FILLTAB(EVV(1,3),FVV(1,3),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
               QENE,QELE,QVDW)
          ! H-LP tables
          JAT=JAT+1 
          CALL FILLTAB(EVV(1,4),FVV(1,4),DR2,RNDING,MXTB1,IAT,JAT,INBL, &
               QENE,QELE,QVDW)
          IF(PRNLEV >= 7) WRITE(OUTU,*) 'TIP4P MODEL. MXTB1=',MXTB1
       END SELECT
       !
       ! Precompute differences for interpolation if requested
       IF(QWLKUPINT)THEN
          DO J=1,NVVTAB/2
             DO I=1,MXTB
                IF(QENE) EVV(I,J+NVVTAB/2)=EVV(I+1,J)-EVV(I,J)
                FVV(I,J+NVVTAB/2)=FVV(I+1,J)-FVV(I,J)
             ENDDO
             IF(QENE) EVV(MXTB1,J+NVVTAB/2)=0.0
             FVV(MXTB1,J+NVVTAB/2)=0.0
          ENDDO
       ENDIF
       ! Debugging output
       IF(PRNLEV == 6)THEN
          WRITE(OUTU,'(/A)') 'Solvent-solvent energy table EVV:'
          ! Would be simpler with an array of names, but...
          SELECT CASE (NVVTAB)
          CASE (3)
             WRITE(OUTU,'(4A12)') 'R2','OO','OH','HH'
          CASE (4)
             WRITE(OUTU,'(5A12)') 'R2','OO','HH','LP-LP','H-LP'
          CASE (6)
             WRITE(OUTU,'(7A12)') 'R2','OO','OH','HH', &
                  'DOO','DOH','DHH'
          CASE (8)
             WRITE(OUTU,'(9A12)') 'R2','OO','HH','LP-LP','H-LP', &
                  'DHH','DLP-LP','DH-LP'
          END SELECT
          DO I=1,MXTB1
             WRITE(OUTU,'(9E12.4)') (I+RNDING)*DR2,(EVV(I,J),J=1,NVVTAB)
          ENDDO
          !
          WRITE(OUTU,'(/A)') 'Solvent-solvent force table FVV:'
          SELECT CASE (NVVTAB)
          CASE (3)
             WRITE(OUTU,'(4A12)') 'R2','OO','OH','HH'
          CASE (4)
             WRITE(OUTU,'(5A12)') 'R2','OO','HH','LP-LP','H-LP'
          CASE (6)
             WRITE(OUTU,'(7A12)') 'R2','OO','OH','HH', &
                  'DOO','DOH','DHH'
          CASE (8)
             WRITE(OUTU,'(9A12)') 'R2','OO','HH','LP-LP','H-LP', &
                  'DHH','DLP-LP','DH-LP'
          END SELECT
          DO I=1,MXTB1
             WRITE(OUTU,'(9E12.4)') (I+RNDING)*DR2,(FVV(I,J),J=1,NVVTAB)
          ENDDO
       ENDIF
    ENDIF !QVV
    !           
    IF(QVU)THEN
       !
       ! SolVent-solUte tables
       !
       ! Coulomb tables, any two distinct atoms will do here
       IAT=IWWO(1)
       JAT=IWWO(2)
       QVDW=.FALSE.
       CALL FILLTAB(EVUC(1,IOX),FVUC(1,IOX),DR2,RNDING,MXTB1, &
            IAT,JAT,INBL,QENE,QELE,QVDW)
       ! Now premultiply with water site charges
       SELECT CASE(WATMODEL)
       CASE(W3SITE)
          IAT=IWWO(1)   ! The oxygen
       CASE(WTIP4P)
          IAT=IWWO(1)+1 ! The L-P carries the charge here
       END SELECT
       JAT=IWWO(1)+2   ! This is a hydrogen for both water models  
       DO I=1,MXTB1
          FVUC(I,IHYD)=CG(JAT)*FVUC(I,IOX)   
          FVUC(I,IOX)=CG(IAT)*FVUC(I,IOX)   
          IF(QENE)THEN
             EVUC(I,IHYD)=CG(JAT)*EVUC(I,IOX)   
             EVUC(I,IOX)=CG(IAT)*EVUC(I,IOX)   
          ENDIF
       ENDDO
       !
       !  Lennard-Jones
       QELE=.FALSE.
       QVDW=.TRUE. 
       !
       ! Now loop over all atom types
       DO J=1,NITCC
          ! Find first  atom with compressed vdW type = IACNB(J)
          JAT=FIND52(IACNB,0,0,0,0,J,0,0,0,0,NATOMX,1,-1)
          ! and save it for later reference
          LJIDX(J)=JAT
          IF(JAT == -1)  &
               CALL WRNDIE(-3,'<TABSETUP>','Internal error/IACNB')
          ! JAT cannot be the same as IAT so we use the second water molecule for IAT
          IAT=IWWO(2)   ! The oxygen for both water models
          CALL FILLTAB(EVULJ(1,J,IOX),FVULJ(1,J,IOX),DR2,RNDING,MXTB1, &
               IAT,JAT,INBL,QENE,QELE,QVDW)
          IF(QVHLJ)THEN
             ! The hydrogen for both water models (but TIP4P should not use LJ on H)
             IAT=IWWO(2)+2
             CALL FILLTAB(EVULJ(1,J,IHYD),FVULJ(1,J,IHYD), &
                  DR2,RNDING,MXTB1,IAT,JAT,INBL,QENE,QELE,QVDW)
          ENDIF
       ENDDO
       !
       IF(PRNLEV == 6)THEN
          WRITE(OUTU,'(/A)') 'SolVent-solUte Coulomb tables'
          IF(QENE)THEN
             WRITE(OUTU,'(5A12)') ' R2','FO/LP','FH','EO/LP','EH'
             DO I=1,MXTB1
                WRITE(OUTU,'(5E12.4)') DR2*(I+RNDING), &
                     FVUC(I,IOX),FVUC(I,IHYD),EVUC(I,IOX),EVUC(I,IHYD)
             ENDDO
          ELSE
             WRITE(OUTU,'(3A12)') ' R2','FO/LP','FH'
             DO I=1,MXTB1
                WRITE(OUTU,'(3E12.4)') DR2*(I+RNDING), &
                     FVUC(I,IOX),FVUC(I,IHYD)
             ENDDO
          ENDIF
          WRITE(OUTU,'(/A)') 'SolVent-solUte van der Waals tables.'
          IF(QENE)THEN
             IF(QVHLJ)THEN
                DO K=1,NVUTAB
                   L=LJIDX(K)
                   WRITE(OUTU,'(A,I5,A10)') &
                        'Atom type:',IAC(L),ATC(IAC(L))
                   WRITE(OUTU,'(5A12)') ' R2','FO','FH','EO','EH'
                   DO I=1,MXTB1
                      WRITE(OUTU,'(5E12.4)') DR2*(I+RNDING), &
                           FVULJ(I,K,IOX),FVULJ(I,K,IHYD), &
                           EVULJ(I,K,IOX),EVULJ(I,K,IHYD)
                   ENDDO
                ENDDO
             ELSE  !QVHLJ
                DO K=1,NVUTAB
                   L=LJIDX(K)
                   WRITE(OUTU,'(A,I5,A10)') &
                        'Atom type:',IAC(L),ATC(IAC(L))
                   WRITE(OUTU,'(3A12)') ' R2','FO','EO'
                   DO I=1,MXTB1
                      WRITE(OUTU,'(5E12.4)') DR2*(I+RNDING), &
                           FVULJ(I,K,IOX),EVULJ(I,K,IOX)
                   ENDDO
                ENDDO
             ENDIF !QVHLJ 
          ELSE  !QENE
             IF(QVHLJ)THEN
                DO K=1,NVUTAB
                   L=LJIDX(K)
                   WRITE(OUTU,'(A,I5,A10)')  &
                        'Atom type:',IAC(L),ATC(IAC(L))
                   WRITE(OUTU,'(3A12)') ' R2','FO','FH'
                   DO I=1,MXTB1
                      WRITE(OUTU,'(3E12.4)') DR2*(I+RNDING), &
                           FVULJ(I,K,IOX),FVULJ(I,K,IHYD)
                   ENDDO
                ENDDO
             ELSE  !QVHLJ
                DO K=1,NVUTAB
                   L=LJIDX(K)
                   WRITE(OUTU,'(A,I5,A10)') &
                        'Atom type:',IAC(L),ATC(IAC(L))
                   WRITE(OUTU,'(2A12)') ' R2','FO'
                   DO I=1,MXTB1
                      WRITE(OUTU,'(2E12.4)') DR2*(I+RNDING), &
                           FVULJ(I,K,IOX)
                   ENDDO
                ENDDO
             ENDIF !QVHLJ 
          ENDIF !QENE
       ENDIF !PRNLEV
    ENDIF !QVU
    !
    IF(QUU)THEN
       ! SolUte-solUte tables
       !
       ! Coulomb table, any two distinct atoms will do here
       IAT=1
       JAT=2
       QELE=.TRUE.
       QVDW=.FALSE.
       CALL FILLTAB(EUUC,FUUC,DR2,RNDING,MXTB1, &
            IAT,JAT,INBL,QENE,QELE,QVDW)
       ! LJ tables
       QELE=.FALSE.
       QVDW=.TRUE.
       DO I=1,NITCC
          ! Find first atom with type IACNB(I)
          IAT=FIND52(IACNB,0,0,0,0,I,0,0,0,0,NATOMX,1,-1)
          DO J=I,NITCC
             ! Find last atom with type IACNB() == J, 
             !   so that we have a chance of having distinct IAT and JAT
             JAT=IAT
             DO K=NATOMX,1,-1
                IF(J == IACNB(K))THEN
                   JAT=K
                   EXIT
                ENDIF
             ENDDO
             IF(IAT == JAT) CYCLE
             !           
             CALL FILLTAB(EUULJ(1,J,I),FUULJ(1,J,I),DR2,RNDING,MXTB1, &
                  IAT,JAT,INBL,QENE,QELE,QVDW)
             IF(I /= J)THEN
                EUULJ(1:MXTB1,I,J)=EUULJ(1:MXTB1,J,I)
                FUULJ(1:MXTB1,I,J)=FUULJ(1:MXTB1,J,I)
             ENDIF
          ENDDO
       ENDDO
    ENDIF !QUU
    !
    IF(ALLOCATED(INBL)) DEALLOCATE(INBL)
    IF(ALLOCATED(LJIDX)) DEALLOCATE(LJIDX)
    RETURN
  END SUBROUTINE TABSETUP
  !
  SUBROUTINE FILLTAB(ETAB,FTAB,DR2,RNDING,MXTAB1, &
       IAT,JAT,INBL,QENERGY,QELE,QVDW)
    !
    !     Fill energy and force tables using R2 index for two distinct atoms pointed
    !     to by IAT and JAT.
    !
    !     QELE  QVDW     Coulomb                     vdW
    !     T     T        Yes                         Yes
    !     F     T        No                          Yes
    !     T     F        Yes, but with unit charges  No  
    !
    !     Uses the non-bond setup of the most recent energy call.
    !     NB! An energy call is needed before calling FILLTAB
    !      in order to properly define all parameter arrays.
    !
    use chm_kinds
    use dimens_fcm
    use coord
    use deriv
    use fast
    use inbnd
    use psf
    use stream
    !
    IMPLICIT NONE
    INTEGER IAT,JAT,MXTAB1,INBL(*)
    real(chm_real4) :: ETAB(MXTAB1),FTAB(MXTAB1)
    real(chm_real) :: DR2
    real(chm_real8) :: RNDING
    LOGICAL QENERGY,QELE,QVDW
    ! Local variables
    INTEGER J,JNBL(1),IFRSTA,NATOMX,MXTAB
    real(chm_real)  XI,YI,ZI,XJ,YJ,ZJ,DXI,DYI,DZI,DXJ,DYJ,DZJ, &
         CHGI,CHGJ, &
         ENB,EEL,ETMP,FTMP
    LOGICAL QBYCBIM,QBYPBCB,LELECX,LVDWX,LUSED
    !
    ! Rudimentary consistency checks
    IF(.NOT. (QELE.OR.QVDW)) RETURN
    IF(IAT == JAT) CALL WRNDIE(-2,'<FILLTAB>','IAT=JAT not allowed')
    ! Save state
    !
    XI=X(IAT)
    YI=Y(IAT)
    ZI=Z(IAT)
    XJ=X(JAT)
    YJ=Y(JAT)
    ZJ=Z(JAT)
    DXI=DX(IAT)
    DYI=DY(IAT)
    DZI=DZ(IAT)
    DXJ=DX(JAT)
    DYJ=DY(JAT)
    DZJ=DZ(JAT)
#if KEY_IMCUBES==1
    QBYCBIM=LBYCBIM     
#endif
    !
    ! Initialize
    X(IAT)=0.0
    Y(IAT)=0.0
    Z(IAT)=0.0
    X(JAT)=0.0
    Y(JAT)=0.0
    Z(JAT)=0.0
    DX(JAT)=0.0
#if KEY_IMCUBES==1
    LBYCBIM=.FALSE.     
#endif
    IFRSTA=IAT
    NATOMX=IAT
    INBL(IAT)=1
    JNBL(1)=JAT
    IF(QELE.AND. .NOT. QVDW)THEN
       CHGI=CG(IAT)
       CHGJ=CG(JAT)
       CG(IAT)=1.0
       CG(JAT)=1.0
    ENDIF
    LELECX=LELEC.AND.QELE
    LVDWX=LVDW.AND.QVDW
    !CCC  
    !CCC    ENBAEXP requires LVDWX to be TRUE, so we have to cheat a little if QVDW=.FALSE
    !CCCC eg, do two calls: one with CG=1 and one with CG=0 => E/F Coul = call CG=1  - call CG=0.
    !CCC  or use CCBNBA-CCNBD zerofilled dummy arrays
    MXTAB=MXTAB1-1
    !
    !      write(*,*) 'DBG-FILLTAB'
    !      write(*,*) 'EWLDT,IBLCKP,BLCOEE,BLCOEV=',EWLDT,IBLCKP,BLCOEE,BLCOEV
    !dbg to get past bounds checks is not essential here
    !      ewldt=1
    !      iblckp=1
    !      blcoee=1
    !      blcoev=1
    !      blcoevr=1
    !      blcoeva=1
    DO J=1,MXTAB
       ! For direct lookup the roundoff correction (fortran trunactes) is precompensated here
       X(JAT)=SQRT(DR2*(J+RNDING))
       DX(JAT)=0.0
       ENB=0.0
       EEL=0.0 
       IF(.NOT. QVDW .AND. QELE)THEN
          ! Two calls...
          LELECX=.TRUE.
          LVDWX=.TRUE.
          CALL  ENBAEXP(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
               CG,JNBL,INBL, &
               IACNB, &
               LUSED)
          IF(.NOT.LUSED) CALL WRNDIE(-2,'<FILLTAB>', &
               'Unsupported combination of nonbond options')
          IF(QENERGY) ETAB(J)=EEL
          FTMP=DX(JAT)/X(JAT)
          LELECX=.FALSE.
          DX(JAT)=0.0
          CALL  ENBAEXP(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
               CG,JNBL,INBL, &
               IACNB, &
               LUSED)
          IF(.NOT.LUSED) CALL WRNDIE(-2,'<FILLTAB>', &
               'Unsupported combination of nonbond options')

          FTAB(J)=FTMP-DX(JAT)/X(JAT)
       ELSE 
          CALL  ENBAEXP(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
               CG,JNBL,INBL, &
               IACNB, &
               LUSED)

          !
          ! We only support the options available in ENBAEXP (but this restriction can easily 
          ! be lifted, through calling ENBOND (perhaps?); ENBAEXP has a simpler interface ...
          !
          IF(.NOT.LUSED) CALL WRNDIE(-2,'<FILLTAB>', &
               'Unsupported combination of nonbond options')
          FTAB(J)=DX(JAT)/X(JAT)
          IF(QENERGY)THEN
             ETAB(J)=0.0
             IF(QELE) ETAB(J)=EEL
             IF(QVDW) ETAB(J)=ETAB(J)+ENB
          ENDIF
       ENDIF
    ENDDO
    !
    FTAB(MXTAB1)=0.0
    IF(QENERGY) ETAB(MXTAB1)=0.0
    !
    ! Restore state 
    INBL(IAT)=0
    X(IAT)=XI
    Y(IAT)=YI
    Z(IAT)=ZI
    X(JAT)=XJ
    Y(JAT)=YJ
    Z(JAT)=ZJ
    DX(IAT)=DXI
    DY(IAT)=DYI
    DZ(IAT)=DZI
    DX(JAT)=DXJ
    DY(JAT)=DYJ
    DZ(JAT)=DZJ
#if KEY_IMCUBES==1
    LBYCBIM=QBYCBIM                    
#endif
    IF(QELE.AND. .NOT. QVDW)THEN
       CG(IAT)=CHGI
       CG(JAT)=CHGJ
    ENDIF
    RETURN 
  END SUBROUTINE FILLTAB
#endif /* (lookup_main)*/
END MODULE LOOKUP 

