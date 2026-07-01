module mce
#if KEY_MC==1
  use chm_types
  use dimens_fcm
  implicit none
contains

  SUBROUTINE MCENER(ECNT,MVTYPE,IDX,IMVNG,IBLSTP,IMCGRP,NOECNP, &
       ISKIP,IMVATM,IMVGRP,MBONDT,QBND,MCBLOP,MCBLGP, &
       MCIMLP,MCIMGP,LFIRST,BDUMMY,BDIMMY,MCATTP, &
       IAMC,NAMC,IMLIMP,MCA14P,IGMC,NGMC,MCGTTP,X,Y,Z, &
       LGCMC, &
#if KEY_PERT==1
       MCBLORP, MCBLOPP,BDUMMYR, BDUMMYP,  & 
       MCIMLRP,MCIMLPP,IMLIMRP,IMLIMPP,    & 
       BDIMMYR,BDIMMYP,    & 
#endif /* KEY_PERT */
       EGSBP)
    !
    !       Computes energy contribution of the moving atoms and returns
    !       the total in ECNT.
    !
    !       In the interest of computational efficiency, Monte Carlo calls
    !       specific energy routines directly, rather than through the main
    !       ENERGY routine.  As a result, not all energy terms are supported.
    !       Those that are supported are bonds, angles, Urey-Bradley, dihedrals,
    !       impropers, vdw, electrostatic, image vdw, image electrostatic,
    !       QM/MM (MOPAC/SCCDFTB), path integral, asp-EEF1, asp-ACE/ACS, NOE
    !       constraints, and user.
    !
    !       All non-bonded calculations can be either atom or group based
    !
    !       Aaron R. Dinner
    !
    use ace_module
    use bases_fcm
    use deriv
    use econtmod
    use eef1_mod
    use energym
    use inbnd
    use image
    use noem
    use number
    use pbeq
#if KEY_PERT==1
    use pert                            
#endif
    use psf
    use quantm
    use stream
#if KEY_SCCDFTB==1
    use sccdftb                          
#endif
    use mcimg, only: mcimge
    use mcmvutil, only: tagatm
#if KEY_GRID==1
    use grid_dock                             
#endif
#if KEY_RGYCONS==1
  use rgym, only: qrgy, ergy  
#endif



    !       Passed Variables
    !
  

    INTEGER MVTYPE, IDX, MBONDT
    type(chm_iptr) :: IMVNG(:), IBLSTP(:), IMCGRP(:), NOECNP(:)
    INTEGER NSKIP, ISKIP(:), IMVATM(:), IMVGRP(:)
    type(chm_iptr),dimension(:) :: MCBLOP, MCBLGP, MCIMLP, MCIMGP, MCATTP, MCGTTP
    integer,dimension(:) :: IMLIMP
    type(nonbondDataStructure) BDUMMY
    type(imageDataStructure) BDIMMY

    INTEGER IAMC, NAMC, IGMC, NGMC
    type(chm_iptr),dimension(:) :: MCA14P
#if KEY_PERT==1 /*pert_decl*/
    type(chm_iptr),dimension(:) :: MCBLORP, MCBLOPP
    type(nonbondDataStructure) BDUMMYR
    type(nonbondDataStructure) BDUMMYP
    type(chm_iptr),dimension(:) :: MCIMLRP, MCIMLPP
    integer,dimension(:) :: IMLIMRP, IMLIMPP
    type(imageDataStructure) BDIMMYR
    type(imageDataStructure) BDIMMYP
#endif /* (pert_decl)*/
    real(chm_real)  ECNT, EGSBP
    real(chm_real), dimension(:) :: x, y, z
    LOGICAL QBND(MBONDT), LFIRST, LGCMC
#if KEY_SCCDFTB==1
    real(chm_real) ECRAP                /*qc_010110*/
#endif
    !
    !       Local Variables
    !
    INTEGER I, J, IAF, IAL, IAFM1, IALP1, ITRMP
    INTEGER K, IFIRST, ILAST
    integer,pointer,dimension(:) :: TEMGP, TEMPP
    real(chm_real)  ENBV, ENBE, EUSER, EIMV, EIME, ESLV, ENOE
    real(chm_real)  EQME, EQMV
#if KEY_GRID==1
    real(chm_real) EGrvdW,EGrElec,Ehbo          !! Yujin       
#endif
#if KEY_PERT==1
    real(chm_real)  EGSBPR                                   
#endif

   
     ECNT = ZERO
#if KEY_GRID==1
     egrvdw = zero 
#endif
#if KEY_GRID==1
     egrelec = zero 
#endif
#if KEY_GRID==1
     ehbo = zero 
#endif
    !
    !       Bonded terms.
    !
    ILAST = 0
    DO I = 1, MBONDT
       IF (QBND(I)) THEN
          ECNT = ECNT + EBNDED(I,ILAST,IDX, IBLSTP, ISKIP,X,Y,Z)
       ENDIF
    ENDDO


#if KEY_QUANTUM==1
    !       QM/MM terms (Mopac 4 only for now)
    IF (NATQM.GT.0) THEN
       ! DTM incorporate Mopac changes
       !          IF (QETERM(QMEL)) CALL EAMPAC(EQME)
       !          IF (NATOM.GT.NATQM .AND. QETERM(QMVDW)) CALL EVDWQM(EQMV)
       IF (QETERM(QMEL)) THEN
          CALL GESDEN(.FALSE.)
          CALL EAMPAC(EQME,X,Y,Z,DX,DY,DZ)
       ENDIF
       IF (NATOM.GT.NATQM .AND. QETERM(QMVDW))  &
            CALL EVDWQM(EQMV,X,Y,Z,DX,DY,DZ)
       ECNT = ECNT + EQME + EQMV
    ENDIF
#endif 
#if KEY_SCCDFTB==1
    !       QM/MM terms (SCCDFTB here - a simple treatment)
    IF (NSCCTC.GT.0) THEN
       !       XIAO_QC_UW0609 (add force flag to SCCTBENE)
       IF (QETERM(QMEL)) CALL SCCTBENE(EQME,X,Y,Z,DX,DY,DZ,NATOM,.true.)
       !       Since SCC/MM van der waals is evaluated as the std term
       !       we essentially don't have to do anything here
       ECNT = ECNT + EQME
    ENDIF
#endif 
    !       All moves change the non-bonded terms with the non-moving atoms.

    !       Tag moving atoms with TAGATM and TAGGRP
    !       Get first and last moving atoms and groups with GTFLMV
    TEMPP => IMVNG(IDX)%A
    IF (LFIRST) THEN
       IF (LGROUP) THEN
          TEMGP => IMCGRP(IDX)%A
          CALL GTFLMV(IGMC, NGMC, TEMGP, NGRP)
          IAMC = IGPBS(IGMC) + 1
          NAMC = IGPBS(NGMC  + 1)
          CALL TAGGRP(IMVGRP, IMVATM, TEMGP ,.FALSE., 1, &
               IGPBS)
       ELSE
          CALL GTFLMV(IAMC, NAMC, TEMPP, NATOM)
          CALL TAGATM(IMVATM, TEMPP, .FALSE., 1)
       ENDIF
    ENDIF

    CALL MCNBND(ENBV,ENBE,IAMC,NAMC, TEMPP, IMVATM, &
         IMVGRP,LFIRST,BDUMMY,BNBND%INBLOG, &
         BNBND%JNBG, MCBLOP, MCBLGP, LGROUP, &
         IGMC,NGMC, MCA14P, LACE,X,Y,Z &
#if KEY_PERT==1
         ,MCBLORP, MCBLOPP, BDUMMYR,BDUMMYP & 
#endif
         )

    ECNT = ECNT + ENBV + ENBE

    !       Set up the images
    !       ARD:  ACE and images together will break I think
    IF(NTRANS.GT.0) THEN
       CALL MCIMGE(EIMV,EIME,IMVATM,IMVGRP,LFIRST, &
            BDIMMY, &
            BIMAG%IMBLO, BIMAG%IMJNB, &
            BIMAG%IMBLOG,BIMAG%IMJNBG, &
            BDUMMY, NATIM, MCIMLP, MCIMGP, &
            BIMAG%IMATPT, &
            NTRANS, MCATTP, LGROUP,IAMC,NAMC, &
            IGMC,NGMC, IMLIMP, MCGTTP, X,Y,Z,LGCMC &
#if KEY_PERT==1
            ,MCIMLRP, MCIMLPP, IMLIMRP, IMLIMPP, & 
            BDUMMYR,BDUMMYP, BDIMMYR,BDIMMYP &   
#endif /* KEY_PERT */
            )
       ECNT = ECNT + EIMV + EIME
    ELSE
       EIMV = ZERO
       EIME = ZERO
    ENDIF

    !       ARD 99-07-13
    !       NOE constraint terms
    IF ((NOENUM .GT. 0) .AND. QETERM(NOE)) THEN
       CALL MCNOEE(ENOE, NOECNP(IDX)%A, X, Y, Z)
       ECNT = ECNT + ENOE
    ENDIF

    !       ARD 98-10-16
    !       Use the non-bonded list for Lazaridis EEF1 solvent term
#if KEY_ASPENER==1
    IF (DOEEF1.AND.QETERM(ASP)) THEN
       !         Added last 3 args for 2nd deriv. I.A.
       CALL EEF1EN(ESLV,X,Y,Z,DX,DY,DZ,.FALSE.,ECONT,IAMC,NAMC, &
            IGMC,NGMC,BDUMMY%JNBG,BDUMMY%INBLOG, &
            BDUMMY%INB14,BDUMMY%IBLO14,.TRUE., &
            .FALSE. &
            )

       ECNT = ECNT + ESLV
    ENDIF
#endif 

#if KEY_GSBP==1

! Tim Miller (5-10-2011): QGSBP is only defined if PBEQ is 
! in pref.dat. If this is not the case, I am just setting 
! EGSBP to ZERO (as if QGSBP is .FALSE.). This should be 
! checked in test cases.
#if KEY_PBEQ==1 /*pbeqtst*/
    !       Total GSBP term is calculated for now.
    IF (QGSBP) THEN
       CALL GSBP0(NATOM,X,Y,Z,CG,EGSBP,DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
            ,.FALSE.,ECRAP &       /*qc_010110*/
#endif
            )
#if KEY_PERT==1 /*pert_gsbp*/
       IF (QPERT) THEN
          CALL GSBP0(NATOM,X,Y,Z,PPCG, &
               EGSBPR,DX,DY,DZ,1,.FALSE.&
#if KEY_SCCDFTB==1
               ,.FALSE.,ECRAP &       /*qc_010110*/
#endif
               )
          EGSBP = EGSBP*LAMDA + EGSBPR*LAMDAM
       ENDIF
#endif /*   (pert_gsbp)*/
       ECNT = ECNT + EGSBP
    ELSE
       EGSBP = ZERO
    ENDIF

#else /* (pbeqtst)*/
    EGSBP = ZERO
#endif /* (pbeqtst)*/

#endif 

#if KEY_ACE==1
    IF (LACE) THEN
       !         Set up the symmetric atom-exclusion list
       IF (LFIRST) CALL STUP14(BDUMMY%IBLO14, &
            BDUMMY%INB14, MCA14P, IMVATM,IAMC,NAMC)
    ENDIF
#endif 

#if KEY_RGYCONS==1
        IF(QRGY) THEN
           !.ab.
           CALL ERGY(ETERM(RGY),X,Y,Z,DX,DY,DZ)
           ecnt = ecnt + eterm(rgy)
        ENDIF
#endif 

#if KEY_GRID==1
!  Grid based vdW, Elec energy and hydrogen bond terms  !! Yujin
 !
    IF (QGrid .and. QGridOK) THEN
       !            Write(6,'(a)')' Call GridEnergy'
       !.ab.
       Call GridEnergy(EGrvdW,EGrElec, Ehbo, &
            XGridCenter, YGridCenter, ZGridCenter, &
            XGridMax, YGridMax, ZGridMax, DGrid, &
            GridForce, Cg, X, Y, Z, Dx, DY, DZ, &
            QETERM(GrvdW), QETerm(GrElec), QETerm(GrHBON), &
            QGrHBon, 0,0,.false.,GridNoSelection)
       ECNT = ECNT + EGrvdW + EGrElec + Ehbo
    ENDIF
#endif 


    !       Use the non-bonded list for user energy calculations
    !       If both user and images, add the BDIMMY data structures
    IF (QETERM(USER)) THEN
       IF (LGROUP) THEN
          CALL MCUSRG(EUSER,X,Y,Z,DX,DY,DZ,NGRP,NATOM, &
               BDUMMY%INBLOG,BDUMMY%JNBG, &
               BDUMMY%IBLO14,BDUMMY%INB14)
       ELSE
          CALL MCUSRA(EUSER,X,Y,Z,DX,DY,DZ,NATOM,IAMC,NAMC, &
               BDUMMY%INBLO,BDUMMY%JNB)
       ENDIF
       ECNT = ECNT + EUSER
    ENDIF

    !       Untag the moving atoms for the next round
    IF (LFIRST) THEN
       IF (LGROUP) THEN
          CALL TAGGRP(IMVGRP, IMVATM, TEMGP, .TRUE., 0, &
               IGPBS)
       ELSE
          CALL TAGATM(IMVATM, TEMPP, .TRUE., 0)
       ENDIF
    ENDIF

    !       WRITE (*,'(a8,999f20.8)') 'MCENER> ',ECNT,ENBV,ENBE,EIMV,EIME

    RETURN
  END SUBROUTINE MCENER

  FUNCTION EBNDED(IBNDT,IBL,IDX,IBLST,ISKIP,X,Y,Z) result(ebnded_1)
    !
    !       Sets up the skip arrays and then gets the energy for
    !       the bonded terms affected by the moving atoms
    !
    !       Aaron R. Dinner
    !
    use code
    use deriv
    use econtmod
    use eintern
    use energym
    use param
    use psf
#if KEY_PATHINT==1
    use mpathint, only: epimc  
#endif
    use memory
    use stream
    use cmapm
    use number

    real(chm_real) :: ebnded_1
    !
    !       Passed Variables
    !
    type(chm_iptr) :: IBLST(:)
    INTEGER IBNDT, ISKIP(*), IDX, IBL
    real(chm_real)  X(*), Y(*), Z(*)
    !
    !       Local Variables
    !
    INTEGER I, J, IBF, IBN, IBX, NB
    real(chm_real)  EUB
#if KEY_PATHINT==1
    INTEGER NBPIMC                              
    integer,allocatable,dimension(:) :: ITEMPP  
#endif

#if KEY_DOMDEC == 1
    call wrndie(-5, '<EBNDED>', 'NOT READY FOR DOMDEC')
    return
#endif /* KEY_DOMDEC */

    !       Initialization
    !
    EBNDED_1 = 0.0

    !       IBL contains the last read entry in the array off IBLST(IDX)
    !       The next entry contains how far to read for this term.
    IBF = IBL + 1
    IBL = IBLST(IDX)%A(IBF)
    IBF = IBF + 1

    !       Set up the skip array --- keep track of the range of terms to
    !       later trick energy routines.
    IBX = 0
    IBN = 999999999
    IF (IBL .GE. IBF) THEN
       DO I = IBF, IBL
          J = IBLST(IDX)%A(I)
          ISKIP(J) = 0
          IF (J .LT. IBN) IBN = J
          IF (J .GT. IBX) IBX = J
       ENDDO
       !
       !         Do the energy calculation.
       !         Trick the bonded routines into only looking at range of interest.
       NB = IBX - IBN + 1
       IF (IBNDT .EQ. 1 .AND. QETERM(BOND)) THEN
          CALL EBOND(EBNDED_1,IB(IBN),JB(IBN),ICB(IBN),NB,CBC,CBB, &
               DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP(IBN), &
               (/ZERO/),(/0/),.FALSE.,2,.FALSE. &
               )

       ELSE IF (IBNDT .EQ. 2) THEN
          IF (QETERM(ANGLE)) THEN
             CALL EANGLE(EBNDED_1,IT(IBN),JT(IBN),KT(IBN),ICT(IBN), &
                  NB,CTC,CTB,DX,DY,DZ,X,Y,Z,QECONT,ECONT,1, &
                  ISKIP(IBN),(/ZERO/),(/0/),.FALSE. &
                  )

          ENDIF
          IF (QETERM(UREYB)) THEN
             CALL EBOND(EUB,IT(IBN),KT(IBN),ICT(IBN),NB,CTUC,CTUB, &
                  DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP(IBN), &
                  (/ZERO/),(/0/),.FALSE.,2,.FALSE. &
                  )

             EBNDED_1 = EBNDED_1 + EUB
          ENDIF
       ELSE IF (IBNDT .EQ. 3 .AND. QETERM(DIHE)) THEN
          CALL EPHI(EBNDED_1,IP(IBN),JP(IBN),KP(IBN),LP(IBN),ICP(IBN), &
               NB,CPC,CPD,CPB,CPCOS,CPSIN,DX,DY,DZ,X,Y,Z, &
               .FALSE.,(/ZERO/),QECONT,ECONT,1,ISKIP(IBN),(/ZERO/),(/0/),.FALSE. &
               )

       ELSE IF (IBNDT .EQ. 4 .AND. QETERM(IMDIHE)) THEN
          CALL EPHI(EBNDED_1,IM(IBN),JM(IBN),KM(IBN),LM(IBN),ICI(IBN), &
               NB,CIC,CID,CIB,CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
               .FALSE.,(/ZERO/),QECONT,ECONT,1,ISKIP(IBN),(/ZERO/),(/0/),.FALSE. &
               )

#if KEY_CMAP==1
       ELSE IF (IBNDT .EQ. 4 .AND. QETERM(CMAP)) THEN
          CALL ECMAP(EBNDED_1,I1CT(IBN),J1CT(IBN),K1CT(IBN), &
               L1CT(IBN), &
               I2CT(IBN),J2CT(IBN),K2CT(IBN),L2CT(IBN), &
               ICCT(IBN),NB, &
               DX,DY,DZ,X,Y,Z, &
               QECONT,ECONT,1,ISKIP(IBN), (/ZERO/), (/0/), .FALSE.)
#endif 

#if KEY_PATHINT==1
          !         ARD 00-11-28
       ELSE IF (IBNDT .EQ. 5 .AND. QETERM(PINT)) THEN
          call chmalloc('mcener.src','EBNDED','ITEMPP',IBL-IBF+1,intg=ITEMPP)
          NBPIMC = 0
          DO I = IBF, IBL
             NBPIMC = NBPIMC + 1
             ITEMPP(NBPIMC) = IBLST(IDX)%A(I)
          ENDDO
          CALL EPIMC(EBNDED_1,AMASS,NBPIMC, ITEMPP, X,Y,Z)
          call chmdealloc('mcener.src','EBNDED','ITEMPP',IBL-IBF+1,intg=ITEMPP)
#endif 
       ENDIF
       !
       !         Reset the skip array for the next round.
       !
       DO I = IBF, IBL
          ISKIP(IBLST(IDX)%A(I)) = 1
       ENDDO
    ENDIF
  END FUNCTION EBNDED

  SUBROUTINE TAGGRP(IMVGRP,IMVATM,IMVNG,LVAL,IVAL,IGPBS)
    !
    !       Tags moving groups and atoms.
    !       If with LVAL is true,  tagged with IVAL.
    !       If with LVAL is false, tagged with MC group number in IMVNG.
    !
    !       Aaron R. Dinner
    !
    INTEGER IMVGRP(:), IMVATM(:), IMVNG(:), IVAL, IGPBS(*)
    LOGICAL LVAL
    !
    INTEGER I, J, K, L, M, IAF, IAL, NG, NN, ITAG, IGF, IGL

    ITAG = IVAL

    NG = IMVNG(1)
    J = NG + 2
    DO I = 2, NG
       IF (.NOT. LVAL) ITAG = I - 1
       NN = IMVNG(I)
       DO K = J, NN, 2
          IGF = IMVNG(K-1)
          IGL = IMVNG(K)
          DO L = IGF, IGL
             IMVGRP(L) = ITAG
             IAF = IGPBS(L) + 1
             IAL = IGPBS(L+1)
             DO M = IAF, IAL
                IMVATM(M) = ITAG
             ENDDO
          ENDDO
       ENDDO
       J = NN + 2
    ENDDO

    RETURN
  END SUBROUTINE TAGGRP

  SUBROUTINE GTFLMV(IAMC,NAMC,IMVNG,NATOMX)
    !
    !       Determine highest and lowest moving atoms.
    !       Aaron R. Dinner
    !
    INTEGER IAMC, NAMC, IMVNG(*), NATOMX
    !
    INTEGER I, J, IAF, IAL, NG, NN

    !       Get highest and lowest moving atom indices
    IAMC = NATOMX + 1
    NAMC = 0

    NG = IMVNG(1)
    J = NG + 1
    DO I = 2, NG
       NN = IMVNG(I)
       IAF = IMVNG(J)
       IAL = IMVNG(NN)
       IF (IAF .LT. IAMC) IAMC = IAF
       IF (IAL .GT. NAMC) NAMC = IAL
       J = NN + 1
    ENDDO

    RETURN
  END SUBROUTINE GTFLMV

  SUBROUTINE MCNBND(ENBV,ENBE,IAMC,NAMC,IMVNG,IMVATM,IMVGRP, &
       LFIRST,BNBND,INBLXG, &
       JNXG,MCBLO,MCBLGP,LGROUP,IGMC,NGMC, &
       MCA14P,LACE,X,Y,Z &
#if KEY_PERT==1
       ,MCBLOR,MCBLOP,BNBNDR,BNBNDP &   
#endif
       )
    !
    !       Non-bonded energy contribution of moving atoms in MC
    !
    !       Works by setting up a miniature non-bonded list which includes
    !       only terms expected to change and then sends this miniature list
    !       to standard ENBOND routine.
    !
    !       The miniature list is returned to be used by other non-bond routines
    !
    !       If LFIRST is .TRUE.,  sets up the miniature list.
    !       If LFIRST is .FALSE., uses the miniature list without checking it.
    !
    !       Aaron R. Dinner
    !
    use number
    use deriv
    use econtmod
    use enbond_mod
    use energym
#if KEY_PERT==1
    use epert_mod                     
#endif
    use param
#if KEY_PERT==1
    use pert                      
#endif
    use psf
    use fast
    use stream

    INTEGER IMVATM(:), IMVGRP(:), IMVNG(:)
    INTEGER IGMC, NGMC, IAMC, NAMC
    type(nonbondDataStructure) :: BNBND
    INTEGER INBLXG(:), JNXG(:)
    type(chm_iptr),dimension(:) :: MCA14P, MCBLO, MCBLGP
#if KEY_PERT==1
    type(chm_iptr),dimension(:) :: MCBLOR, MCBLOP
    type(nonbondDataStructure) :: BNBNDR, BNBNDP
#endif 
    real(chm_real)  ENBV, ENBE
    real(chm_real)  X(*), Y(*), Z(*)
    LOGICAL LGROUP, LFIRST, LACE
    !
    INTEGER N, I, J, K, KS, NB, NN, NG
    INTEGER IAF, IAL, ISTRT
#if KEY_PERT==1
    real(chm_real)  ENBVR, ENBER, ENBVP, ENBEP 
#endif
    LOGICAL LELECX
#if KEY_ACE==1
    LOGICAL LACTMP 
#endif

#if KEY_ACE==1
    LELECX = QETERM(ELEC) .AND. .NOT. LACE
    !       Turn ACE off and handle later in MCACEE
    LACTMP = LACE
    LACE = .FALSE.
#else /**/
    LELECX = QETERM(ELEC)
#endif 

    !       Setup a miniature non-bonded list
    IF (LFIRST) THEN
       IF (LGROUP) THEN
          CALL MKNLST(BNBND%INBLOG,BNBND%JNBG, IMVGRP, MCBLGP, IGMC,NGMC)
          !           Setup a miniature exclusion list
          CALL STUP14(BNBND%IBLO14,BNBND%INB14, MCA14P, IMVATM,IAMC,NAMC)
       ELSE
          CALL MKNLST(BNBND%INBLO,BNBND%JNB,IMVATM, MCBLO, IAMC,NAMC)
#if KEY_PERT==1
          IF (QPERT) THEN
             CALL MKNLST(BNBNDR%INBLO,BNBNDR%JNB,IMVATM, MCBLOR, IAMC,NAMC)
             CALL MKNLST(BNBNDP%INBLO,BNBNDP%JNB,IMVATM, MCBLOP, IAMC,NAMC)
          ENDIF
#endif 
       ENDIF
    ENDIF

    IF (LGROUP) THEN
       ISTRT = IGMC
    ELSE
       ISTRT = IAMC
    ENDIF

    CALL ENBOND(ENBV,ENBE,BNBND,ISTRT,NAMC, &
         CG,RSCLF,NGMC,IGPBS,IGPTYP,IAC,IACNB, &
         DX,DY,DZ,X,Y,Z,.FALSE.,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/), &
         .FALSE.,QETERM(VDW),LELECX, &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),       & 
#endif
         QETERM(ST2),NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
         ,.FALSE., ONE, WCA      & 
#endif
#if KEY_LJPME==1
         ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
         )

#if KEY_PERT==1
    IF (QPERT) THEN
       CALL ENBOND(ENBVR,ENBER,BNBNDR,ISTRT,NAMC, &
            PPCG,PPRSCLF, &
            NGMC,PPIGPBS,PPIGPTP, &
            PPIAC,PPIACNB, &
            DX,DY,DZ,X,Y,Z,.FALSE.,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/), &
            .FALSE.,QETPRT(VDW),QETPRT(ELEC), &
#if KEY_FLUCQ==1
            .FALSE.,(/ZERO/),       & 
#endif
            QETPRT(ST2),NST2P,ETPRTM(ST2),.FALSE. &
#if KEY_WCA==1
            ,LSOFTCORE0, SCCUTR0,PPWCA   & 
#endif
#if KEY_LJPME==1
            ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
            )

       CALL ENBOND(ENBVP,ENBEP,BNBNDP,ISTRT,NAMC, &
            CG,RSCLF,NGMC,IGPBS,IGPTYP,IAC,IACNB, &
            DX,DY,DZ,X,Y,Z,.FALSE.,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/), &
            .FALSE.,QETERM(VDW),LELECX, &
#if KEY_FLUCQ==1
            .FALSE.,(/ZERO/),       & 
#endif
            QETERM(ST2),NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
            ,LSOFTCORE1, SCCUTR1, WCA    & 
#endif
#if KEY_LJPME==1
            ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
            )

       ENBV = ENBV + ENBVR*LAMDAM + ENBVP*LAMDA
       ENBE = ENBE + ENBER*LAMDAM + ENBEP*LAMDA
    ENDIF
#endif 

#if KEY_ACE==1
    LACE = LACTMP
#endif 

    RETURN
  END SUBROUTINE MCNBND

  SUBROUTINE MCNOEE(ENOE,CNSARR,X,Y,Z)
    !
    !       Computes the MC energy associated with NOE constraints
    !
    !       Aaron R. Dinner
    !       99-07-13
    !
    use deriv
    use noem
    use number
    INTEGER CNSARR(:)
    real(chm_real)  ENOE, X(*), Y(*), Z(*)
    !
    INTEGER I, J, N, IAF, IAL
    real(chm_real)  EN

    ENOE = ZERO

    N = CNSARR(1)
    DO I = 3, N, 2
       IAF = CNSARR(I-1)
       IAL = CNSARR(I)
       DO J = IAF, IAL
          !           Trick NOECNS to thinking there is only the constraint in which
          !           we are interested.
          CALL NOECNS(EN,DX,DY,DZ,X,Y,Z,.FALSE.,ZERO,1,NOESCA, &
               NOEIPT(J),NOEJPT(J),NOEINM(J),NOEJNM(J),NOELIS, &
               NOEEXP,NOERMN(J),NOEKMN(J),NOERMX(J),NOEKMX(J), &
               NOEFMX(J),NOETCN(J),NOEAVE(J),NOEMIN(J), &
               ZERO,0,.FALSE. &
               !JH
               ,NOERSW(J),NOESEX(J),NOERAM(J) &
#if KEY_PNOE==1
               ,IsPNOE,C0X,C0Y,C0Z      & 
#endif
#if KEY_PNOE==1
               , MVPNOE,OC0X,OC0Y,OC0Z   & 
#endif
#if KEY_PNOE==1
               ,TC0X,TC0Y,TC0Z   & 
#endif
#if KEY_PNOE==1
               , NMPNOE, IMPNOE          & 
#endif
               )
          ENOE = ENOE + EN
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE MCNOEE

  SUBROUTINE GTSYMNB(INBLO,JNB,MCNBLP,NOLD,NATOM,IA1,IAN,QPRIM)
    !
    !       Creates a symmetric non-bonded list for faster MC calculations.
    !
    !       Aaron R. Dinner
    !
    use dimens_fcm
    use memory

    INTEGER INBLO(:), JNB(:)
    type(iptr_ptr) :: MCNBLP
    INTEGER NOLD,NATOM, IA1, IAN
    LOGICAL QPRIM

    INTEGER I, J, N
    integer,pointer,dimension(:) :: TEMPP

    IF (associated(MCNBLP%A)) THEN
       DO I = 1, NOLD
          TEMPP => MCNBLP%A(I)%A
          N = TEMPP(1)
          call chmdealloc('mcener.src','GTSYMNB','TEMPP',N,intgp=TEMPP)
       ENDDO
       deallocate(MCNBLP%A)
    ENDIF
    allocate(MCNBLP%A(NATOM))

    CALL CTMCNB(INBLO,JNB, MCNBLP%A, NATOM,IA1,IAN,QPRIM)

    CALL FLMCNB(INBLO, JNB, MCNBLP%A, NATOM, &
         IA1,IAN,QPRIM)

    RETURN
  END SUBROUTINE GTSYMNB

  SUBROUTINE CTMCNB(INBLO,JNB,MCBLO,NATOM,IA1,IAN,QPRIM)
    !
    !       Counts elements for the MC symmetric non-bonded list
    !
    !       Aaron R. Dinner
    !
    INTEGER INBLO(:), JNB(:)
    type(chm_iptr) :: MCBLO(:)
    INTEGER NATOM, IA1, IAN
    LOGICAL QPRIM

    INTEGER I, J, K, N, IFIRST, ILAST

    DO I = 1, NATOM
       MCBLO(I)%LEN = 0
    ENDDO

    IFIRST=1
    DO I=IA1,IAN
       ILAST=INBLO(I)
       DO J=IFIRST,ILAST
          K=JNB(J)
          IF(K.LT.0) K=-K
          MCBLO(K)%LEN = MCBLO(K)%LEN + 1
          IF (QPRIM .AND. (I.NE.K)) MCBLO(I)%LEN = MCBLO(I)%LEN + 1
       ENDDO
       IFIRST = ILAST + 1
    ENDDO

    RETURN
  END SUBROUTINE CTMCNB

  SUBROUTINE FLMCNB(INBLO,JNB,MCBLO,NATOM,IA1,IAN,QPRIM)
    !
    !       Fills elements for the MC symmetric non-bonded list
    !
    !       Aaron R. Dinner
    !
    use mc
    use memory

    INTEGER INBLO(:), JNB(:)
    type(chm_iptr) :: MCBLO(:)
    INTEGER NATOM, IA1, IAN
    LOGICAL QPRIM

    integer,dimension(NATOM) :: MCCTR
    INTEGER I, J, K, N, IFIRST, ILAST, IS, KS
    integer,pointer,dimension(:) :: TEMPP
#if KEY_GCMC==1
    integer,parameter :: NBBUFF = 100
#else /**/
    integer,parameter :: NBBUFF = 0
#endif 

    DO I = 1, NATOM
       N = MCBLO(I)%LEN + 2 + NBBUFF
       call chmalloc('mcener.src','FLMCNB','TEMPP',N,intgp=TEMPP)
       TEMPP(1) = N
       TEMPP(2) = MCBLO(I)%LEN + 2
       MCBLO(I)%A => TEMPP
       MCCTR(I) = 2
    ENDDO

    IFIRST=1
    DO I=IA1,IAN
       ILAST=INBLO(I)
       DO J=IFIRST,ILAST

          K=JNB(J)
          IS = ISIGN(I,K)
          KS = K
          IF(K.LT.0) K  = -K

          MCCTR(K) =  MCCTR(K) + 1
          MCBLO(K)%A(MCCTR(K)) = IS

          IF (QPRIM .AND. (I.NE.K)) THEN
             !           IF (QPRIM) THEN
             MCCTR(I) =  MCCTR(I) + 1
             MCBLO(I)%A(MCCTR(I)) = KS
          ENDIF

       ENDDO
       IFIRST = ILAST + 1
    ENDDO

    RETURN
  END SUBROUTINE FLMCNB

  SUBROUTINE MKNLST(INBLOG,JNBG,IMVGR,MCBLG,IGMC,NGMC)
    !
    !       Setup the group based non-bonded lists between IGMC and NGMC
    !
    !       Aaron R. Dinner
    !
    INTEGER IMVGR(*)
    type(chm_iptr) :: MCBLG(:)
    INTEGER INBLOG(:), JNBG(:)
    INTEGER IGMC, NGMC
    !
    INTEGER I,J,K,L,NB,N,KS

    !       Fill the list
    N = 0
    IF (IGMC .GT. 1) INBLOG(IGMC-1) = N
    DO L = IGMC, NGMC
       IF (IMVGR(L) .GT. 0) THEN
          NB = MCBLG(L)%A(2)
          DO J = 3, NB
             K = MCBLG(L)%A(J)
             KS = K
             IF (K .LT. 0) K = -K
             IF ((IMVGR(L).GT.IMVGR(K)).OR.(K.EQ.L)) THEN
                N = N + 1
                JNBG(N) = KS
             ENDIF
          ENDDO
       ENDIF
       INBLOG(L) = N
    ENDDO
    RETURN
  END SUBROUTINE MKNLST

  SUBROUTINE STUP14(IBLO14,INB14,MCA14,IMVATM,IAMC,NAMC)
    !
    !       Sets up the atom-exclusion list for group based and/or
    !       ACE calculations in MC.
    !
    !       Aaron R. Dinner
    !
    type(chm_iptr) :: MCA14(:)
    INTEGER IBLO14(:), INB14(:), IMVATM(:)
    INTEGER IAMC, NAMC
    !
    INTEGER I, J, K, KS, NB, N14

    !       Set up the exclusion list between IAMC-1 and NAMC
    N14 = 0
    IF (IAMC.GT.1) IBLO14(IAMC-1) = N14
    DO I = IAMC, NAMC
       IF (IMVATM(I) .GT. 0) THEN

          NB = MCA14(I)%A(2)
          !           For each non-bond exclusion partner
          DO J = 3, NB
             K = MCA14(I)%A(J)
             KS = K
             IF (K .LT. 0) K = -K
             IF (IMVATM(I) .GE. IMVATM(K)) THEN
                N14 = N14 + 1
                INB14(N14) = KS
             ENDIF
          ENDDO

       ENDIF
       IBLO14(I) = N14
    ENDDO
    RETURN
  END SUBROUTINE STUP14

  SUBROUTINE MCUPDT(ETOT,MCBLOP,MCBLGP,MCIMLP,MCIMGP, &
       MCATTP,ISTEP,ICYCLE,INBFMC,IEFRQ, &
       IMGFMC,QHEUR,X,Y,Z,WMAIN,DX,DY,DZ, &
       NATOM,NGRP,BNBND,BIMAG, &
       BDUMMY,BDIMMY, &
       IA2GP,IGPBS,MCGTTP, &
#if KEY_ACE==1
       RSYS,ESMAX,FACT1, &  
#endif
#if KEY_ACE==1
       FACT2,FACT2H,ETURN,ETURNH,MEREF,MEREFH,CSWIT, & 
#endif
       RVOLMC,TKELV &
#if KEY_GCMC==1
       ,GCMCON,LGCMC,NMVTYP,MVTYPE,QGCGRD,NRGRID, &  
#endif
#if KEY_GCMC==1
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &  
#endif
#if KEY_GCMC==1
       NPSUM,GCSUM,GC2SUM,NGCIN,NIDXGC,QGCSPH, &  
#endif
#if KEY_GCMC==1
       IDXGCP,IMVNGP,NGCTRY,GCBLKR, NGRDBK, &  
#endif
#if KEY_GCMC==1
       NGDTOT, NGRDCV, GRDSIZ,NGRDX,NGRDY,NGRDZ &  
#endif
#if KEY_PERT==1
       ,MCBLORP,MCBLOPP,BDUMMYR,BDUMMYP,BNBNDR,BNBNDP &   
#endif
#if KEY_PERT==1
       ,MCIMLRP,MCIMLPP,BDIMMYR,BDIMMYP,BIMAGP,BIMAGR &   
#endif
       )
    !
    !         Update the non-bonded list, the image list, and the total energy.
    !
    !         Aaron R. Dinner
    !
    use consta
    use number
    !
    use ace_module
    use block_ltm
    use econtmod
    use energym
    use image
    use imgup
    use inbnd
    use pert
    use stream
    use datstr
    use memory
#if KEY_ACE==1
    use mcace, only: acupdt  
#endif
    use mccent
    use mcimg, only: gtsim, gtsimg
#if KEY_GCMC==1
    use mcmvgcmc  
#endif
    use mcmvutil, only: cntall
    use prssre

    type(iptr_ptr) :: MCBLOP, MCBLGP, MCIMLP, MCIMGP, MCATTP
    INTEGER ISTEP, INBFMC, IEFRQ, IMGFMC
    INTEGER NATOM, NGRP, ICYCLE
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG
    type(nonbondDataStructure) BDUMMY
    type(imageDataStructure) BDIMMY

    INTEGER IA2GP(:), IGPBS(*)
    type(iptr_ptr) :: MCGTTP
#if KEY_ACE==1
    real(chm_real)  RSYS, ESMAX, FACT1
    LOGICAL CSWIT
    !       Added for ACE2
    real(chm_real)  FACT2, FACT2H, ETURN, ETURNH, MEREF, MEREFH
#endif 
    real(chm_real) :: X(:), Y(:), Z(:), WMAIN(*), DX(:), DY(:), DZ(:)
    real(chm_real)  ETOT, RSTEP, RVOLMC, TKELV
    LOGICAL QHEUR
#if KEY_GCMC==1
    INTEGER NGCIN(*),NIDXGC(*)
    type(chm_iptr) :: IDXGCP(:)
    type(iptr_ptr) :: IMVNGP(:)
    INTEGER NMVTYP, MVTYPE(*), NPSUM(*)
    INTEGER NGCTRY, NGRDBK, NGDTOT
    INTEGER NRGRID,NGRDCV, NGRDX,NGRDY,NGRDZ
    real(chm_real)  GCSUM(*),GC2SUM(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    real(chm_real)  GRDSIZ
    LOGICAL LGCMC, GCMCON(:), GCBLKR(:), QGCSPH, QGCGRD
#endif 
#if KEY_PERT==1
    type(iptr_ptr) :: MCBLORP, MCBLOPP
    type(nonbondDataStructure) BNBNDR
    type(nonbondDataStructure) BNBNDP
    type(nonbondDataStructure) BDUMMYR
    type(nonbondDataStructure) BDUMMYP

    type(iptr_ptr) :: MCIMLRP,MCIMLPP
    type(imageDataStructure) BDIMMYR
    type(imageDataStructure) BDIMMYP
    type(imageDataStructure) BIMAGP
    type(imageDataStructure) BIMAGR
#endif 
    !
    INTEGER I
#if KEY_GCMC==1
    INTEGER NIMOLD     
#endif
    LOGICAL QNONB, QIMUP, QIEUP

    !       Check if the non-bond list should be updated
    IF (INBFMC .GT. 0) THEN
       QNONB = (MOD(ISTEP,INBFMC) .EQ. 0)
    ELSE
       !         The next line deals correctly with INBFMC = 0 if QHEUR
       !         is initialized to FALSE
       QNONB = QHEUR
    ENDIF

    !       Check if the image list should be updated
    QIMUP = .FALSE.
    IF (NTRANS .GT. 0) THEN
       IF (IMGFMC .GT. 0) THEN
          QIMUP = (MOD(ISTEP,IMGFMC) .EQ. 0)
       ELSE IF (IMGFMC .LT. 0) THEN
          QIMUP = QHEUR
       ENDIF
    ENDIF
#if KEY_GCMC==1
    IF (LGROUP) THEN
       NIMOLD = NIMGRP
    ELSE
       NIMOLD = NATIM
    ENDIF
#endif 

    !       Check if the total energy should be evaluated
    IF (IEFRQ .GT. 0) THEN
       QIEUP = (MOD(ISTEP,IEFRQ) .EQ. 0)
    ELSE IF (IEFRQ .LT. 0) THEN
       QIEUP = QHEUR
    ELSE
       QIEUP = .FALSE.
    ENDIF

    IF (QIMUP) THEN
       !         Free the list that holds the group centers
       IF (LGROUP) THEN
          IF (ISTEP .GT. 0) THEN
             call mc_dealloc_centers('mcener.src','MCUPDT',NIMGRP)
          ENDIF
       ENDIF
       CALL UPIMAG0(X, Y, Z, WMAIN, 0)
       !         Create a primary->image list
       CALL GTSIM(NTRANS,BIMAG%IMATPT, &
            BIMAG%IMATTR, MCATTP, &
            NATOM+1,NATOM)
       IF (LGROUP) THEN
          CALL GTSIMG(BIMAG%IMATTR, MCGTTP, &
               IA2GP,IGPBS,NGRP,NIMGRP)

          !           Allocate space for and get the group centers
          call mc_alloc_centers('mcener.src','MCUPDT',NIMGRP)

          CALL CNTALL(NIMGRP,X,Y,Z,XCENT,YCENT, &
               ZCENT,QCENT)
       ENDIF
       !         Update GCMCON flags
#if KEY_GCMC==1
       CALL GCUPIM(GCMCON, MCATTP%A, NATOM) 
#endif
    ENDIF

    !       Force a non-bond list update if the image list changes
    IF (QNONB .OR. QIMUP) THEN

       CALL NBONDS(X,Y,Z,BNBND,BIMAG)

       CALL DUPLDT_nbond(BDUMMY,BNBND)
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL DUPLDT_nbond(BDUMMYR,BNBNDR)
          CALL DUPLDT_nbond(BDUMMYP,BNBNDP)
       ENDIF
#endif 
       !         Regenerate the symmetric non-bond list
       !         Primary
       IF (LGROUP) THEN
          CALL GTSYMNB(BNBND%INBLOG, &
               BNBND%JNBG,MCBLGP,NGRP,NGRP,1,NGRP,.TRUE.)
       ELSE
          CALL GTSYMNB(BNBND%INBLO,BNBND%JNB, &
               MCBLOP,NATOM,NATOM,1,NATOM,.TRUE.)
#if KEY_PERT==1
          IF (QPERT) THEN
             CALL GTSYMNB(BNBNDR%INBLO,BNBNDR%JNB, &
                  MCBLORP,NATOM,NATOM,1,NATOM,.TRUE.)
             CALL GTSYMNB(BNBNDP%INBLO,BNBNDP%JNB, &
                  MCBLOPP,NATOM,NATOM,1,NATOM,.TRUE.)
          ENDIF
#endif 
#if KEY_GCMC==1
          !           Remove non-active atoms from updated non-bonded list.
          DO I = 1, NATOM
             CALL GCUPNB(MCBLOP%A(I)%A, I)
#if KEY_PERT==1 /*pert_gcmc*/
             IF (QPERT) THEN
                CALL GCUPNB(MCBLORP%A(I)%A, I)
                CALL GCUPNB(MCBLOPP%A(I)%A, I)
             ENDIF
#endif /*   (pert_gcmc)*/
          ENDDO
#endif 
       ENDIF
       !         Images
       IF (NTRANS .GT. 0) THEN
#if KEY_GCMC==1
          IF (LGCMC) THEN
             !             Symmetrize the image non-bonded list if GCMC
             IF (LGROUP) THEN
                CALL GTSYMNB(BIMAG%IMBLOG,BIMAG%IMJNBG, &
                     MCIMGP,NIMOLD,NIMGRP,NGRP+1,NIMGRP,.TRUE.)
             ELSE
                CALL GTSYMNB(BIMAG%IMBLO,BIMAG%IMJNB, &
                     MCIMLP,NIMOLD,NATIM,NATOM+1,NATIM,.TRUE.)
#if KEY_PERT==1 /*pert_gcmc*/
                IF (QPERT) THEN
                   CALL GTSYMNB(BIMAGR%IMBLO,BIMAGR%IMJNB, &
                        MCIMLRP,NIMOLD,NATIM,NATOM+1,NATIM,.TRUE.)
                   CALL GTSYMNB(BIMAGP%IMBLO,BIMAGP%IMJNB, &
                        MCIMLPP,NIMOLD,NATIM,NATOM+1,NATIM,.TRUE.)
                ENDIF
#endif /* (pert_gcmc)*/
             ENDIF
             DO I = 1, NATIM
                CALL GCUPNB(MCIMLP%A(I)%A, I)
#if KEY_PERT==1 /*pert_gcmc*/
                IF (QPERT) THEN
                   CALL GCUPNB(MCIMLRP%A(I)%A, I)
                   CALL GCUPNB(MCIMLPP%A(I)%A, I)
                ENDIF
#endif /* (pert_gcmc)*/
             ENDDO

          ELSE
#endif 
             IF (LGROUP) THEN
                CALL GTSYMNB(BIMAG%IMBLOG,BIMAG%IMJNBG, &
                     MCIMGP,NGRP,NGRP,NGRP+1,NIMGRP,.FALSE.)
             ELSE
                CALL GTSYMNB(BIMAG%IMBLO,BIMAG%IMJNB, &
                     MCIMLP,NATOM,NATOM,NATOM+1,NATIM,.FALSE.)
#if KEY_PERT==1
                IF (QPERT) THEN
                   CALL GTSYMNB(BIMAGR%IMBLO,BIMAGR%IMJNB, &
                        MCIMLRP,NATOM,NATOM,NATOM+1,NATIM,.FALSE.)
                   CALL GTSYMNB(BIMAGP%IMBLO,BIMAGP%IMJNB, &
                        MCIMLPP,NATOM,NATOM,NATOM+1,NATIM,.FALSE.)
                ENDIF
#endif 
             ENDIF

#if KEY_GCMC==1
          ENDIF 
#endif

          CALL DUPLDT_image(BDIMMY,BIMAG)
#if KEY_PERT==1
          IF (QPERT) THEN
             CALL DUPLDT_image(BDIMMYR,BIMAGR)
             CALL DUPLDT_image(BDIMMYP,BIMAGP)
          ENDIF
#endif 
       ENDIF
#if KEY_ACE==1
       IF (LACE) THEN
          CALL ACUPDT(RSYS,ESMAX, &
               FACT1,X,Y,Z,NATOM,BNBND,FACT2,FACT2H,ETURN, &
               ETURNH,MEREF,MEREFH,CSWIT)
       ENDIF
#endif 
    ENDIF

#if KEY_GCMC==1
    IF (QIMUP .OR. QIEUP .OR. (ISTEP .EQ. 0)) THEN  ! ad050629
       !         Update additional counters for GCMC
       CALL GCMCUP(NMVTYP,MVTYPE,QGCGRD,NRGRID,GCMCON, &
            XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
            NATOM,NATIM,NTRANS,LIMALL,BIMAG%IMATTR,NPSUM,GCSUM, &
            GC2SUM,NGCIN,NIDXGC,QGCSPH,IDXGCP,IMVNGP,NGCTRY, &
            GCBLKR,NGCBLK,NGRDBK,LSTGRD,LSTIND,NGDTOT, &
            NGRDCV,GRDBLK,GRDSIZ,NGRDX,NGRDY,NGRDZ,X,Y,Z)
    ENDIF
#endif 

    !       Check the energy
    !       Errors can arise from changes to the non-bond and image lists
    IF (QIEUP) THEN

#if KEY_BLOCK==1
       NOFORC = .FALSE.  
#endif
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL EPERT(X,Y,Z,DX,DY,DZ,.FALSE.,ECONT,0,0,.FALSE.,0)
       ELSE
#endif 
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
#if KEY_PERT==1
       ENDIF 
#endif

       !         Calculate the virial
       CALL PRSEXT
       IF (XDIM .EQ. 0) THEN
          EPROP(VOLUME) = RVOLMC
       ELSE
          RVOLMC = EPROP(VOLUME)
       ENDIF
       IF (EPROP(VOLUME) .GT. ZERO) THEN
          EPROP(PRESSI) = PATMOS / ( THREE * EPROP(VOLUME) ) * &
               ( KBOLTZ * 3 * NATOM * TKELV + &
               EPRESS(VIXX) + EPRESS(VIYY) + EPRESS(VIZZ) )
       ELSE
          EPROP(PRESSI) = ZERO
       ENDIF
#if KEY_BLOCK==1
       NOFORC = .TRUE.  
#endif

       IF (ISTEP .EQ. 0) THEN
          ETOT = EPROP(EPOT)
          ICYCLE = 0
       ELSE
          EOLD = ETOT
          ICYCLE = ICYCLE + 1
       ENDIF
       !         Time is divided by 1000 to match trajectory write
       RSTEP=ISTEP/1000.0
       IF (PRNLEV .GE. 1) CALL PRINTE(OUTU, EPROP, ETERM, &
            'MC E','ENR',.TRUE.,ICYCLE,RSTEP,RSTEP,.TRUE.)
       !         One could add a check here for too large an E change
       ETOT = EPROP(EPOT)
    ENDIF

    RETURN
  END SUBROUTINE MCUPDT

  SUBROUTINE FRUPDT(MCBLOP,MCBLGP,MCIMLP,MCIMGP,MCATTP, &
       BDUMMY,BDIMMY,NATOM,NGRP, &
       MCGTTP &
#if KEY_GCMC==1
       ,LGCMC        & 
#endif
#if KEY_PERT==1
       ,MCBLORP,MCBLOPP,BDUMMYR,BDUMMYP  & 
#endif
#if KEY_PERT==1
       ,MCIMLRP,MCIMLPP,BDIMMYR,BDIMMYP  & 
#endif
       )
    !
    !         Free the MC non-bonded lists allocated in MCUPDT
    !
    !         Aaron R. Dinner
    !
    use inbnd
    use image
    use pert
    use datstr
    use memory
    use mccent

    INTEGER NATOM, NGRP
    type(iptr_ptr) :: MCBLOP, MCBLGP, MCIMLP, MCIMGP, MCATTP, MCGTTP

    type(nonbondDataStructure) BDUMMY
    type(imageDataStructure) BDIMMY
#if KEY_PERT==1 /*pert_decl*/
    type(iptr_ptr) :: MCBLORP, MCBLOPP, MCIMLRP, MCIMLPP
    type(nonbondDataStructure) BDUMMYR
    type(nonbondDataStructure) BDUMMYP
    type(imageDataStructure) BDIMMYR
    type(imageDataStructure) BDIMMYP
#endif /* (pert_decl)*/

#if KEY_GCMC==1
    LOGICAL LGCMC    
#endif

    CALL FREEDT_nbond(BDUMMY)
#if KEY_PERT==1
    IF (QPERT) THEN
       CALL FREEDT_nbond(BDUMMYR)
       CALL FREEDT_nbond(BDUMMYP)
    ENDIF
#endif 
    IF (LGROUP) THEN
       CALL FRSYMNB(MCBLGP,NGRP)
    ELSE
       CALL FRSYMNB(MCBLOP,NATOM)
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL FRSYMNB(MCBLORP,NATOM)
          CALL FRSYMNB(MCBLOPP,NATOM)
       ENDIF
#endif 
    ENDIF

    IF (NTRANS .GT. 0) THEN
       CALL FRSYMNB(MCATTP,NATOM)
       IF (LGROUP) THEN
#if KEY_GCMC==1
          IF (LGCMC) THEN
             CALL FRSYMNB(MCIMGP,NIMGRP)
          ELSE
#endif 
             CALL FRSYMNB(MCIMGP,NGRP)
#if KEY_GCMC==1
          ENDIF 
#endif
          CALL FRSYMNB(MCGTTP,NGRP)
          call mc_dealloc_centers('mcener.src', 'FRUPDT', NIMGRP)
       ELSE
#if KEY_GCMC==1
          IF (LGCMC) THEN
             CALL FRSYMNB(MCIMLP,NATIM)
#if KEY_PERT==1 /*pert_gcmc*/
             IF (QPERT) THEN
                CALL FRSYMNB(MCIMLRP,NATIM)
                CALL FRSYMNB(MCIMLPP,NATIM)
             ENDIF
#endif /*   (pert_gcmc)*/
          ELSE
#endif 
             CALL FRSYMNB(MCIMLP,NATOM)
#if KEY_PERT==1
             IF (QPERT) THEN
                CALL FRSYMNB(MCIMLRP,NATIM)
                CALL FRSYMNB(MCIMLPP,NATIM)
             ENDIF
#endif 
#if KEY_GCMC==1
          ENDIF 
#endif
       ENDIF
       CALL FREEDT_image(BDIMMY)
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL FREEDT_image(BDIMMYR)
          CALL FREEDT_image(BDIMMYP)
       ENDIF
#endif 
    ELSE
       IF (LGROUP) THEN
          call mc_dealloc_centers('mcener.src', 'FRUPDT', NGRP)
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE FRUPDT

  SUBROUTINE FRSYMNB(MCNBLP,NATOM)
    !
    !       Frees a symmetric non-bonded list.
    !
    !       Aaron R. Dinner
    !
    use memory

    type(iptr_ptr) :: MCNBLP
    INTEGER NATOM
    INTEGER I, N
    integer,pointer,dimension(:) :: TEMPP

    IF (associated(MCNBLP%A)) THEN
       DO I = 1, NATOM
          TEMPP => MCNBLP%A(I)%A
          N = TEMPP(1)
          call chmdealloc('mcener.src','FRSYMNB','TEMPP',N,intgp=TEMPP)
       ENDDO
       deallocate(MCNBLP%A)
    ENDIF

    RETURN
  END SUBROUTINE FRSYMNB

#endif 
end module mce

