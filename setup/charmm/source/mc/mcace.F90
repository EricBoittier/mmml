module mcace
#if KEY_MC==1
#if KEY_ACE==1
  use chm_types
  use dimens_fcm
  implicit none
  
  type(chm_ptr),allocatable,dimension(:) :: S2A14P
  real(chm_real),allocatable,dimension(:) :: ESARP, ETMPP, ESR1P, ESR2P
  real(chm_real),allocatable,dimension(:) :: BSARP, CG2RP
  integer,allocatable,dimension(:) :: IBCHGP, IBCUTP, NOTADP
  
contains
  
  SUBROUTINE MCACEE(EEL,IFRSTA,NATOMX,ESMCFX,ESARR,BSARR,CG2ARR, &
       RSYS,NOTADD,MCBLO,MCA14,IBCHG,NACEBC,IBCUT, &
       NCUTBC, FACT2,FACT2H,ETURN,ETURNH,MEREF, &
       MEREFH,CSWIT)
    !
    !       A stripped down version of ENACE for MC in which the
    !       self energies are calculated by continuous update rather
    !       than from scratch each time.
    !
    !       Also, no force calculations!
    !
    !       Aaron R. Dinner, November 1998
    !       Based on ENACE by Michael Schaefer and Christian Bartels
    !
    use consta
    use dimens_fcm
    use number
    !
    use ace_module,only:esii,cgiacp,epsi,epss,khyd
    use block_fcm
    use coord
    use deriv
    use energym
    use inbnd
    use param
    use psf
    use stream
    !
    INTEGER IFRSTA,NATOMX
    INTEGER NOTADD(:), IBCHG(:), NACEBC, IBCUT(:), NCUTBC
    type(chm_iptr) :: MCBLO(:), MCA14(:)
    real(chm_real)  EEL
    LOGICAL CSWIT
    !       Added for ACE2
    real(chm_real)  FACT2, FACT2H, ETURN, ETURNH, MEREF, MEREFH
    !
    !       ESARR = atomic solvation energy,
    !       BSARR = solvation radius,
    !       CG2ARR= TAU * square of atomic charges.
    !
    real(chm_real)  ESMCFX(:), ESARR(:), BSARR(:), CG2ARR(:), RSYS
    !
    !       Local variables
    !
    INTEGER I
    real(chm_real)  TAU, FACT1, CFACT1, EEL0, EHYD, ECOUL
    real(chm_real)  ESMAX, RUL3, C2OFNB, C2ONNB

    EEL=ZERO
    EHYD=ZERO
    ECOUL=ZERO

    C2OFNB=CTOFNB*CTOFNB
    C2ONNB=CTONNB*CTONNB
    TAU=ONE/EPSI-ONE/EPSS

    IF (CSWIT .AND. (CTOFNB.GT.CTONNB)) RUL3=ONE/(C2OFNB-C2ONNB)**3

    FACT1=-CCELEC/TWO
    ESMAX=FACT1/RSYS

    !       In ENACE, this is the second loop (first is self energies)
    !       to calculate solvation radii and scale the self energies.

    DO I = 1, NACEBC
       CALL ACELP2(EHYD,EEL,IBCHG(I),BSARR,ESARR,ESMCFX,CG2ARR, &
            RSYS,ESMAX,FACT1,KHYD,ESII, &
            FACT2,FACT2H,ETURN,ETURNH,MEREF,MEREFH)
    ENDDO

    EEL0=EEL

    !       Calculate the screening energy.
    !       This loop involves all terms in which the BSARR have changed.
    !       It cannot be combined with the loop above since the BSARR must
    !       all be set up before entering the loop.
    !
    !       Do the calculation only for those BSARR changing by more than
    !       a cutoff.

    FACT1 = -CCELEC*TAU
    CFACT1 = CCELEC/EPSI

    DO I = 1, NCUTBC
       IF (CGIACP(IBCUT(I)).NE.ZERO) THEN
          !           SCLOOP loops over all bonded and non-bonded pairs of the atom
          CALL SCLOOP(EEL,ECOUL,NOTADD,IBCUT(I),MCBLO,MCA14,BSARR, &
               CGIACP,C2ONNB,C2OFNB,RUL3,FACT1,CSWIT, &
               S2A14P(IBCUT(I))%A, CFACT1,E14FAC,X,Y,Z)
          !           Mark so we do not double count pairs
          NOTADD(IBCUT(I)) = 2
       ENDIF
    ENDDO

    !       Clear NOTADD
    DO I = 1, NCUTBC
       NOTADD(IBCUT(I)) = 1
    ENDDO

    EEL=EEL + EHYD + ECOUL
    !
    RETURN
  END SUBROUTINE MCACEE

  SUBROUTINE SCLOOP(EEL,ECOUL,NOTADD,JV,MCBLO,MCA14,BSARR,CGIACE, &
       C2ONNB,C2OFNB,RUL3,FACT1,CSWIT,S2A14,CFACT1, &
       E14FAC,X,Y,Z)
    !
    !       Loops over all the atom pairs of atom jv
    !       to get its contribution to the screening energy
    !       Aaron R. Dinner, November 1998
    !
    use number
    use ace_module,only:dxyzpb

    INTEGER JV, NOTADD(:)
    type(chm_iptr) :: MCBLO(:), MCA14(:)
    real(chm_real)  EEL, ECOUL, BSARR(:), CGIACE(*),  &
         C2ONNB, C2OFNB, RUL3
    real(chm_real)  X(*), Y(*), Z(*), S2A14(*), CFACT1,E14FAC,FACT1
    LOGICAL CSWIT
    !
    INTEGER I, K, NB
    real(chm_real)  TXIJ, TYIJ, TZIJ, S2
    LOGICAL L14

    NB = MCBLO(JV)%A(2)
    !       For each non-bond partner
    DO I = 3, NB
       K = MCBLO(JV)%A(I)
       L14 = K .LT. 0
       IF (L14) K = -K
       IF (NOTADD(K).NE.2 .AND. CGIACE(K) .NE. ZERO) THEN
          CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
               X(K),Y(K),Z(K),X(JV),Y(JV),Z(JV))
          S2 = MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)
          IF (S2.LT.C2OFNB) THEN
             CALL GTESCR(EEL,K,JV,CSWIT,C2ONNB,C2OFNB,S2,RUL3,BSARR, &
                  CGIACE,FACT1,ECOUL,.TRUE.,L14,CFACT1,E14FAC)
          ENDIF
       ENDIF
    ENDDO

    !       Duplicate above structure to set up INBLO14
    NB = MCA14(JV)%A(2)
    !       For each non-bond exclusion partner
    DO I = 3, NB
       K = MCA14(JV)%A(I)
       !         Do not repeat 1-4
       IF (K .GT. 0) THEN
          IF (NOTADD(K).NE.2 .AND. CGIACE(K).NE.ZERO) THEN
             S2 = S2A14(I-2)
             IF (S2.LT.C2OFNB) THEN
                CALL GTESCR(EEL,K,JV,CSWIT,C2ONNB,C2OFNB,S2,RUL3,BSARR, &
                     CGIACE,FACT1,ZERO,.FALSE.,.FALSE.,CFACT1, &
                     E14FAC)
             ENDIF
          ENDIF
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE SCLOOP

  SUBROUTINE GTESCR(EEL,I,JVECT,CSWIT,C2ONNB,C2OFNB,S2,RUL3,BSARR, &
       CGIACE,FACT1,ECOUL,LCOUL,L14,CFACT1,E14FAC)
    !
    !       Adds the screening contribution for one atom pair.
    !       Also returns SW the switching function.
    !       Aaron R. Dinner, November 1998
    !
    use number

    INTEGER I, JVECT
    real(chm_real)  EEL, ECOUL, C2ONNB, C2OFNB, S2, RUL3
    real(chm_real)  BSARR(:), CGIACE(*), FACT1,CFACT1,E14FAC
    LOGICAL CSWIT, LCOUL, L14
    !
    real(chm_real) SW, CGIJ, EC
    real(chm_real) RIJL, RIJU, BIJ2, EXPO, FEXP, RIJ2, RIJ, E

    IF (CSWIT .AND. (S2 .GT. C2ONNB)) THEN
       RIJL=C2ONNB-S2
       RIJU=C2OFNB-S2
       SW  =RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
    ELSE
       SW  =ONE
    ENDIF

    BIJ2  = BSARR(I)*BSARR(JVECT)
    EXPO  = S2/(FOUR*BIJ2)
    FEXP  = EXP(-EXPO)
    RIJ2  = S2 + BIJ2*FEXP
    RIJ   = SQRT(RIJ2)
    CGIJ  = CGIACE(I)*CGIACE(JVECT)
    E     = FACT1*CGIJ/RIJ

    EEL   = EEL + E*SW

    !       ARD 02-06-14
    !       Added due to discrepancies with standard Coulomb potential
    IF (LCOUL) THEN
       EC = CFACT1*CGIJ/SQRT(S2)
       IF (L14) EC=EC*E14FAC
       ECOUL = ECOUL + EC*SW
    ENDIF

    RETURN
  END SUBROUTINE GTESCR

  SUBROUTINE ACUPDT(RSYS,ESMAX,FACT1, &
       X,Y,Z,NATOM,BNBND,FACT2,FACT2H,ETURN,ETURNH, &
       MEREF,MEREFH,CSWIT)
    !
    !       Updates the ACE self energies in MC
    !
    use dimens_fcm
    !
    use ace_module,only:sig2i,mue4,esii,ces1,ces2,khyd,esfix
    use inbnd
    use number
    INTEGER NATOM
    type(nonbondDataStructure) BNBND
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  RSYS, ESMAX, FACT1
    LOGICAL CSWIT
    !       Added for ACE2
    real(chm_real)  FACT2, FACT2H, ETURN, ETURNH, MEREF, MEREFH

    INTEGER I
    real(chm_real)  EDUMMY

    !       If the non-bond list has changed, update the ACE self energies
    ESARP(1:NATOM) = ESFIX(1:NATOM)
    CALL GTESLF(ESARP, BNBND%INBLO, BNBND%JNB, &
         1,NATOM,X,Y,Z,CES1,CES2,sig2i, &
         MUE4,CSWIT,CTOFNB*CTOFNB,CTONNB*CTONNB)
    !       Initialize the BSARR
    DO I = 1, NATOM
       CALL ACELP2(EDUMMY,EDUMMY,I, BSARP, ETMPP, &
            ESARP, CG2RP, RSYS,ESMAX,FACT1, &
            KHYD,ESII,FACT2,FACT2H, &
            ETURN,ETURNH,MEREF,MEREFH)
    ENDDO
    RETURN
  END SUBROUTINE ACUPDT

  SUBROUTINE GTESLF(ESARR,INBLO,JNB,IFRSTA,NATOMX,X,Y,Z,CES1,CES2, &
       SIG2I,MUE4,CSWIT,C2OFNB,C2ONNB)
    !
    !       Calculate the ACE self energy term for all atoms
    !
    !       Aaron R. Dinner, October 1998
    !       Based on the first loop in ENACE by Schaefer and Bartels
    !
    use number
    use psf
    use ace_module,only:dxyzpb
    implicit none

    INTEGER IFRSTA, NATOMX
    INTEGER INBLO(:), JNB(:)
    real(chm_real)  X(*), Y(*), Z(*), C2OFNB, C2ONNB
    real(chm_real)  ESARR(:), CES1(MAXATC,MAXATC), CES2(*)
    real(chm_real)  SIG2I(MAXATC,MAXATC), MUE4(MAXATC,MAXATC)
    LOGICAL CSWIT

    INTEGER I, J, JVECT, ISTART
    real(chm_real)  TXIJ, TYIJ, TZIJ, RIJL, RIJU, SW, S2
    real(chm_real)  RUL3, S

    !     check electrostatic switching function
    IF (CSWIT .AND. (C2OFNB .GT. C2ONNB)) RUL3=ONE/(C2OFNB-C2ONNB)**3

    IF (IFRSTA .GT. 1) THEN
       ISTART = INBLO(IFRSTA-1) + 1
    ELSE
       ISTART = 1
    ENDIF

    DO I = IFRSTA, NATOMX
       DO J = ISTART, INBLO(I)

          JVECT = JNB(J)
          IF (JVECT .LT. 0) JVECT = -JVECT

          !         Calculate the distance between i and j
          CALL DXYZPB(TXIJ,TYIJ,TZIJ, &
               X(I),Y(I),Z(I),X(JVECT),Y(JVECT),Z(JVECT))
          S2 = MAX(RSMALL,TXIJ*TXIJ+TYIJ*TYIJ+TZIJ*TZIJ)

          !         Set up the switching function
          IF (S2.LT.C2OFNB) THEN
             S = SQRT(S2)
             IF (CSWIT .AND. (S2 .GT. C2ONNB)) THEN
                RIJL = C2ONNB-S2
                RIJU = C2OFNB-S2
                SW   = RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
             ELSE
                SW   = ONE
             ENDIF

             !           skip if atom JVECT has zero volume:
             IF (CES2(IAC(JVECT)).GT.ZERO) CALL ESELFMC(ESARR(I), &
                  IAC(I),IAC(JVECT),S,S2,SW,CES1,CES2,SIG2I,MUE4)

             !           skip if atom I has zero volume:
             IF (CES2(IAC(I)).GT.ZERO) CALL ESELFMC(ESARR(JVECT), &
                  IAC(JVECT),IAC(I),S,S2,SW,CES1,CES2,SIG2I,MUE4)

          ENDIF

       ENDDO
       ISTART = INBLO(I) + 1
    ENDDO

    RETURN
  END SUBROUTINE GTESLF

  SUBROUTINE ACELP2(EHYD,EEL,I,BSARR,ESARR,ESMCFX,CG2ARR, &
       RSYS,ESMAX,FACT1,KHYD,ESII,FACT2,FACT2H, &
       ETURN,ETURNH,MEREF,MEREFH)
    !
    !       The calculations for the second loop of ACE
    !       Aaron R. Dinner, November 1998
    !
    use consta
    use ace_module,only:lace2,mxbsol,epsi,epss,esfix
    use inbnd
    use number
    use param
    use psf
    use stream
    INTEGER I
    real(chm_real)  EHYD, EEL
    real(chm_real)  BSARR(:), ESARR(:), ESMCFX(:), CG2ARR(:)
    real(chm_real)  KHYD(*), ESII(*)
    real(chm_real)  ESMAX, RSYS, FACT1
    !       Added for ACE2
    real(chm_real)  FACT2, FACT2H, ETURN, ETURNH, MEREF, MEREFH

    !
    real(chm_real)  FACT2X, ETURNX, MEREFX, DENOM, DELTB, SMARAD

    !       Must match value in ENACE
    PARAMETER (SMARAD=1.39D0)

    ESARR(I) = ESMCFX(I) + FACT1/CHRAD(IAC(I))+ESII(IAC(I))

    !       hydrophobic effect
    EHYD = EHYD + KHYD(IAC(I))*ESARR(I)

    IF (LACE2) THEN

       IF (CHRAD(IAC(I)) .GT. SMARAD) THEN
          ETURNX = ETURN
          MEREFX = MEREF
          FACT2X = FACT2
       ELSE
          ETURNX = ETURNH
          MEREFX = MEREFH
          FACT2X = FACT2H
       ENDIF

       If (ESARR(I).LE.ETURNX) THEN
          BSARR(I) = FACT1 / ESARR(I)
       ELSE
          DENOM = ESARR(I) + MEREFX
          DELTB = FACT2X / DENOM
          BSARR(I) = MXBSOL + DELTB
       ENDIF

       !         Back-calculate the ESARR for consistency
       ESARR(I) = FACT1 / BSARR(I)

    ELSE

       !         The following code is for ACE1
       IF(ESARR(I) .LE. ESMAX) THEN
          BSARR(I) = FACT1 / ESARR(I)
       ELSE
          BSARR(I) = RSYS*(TWO-ESARR(I)/ESMAX)
          IF(WRNLEV.GE.6) THEN
             WRITE(OUTU,10) I,ESARR(I)
10           FORMAT('ACELP2> WARNING, ATOMIC SOLVATION > ESMAX ', &
                  ' IATOM = ',I6,' DESELF = ',F12.3)
          ENDIF
       ENDIF

    ENDIF

    ESARR(I) = ESARR(I)*CG2ARR(I)

    EEL = EEL + ESARR(I)
    RETURN
  END SUBROUTINE ACELP2

  SUBROUTINE ESELFMC(ESI,IT,KT,R,R2,SW,CES1,CES2,SIG2I,MUE4)
    !
    !       A stripped down version of ACE ESELFIK for MC.
    !       No Forces and no BLOCK.
    !
    !       Aaron R. Dinner, October 1998
    !       Based on ESELFIK by Schaefer and Bartels
    !
    use consta
    use ace_module,only:
    use param
    !
    real(chm_real)  ESI,R,R2,SW
    INTEGER IT,KT
    !
    !
    real(chm_real) RHO,RHO2,FEXP,DENO,R4
    real(chm_real) EXPO,E, FAC1, FAC2
    real(chm_real)  CES1(MAXATC,MAXATC), CES2(*)
    real(chm_real)  SIG2I(MAXATC,MAXATC), MUE4(MAXATC,MAXATC)
    !
    R4   = R2*R2
    DENO = R4 + MUE4(IT,KT)
    RHO  = R2*R/DENO
    RHO2 = RHO*RHO
    EXPO = R2*SIG2I(IT,KT)
    FEXP = EXP(-EXPO)
    !
    FAC1 = CES1(IT,KT)*FEXP
    FAC2 = CES2(KT)*RHO2*RHO
    !       ESI  = ESI + CES1(IT,KT)*FEXP + CES2(KT)*RHO2*RHO2
    E    = FAC1 + FAC2*RHO
    !       SW accounts for the switching function
    ESI  = ESI + E * SW
    !
    RETURN
  END SUBROUTINE ESELFMC

  SUBROUTINE STUPBC(IBCHG,NACEBC,NOTADD,INBLO,JNB,IFRSTA,NATOMX)
    !
    !       Make a list of atoms for which the BSARR change.
    !       Aaron R. Dinner, November 1998
    !
    !       Modified significantly May 2002 due to c29 change to
    !       the ACE potential (bonded atom distances are treated
    !       as fixed at ideal lengths, so need not go through the
    !       exclusion list).
    !
    INTEGER IFRSTA, NATOMX, NACEBC
    INTEGER INBLO(:), JNB(:), NOTADD(:), IBCHG(:)

    INTEGER I, J, ISTART, JVECT

    NACEBC = 0

    IF (IFRSTA .GT. 1) THEN
       ISTART = INBLO(IFRSTA-1) + 1
    ELSE
       ISTART = 1
    ENDIF

    DO I = IFRSTA, NATOMX
       DO J = ISTART, INBLO(I)

          JVECT = JNB(J)
          IF (JVECT .LT. 0) JVECT = -JVECT

          IF (NOTADD(I) .EQ. 1) THEN
             NACEBC = NACEBC + 1
             IBCHG(NACEBC) = I
             NOTADD(I) = 0
          ENDIF

          IF (NOTADD(JVECT) .EQ. 1) THEN
             NACEBC = NACEBC + 1
             IBCHG(NACEBC) = JVECT
             NOTADD(JVECT) = 0
          ENDIF

       ENDDO
       ISTART = INBLO(I) + 1
    ENDDO

    DO I = 1, NACEBC
       NOTADD(IBCHG(I)) = 1
    ENDDO

    RETURN
  END SUBROUTINE STUPBC

  SUBROUTINE ADESLF(IBCUT,NCUTBC,ECUT,ESARR1,ESARR2, &
       IBCHG,NACEBC,MODE)
    !
    !       Add (or clear) the ACE self energy differences to the totals
    !
    !       MODE == 0 -> Add the differences
    !       MODE == 1 -> Add the differences and make a list of the
    !                    differences that are more than a cutoff
    !       MODE == 2 -> Clear the differences
    !
    !       Aaron R. Dinner, November 1998
    !
    use number

    INTEGER IBCUT(:), NCUTBC
    INTEGER IBCHG(:), NACEBC, MODE
    real(chm_real) ESARR1(:), ESARR2(:), ECUT
    !
    INTEGER I, J
    real(chm_real)  EDIFF

    IF (MODE .EQ. 0) THEN
       DO I = 1, NACEBC
          J = IBCHG(I)
          ESARP(J) = ESARP(J) + ESARR2(J) - ESARR1(J)
       ENDDO
    ELSE IF (MODE .EQ. 1) THEN
       NCUTBC = 0
       DO I = 1, NACEBC
          J = IBCHG(I)
          EDIFF = ESARR2(J) - ESARR1(J)
          ESARP(J) = ESARP(J) + EDIFF
          IF (ABS(EDIFF) .GE. ECUT) THEN
             NCUTBC = NCUTBC + 1
             IBCUT(NCUTBC) = J
          ENDIF
       ENDDO
    ELSE IF (MODE .EQ. 2) THEN
       DO I = 1, NACEBC
          J = IBCHG(I)
          ESARR1(J) = ZERO
          ESARR2(J) = ZERO
       ENDDO
    ELSE
       CALL WRNDIE(-5,'<ADESLF>','Internal Error:  Unknown MODE')
    ENDIF

    RETURN
  END SUBROUTINE ADESLF

  SUBROUTINE MCACIN(TAU,RSYS,FACT1,ESMAX, &
       INBL14,JNBL14,MCA14, CGSACE,SA14, &
       ETURN,ETURNH,MEREF,MEREFH,FACT2,FACT2H,CSWIT, &
       X,Y,Z)

    use dimens_fcm
    use consta
    use memory
    !
    use ace_module,only:tbsolv,tbsolh,lace,epsi,epss,mxbsol
    use inbnd
    use number
    use param
    use psf

    INTEGER NACEBC, NCUTBC
    INTEGER INBL14(:), JNBL14(:)
    type(chm_iptr) :: MCA14(:)
    real(chm_real)  RSYS, TAU, ESMAX, FACT1, X(*), Y(*), Z(*),  &
         CGSACE(*)
    real(chm_real)  SA14(*)
    LOGICAL CSWIT
    !       Added for ACE2
    real(chm_real)  ETURN, ETURNH, MEREF, MEREFH, FACT2, FACT2H
    !
    INTEGER I, J, K, L, NB, JVECT, ISTART
    real(chm_real),pointer,dimension(:) :: TEMPP
    real(chm_real)  TXIJ, TYIJ, TZIJ, S2

    IF (LACE) THEN
       !         Initialize the ACE ESARR structure for the self energies
       call chmalloc('mcace.src','MCACIN','ESARR',NATOM,crl=ESARP)
       call chmalloc('mcace.src','MCACIN','ETMPP',NATOM,crl=ETMPP)
       call chmalloc('mcace.src','MCACIN','ESR1P',NATOM,crl=ESR1P)
       call chmalloc('mcace.src','MCACIN','ESR2P',NATOM,crl=ESR2P)
       call chmalloc('mcace.src','MCACIN','BSARR',NATOM,crl=BSARP)
       call chmalloc('mcace.src','MCACIN','CG2RP',NATOM,crl=CG2RP)
       call chmalloc('mcace.src','MCACIN','IBCHGP',NATOM,intg=IBCHGP)
       call chmalloc('mcace.src','MCACIN','IBCUTP',NATOM,intg=IBCUTP)
       call chmalloc('mcace.src','MCACIN','NOTADP',NATOM,intg=NOTADP)
       TAU=ONE/EPSI-ONE/EPSS
       RSYS=ZERO
       DO I = 1, NATOM
          ESR1P(I) = ZERO
          ESR2P(I) = ZERO
          CG2RP(I) = TAU * CGSACE(I) * CGSACE(I)
          NOTADP(I) = 1
          RSYS=RSYS+EFVOL(IAC(I))
       ENDDO
       RSYS=(THREE*RSYS/(FOUR*PI))**(THIRD)
       FACT1=-CCELEC/TWO
       ESMAX=FACT1/RSYS
       !         Check electrostatic switching function
       CSWIT = .NOT. LSHFT .AND. .NOT. LFSWT

       !         Added for updated ACE1
       !         Save distances of the 1-2 and 1-3 atoms.  Space is allocated
       !         for 1-4 but not used so make reference to 1-2 and 1-3 easier.
       allocate(S2A14P(NATOM))
       DO I = 1, NATOM
          NB = MCA14(I)%A(2)
          call chmalloc('mcace.src','MCACIN','TEMPP',NB-2,crlp=TEMPP)
          S2A14P(I)%A => TEMPP
       ENDDO

       !         Symmetrize the SA14 list and square
       ISTART = 1
       DO I = 1, NATOM
          DO J = ISTART, INBL14(I)

             JVECT = JNBL14(J)
             IF (JVECT .GT. 0) THEN

                NB = MCA14(I)%A(2)
                TEMPP => S2A14P(I)%A
                DO L = 3, NB
                   K = MCA14(I)%A(L)
                   IF (K .EQ. JVECT) THEN
                      S2 = SA14(J)*SA14(J)
                      TEMPP(L-2) = S2
                   ENDIF
                ENDDO

                NB = MCA14(JVECT)%A(2)
                TEMPP => S2A14P(JVECT)%A
                DO L = 3, NB
                   K = MCA14(JVECT)%A(L)
                   IF (K .EQ. I) THEN
                      S2 = SA14(J)*SA14(J)
                      TEMPP(L-2) = S2
                   ENDIF
                ENDDO

             ENDIF

          ENDDO
          ISTART = INBL14(I) + 1
       ENDDO

       !         ACE2 initializations
       ETURN  =  FACT1 / TBSOLV
       MEREF  = -ETURN  * MXBSOL / TBSOLV
       FACT2  =  ONE - MXBSOL / TBSOLV
       FACT2  =  FACT1 * FACT2  * FACT2

       ETURNH =  FACT1 / TBSOLH
       MEREFH = -ETURNH * MXBSOL / TBSOLH
       FACT2H =  ONE - MXBSOL / TBSOLH
       FACT2H =  FACT1 * FACT2H * FACT2H

    ENDIF
    RETURN
  END SUBROUTINE MCACIN

  SUBROUTINE FREACE(NATOM, MCA14)
    !
    !       Free the arrays for ACE in MC
    !
    use memory

    INTEGER NATOM
    type(chm_iptr) :: MCA14(:)
    !
    INTEGER I, NB
    real(chm_real),pointer,dimension(:) :: TEMPP

    call chmdealloc('mcace.src','FREACE','ESARR',NATOM,crl=ESARP)
    call chmdealloc('mcace.src','FREACE','ETMPP',NATOM,crl=ETMPP)
    call chmdealloc('mcace.src','FREACE','ESR1P',NATOM,crl=ESR1P)
    call chmdealloc('mcace.src','FREACE','ESR2P',NATOM,crl=ESR2P)
    call chmdealloc('mcace.src','FREACE','BSARR',NATOM,crl=BSARP)
    call chmdealloc('mcace.src','FREACE','CG2RP',NATOM,crl=CG2RP)
    call chmdealloc('mcace.src','FREACE','IBCHGP',NATOM,intg=IBCHGP)
    call chmdealloc('mcace.src','FREACE','IBCUTP',NATOM,intg=IBCUTP)
    call chmdealloc('mcace.src','FREACE','NOTADP',NATOM,intg=NOTADP)

    DO I = 1, NATOM
       NB = MCA14(I)%A(2)
       TEMPP => S2A14P(I)%A
       call chmdealloc('mcace.src','FREACE','TEMPP',NB-2,crlp=TEMPP)
    ENDDO
    deallocate(S2A14P)
    RETURN
  END SUBROUTINE FREACE
#endif 
#endif 
end module mcace

