SUBROUTINE NULL_efast_MM

  RETURN
END SUBROUTINE NULL_efast_MM
!CHARMM Element source/mmff/efast_mm.src 1.1
!
! "Wrapper" routines for MMFF vector "bonded" energy evaluations.
! Loop over CPUs (if parallel; else NCPU=1), allocate needed arrays on
! the heap, and call the routine that does the energy calculation.
! These are modeled on the routines in energy/enefvp.src.
! Jay Banks, 19-20 December 1995, adapted from routines in
! energy/enefvp.src by Axel Brunger, May-1985.
!
#if KEY_MMFF==1
SUBROUTINE MAKITC_MM(NITCC,NITCC2,IGCNB,LOWTP,LVDW,LVSHFT,CTOFNB, &
     CCNBA,CCNBB,CCNBC,CCNBD,GAMBUF,DELBUF)
  !-----------------------------------------------------------------------
  !      This routine does the actual work of filling the MMFF vdw tables.
  !           - BRB    1/4/85, overhauled  3/19/98
  !
  !      Adapted for MMFF by Jay Banks, 03-JAN-96.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use param
  implicit none
  !
  INTEGER NITCC,NITCC2,IGCNB(*),LOWTP(*)
  LOGICAL LVDW,LVSHFT
  real(chm_real)  CTOFNB
  real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
  real(chm_real)  GAMBUF,DELBUF
  !
  !
  INTEGER I,J,IU,JU,IPT,JPT
  real(chm_real)  CA,CB,CC,CD
  real(chm_real) poff, poff2, poff3, poff6, poff7, poffdi, poff7gi
  real(chm_real) foff, hoff, ghoff, vtoff, vtpoff
  !
  IF(LVDW) THEN
     !
     !  vdw "generalized shifting" coefficients (see comment in
     !  mmff/evdw_mm.src):
     !
     !     E(R) = CB*{[((1+GAMBUF)*CA)/(R+GAMBUF*CA)]**7 *
     !                [((1+DELBUF)*CA**7)/(R**7+DELBUF*CA**7) - 2]}
     !            + CC*R**4 + CD,
     !
     !     where CC and CD are chosen s.t. E(CTOFNB) = E'(CTOFNB) = 0.
     !
     JPT=0
     DO IU=1,NITCC
        DO JU=1,IU
           JPT=JPT+1
           IPT=MAX(IGCNB(IU),IGCNB(JU))
           IPT=LOWTP(IPT) + IGCNB(IU) + IGCNB(JU)
           CA=CNBA(IPT)
           CB=CNBB(IPT)
           CCNBA(JPT)=CA
           CCNBB(JPT)=CB
           IF(LVSHFT) THEN
              poff = CTOFNB/CA
              poff2 = poff * poff
              poff3 = poff * poff2
              poff6 = poff3 * poff3
              poff7 = poff6 * poff
              poffdi = 1.0 / (poff + DELBUF)
              poff7gi = 1.0 / (poff7 + GAMBUF)
              foff = CB * ((1.+DELBUF)*poffdi)**7
              hoff = (1.+GAMBUF)*poff7gi
              ghoff = hoff - 2.
              vtoff = foff * ghoff                      !V_true(r_off)
              vtpoff = -(SEVEN*foff/CA) * &
                   (poffdi*ghoff + poff6*hoff*poff7gi)    !V_true'(r_off)
              CC = -vtpoff/(CTOFNB*CTOFNB*CTOFNB)       !beta=4
              CD = -vtoff + 0.25*vtpoff*CTOFNB
              CCNBC(JPT)=CC
              CCNBD(JPT)=CD
           ENDIF
           !
           !     do 1-4 interaction vdw parameters
           !
           IPT=IPT+MAXCN
           CA=CNBA(IPT)
           CB=CNBB(IPT)
           CCNBA(JPT+NITCC2)=CA
           CCNBB(JPT+NITCC2)=CB
           IF(LVSHFT) THEN
              poff = CTOFNB/CA
              poff2 = poff * poff
              poff3 = poff * poff2
              poff6 = poff3 * poff3
              poff7 = poff6 * poff
              poffdi = 1.0 / (poff + DELBUF)
              poff7gi = 1.0 / (poff7 + GAMBUF)
              foff = CB * ((1.+DELBUF)*poffdi)**7
              hoff = (1.+GAMBUF)*poff7gi
              ghoff = hoff - 2.
              vtoff = foff * ghoff                      !V_true(r_off)
              vtpoff = -(SEVEN*foff/CA) * &
                   (poffdi*ghoff + poff6*hoff*poff7gi)    !V_true'(r_off)
              CC = -vtpoff/(CTOFNB*CTOFNB*CTOFNB)       !beta=4
              CD = -vtoff + 0.25*vtpoff*CTOFNB
              CCNBC(JPT+NITCC2)=CC
              CCNBD(JPT+NITCC2)=CD
           ENDIF
        ENDDO
     ENDDO
     !
  ELSE
     DO I=1,NITCC2
        CCNBA(I)=0.0
        CCNBB(I)=0.0
        !brb..07-FEB-99 Protect initialization of array assignments
        IF(LVSHFT) THEN
           CCNBC(I)=0.0
           CCNBD(I)=0.0
        ENDIF
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE MAKITC_MM

SUBROUTINE MAKSTRB(NSTRBV,ISBV,JSBV,KSBV,LSBV,VINDSB, &
     VFK,VBEQ,VTEQ)
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use psf
  use param
  use mmffm
  use stream
  use string
  use code
  implicit none
  !
  INTEGER ISBV(*),JSBV(*),KSBV(*),LSBV(*)
  INTEGER NSTRBV,VINDSB(*)
  real(chm_real) VFK(*),VBEQ(*),VTEQ(*)
  !
  !     indices
  INTEGER IC,ISTRB,ND,NB,CUR_ICB,CUR_ICT,CUR_JT
  real(chm_real) FK
  !
  IF (NTHETA.EQ.0) RETURN
  !
  NSTRBV=0
  DO ISTRB = 1,NTHETA
     IC=ICSTBN(ISTRB)
     IF (IC.NE.0) THEN
        CUR_ICT=ICT(ISTRB)
        CUR_JT = JT(ISTRB)
        IF(CUR_ICT.GT.0.AND.MLINBA(MTYPE(CUR_JT)).EQ.0) THEN
           DO ND=1,2
              IF(IC.GT.0) THEN
                 FK=STBNP(ND,IC)
              ELSE
                 FK=STBNP(3-ND,ABS(IC))
              ENDIF
              NB=StrbList(ND,ISTRB)
              if(nb.le.0 .or. nb.gt.MAXB) then
                 write(SCRTCH,'(a,2i6)') &
                      'nb out of range: nb,MAXB=',nb,MAXB
                 call wrndie(-5,'<MAKSTRB>',SCRTCH(:37))
                 return
              endif
              CUR_ICB=ICB(NB)
              IF(FK.NE.ZERO.AND.CUR_ICB.GT.0) THEN
                 NSTRBV=NSTRBV+1
                 ISBV(NSTRBV) = IT(ISTRB)
                 JSBV(NSTRBV) = CUR_JT
                 KSBV(NSTRBV) = KT(ISTRB)
                 LSBV(NSTRBV) = IB(NB)
                 IF(LSBV(NSTRBV).EQ.CUR_JT) LSBV(NSTRBV) = JB(NB)
                 VINDSB(NSTRBV)=ISTRB
                 VFK(NSTRBV)=FK
                 VBEQ(NSTRBV)=BondEq(CUR_ICB)
                 VTEQ(NSTRBV)=AnglEq(CUR_ICT)
              ENDIF
           ENDDO
        ENDIF  ! (CUR_ICT, MLINBA(CUR_JT))
     ENDIF  ! (IC.NE.0)
  ENDDO
  RETURN
END SUBROUTINE MAKSTRB
!
SUBROUTINE MAKOOPL(NOOPLV,IOPV,JOPV,KOPV,LOPV,VINDOP, &
     VOOPLFC)
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use psf
  use param
  use mmffm
  use stream
  use string
  use code
  implicit none
  !
  INTEGER IOPV(*),JOPV(*),KOPV(*),LOPV(*)
  INTEGER NOOPLV,VINDOP(*)
  real(chm_real) VOOPLFC(*)
  !
  !     indices
  INTEGER IC,IOOPL,CUR_LT
  !
  IF (NTHETA.EQ.0) RETURN
  !
  NOOPLV=0
  DO IOOPL = 1,NTHETA
     IC=ICOOP(IOOPL)
     CUR_LT=LTHETA(IOOPL)
     IF (IC.GT.0.AND.CUR_LT.NE.0) THEN
        NOOPLV=NOOPLV+1
        IOPV(NOOPLV) = IT(IOOPL)
        JOPV(NOOPLV) = JT(IOOPL)
        KOPV(NOOPLV) = KT(IOOPL)
        LOPV(NOOPLV) = CUR_LT
        VINDOP(NOOPLV)=IOOPL
        VOOPLFC(NOOPLV)=OOPLFC(IC)
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE MAKOOPL

SUBROUTINE MAKPHI2_MM(NPHIV,IPV,JPV,KPV,LPV,VIND, &
     TorFC1,TorFC2,TorFC3)
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use psf
  use param
  use stream
  use code
  implicit none
  !
  INTEGER IPV(*),JPV(*),KPV(*),LPV(*),VIND(*),NPHIV
  real(chm_real) TorFC1(*),TorFC2(*),TorFC3(*)
  !
  !     indices
  INTEGER IC,IPHI
  IF (NPHI.EQ.0) RETURN
  !
  NPHIV=0
  DO IPHI = 1,NPHI
     IC=ICP(IPHI)
     IF (IC.NE.0) THEN
        NPHIV=NPHIV+1
        IPV(NPHIV) = IP(IPHI)
        JPV(NPHIV) = JP(IPHI)
        KPV(NPHIV) = KP(IPHI)
        LPV(NPHIV) = LP(IPHI)
        VIND(NPHIV)=IPHI
        TorFC1(NPHIV) = CPC(3 * IC - 2)
        TorFC2(NPHIV) = CPC(3 * IC - 1)
        TorFC3(NPHIV) = CPC(3 * IC)
     ENDIF
  ENDDO
  !
end SUBROUTINE MAKPHI2_MM
#endif 

