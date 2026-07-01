#if KEY_PIPF==1 /*pipf_main*/
SUBROUTINE VPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
     DD1,IUPT,QSECD &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES MEMORY FOR VIBRATIONAL ANALYSIS WITH 
  !     POLARIZATION
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !     QSECD  - calculate energy second derivative
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use memory
  use exfunc
  use energym
  use pipfm
  implicit none
  !
  INTEGER IFRSTA, NATOM, IUPT(*) 
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,QSECD
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*),DD1(*)
  !
  real(chm_real),allocatable,dimension(:) :: IMTXA
  real(chm_real),allocatable,dimension(:) :: IMTXB
  real(chm_real),allocatable,dimension(:) :: IDMUIND
  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: ITXWW
  real(chm_real),allocatable,dimension(:) :: IVECE
  real(chm_real),allocatable,dimension(:) :: IVV
  real(chm_real),allocatable,dimension(:) :: IINDXX
  real(chm_real),allocatable,dimension(:) :: IHSSN
  real(chm_real),allocatable,dimension(:) :: INDTS
  real(chm_real),allocatable,dimension(:) :: IRDTS
  real(chm_real),allocatable,dimension(:) :: IFTHTS

  INTEGER I,J,NB,ITEMP,NPR,N3,NB6
  INTEGER NN3,NITPR,NITPR9,NITPR27,NITPR81,I1
  INTEGER ITTSR
  !
  !
  NB = 0
  ITEMP = 0

  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) NB = NB + NPR
  ENDDO


  !
  !     ALLOCATE MEMORY
  !
  N3 = 3*NATOM
  NN3 = N3*N3
  NITPR = NATOM*(NATOM - 1)
  NB6 = 6*NB
  NITPR = NATOM*NATOM
  NITPR9 = 9*NITPR
  NITPR27 = 27*NITPR
  NITPR81 = 81*NITPR
  !
  !     ALLOCATE MEMORY
  !
  call chmalloc('vpipf.src','VPIPF','IMTXA',NN3,crl=IMTXA)
  call chmalloc('vpipf.src','VPIPF','IMTXB',NN3,crl=IMTXB)
  call chmalloc('vpipf.src','VPIPF','IDMUIND',N3,crl=IDMUIND)
  call chmalloc('vpipf.src','VPIPF','IEFIELD',N3,crl=IEFIELD)
  call chmalloc('vpipf.src','VPIPF','ITXWW',NB6,crl=ITXWW)
  call chmalloc('vpipf.src','VPIPF','IVECE',N3,crl=IVECE)
  call chmalloc('vpipf.src','VPIPF','IVV',N3,crl=IVV)
  call chmalloc('vpipf.src','VPIPF','IINDXX',N3,crl=IINDXX)
  call chmalloc('vpipf.src','VPIPF','IHSSN',NITPR9,crl=IHSSN)
  call chmalloc('vpipf.src','VPIPF','INDTS',NITPR9,crl=INDTS)
  call chmalloc('vpipf.src','VPIPF','IRDTS',NITPR27,crl=IRDTS)
  call chmalloc('vpipf.src','VPIPF','IFTHTS',NITPR81,crl=IFTHTS)

  call VPIPF2(IFRSTA,NATOM,JNB,INBLO,CG,MAXROW,IAC,ITC,LELEC,LCONS, &
       LSHFT,LVSHFT,LFSWT,LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB, &
       EPS,E14FAC,ALP,N3,INDTS,IRDTS,IFTHTS,IHSSN,IEFIELD,ITXWW, &
       IDMUIND,IMTXA,IMTXB,IVECE,IVV,IINDXX,DD1,IUPT, &
       QSECD)
  !
  !     FREE MEMORY
  !
  call chmdealloc('vpipf.src','VPIPF','IEFIELD',N3,crl=IEFIELD)
  call chmdealloc('vpipf.src','VPIPF','ITXWW',NB6,crl=ITXWW)
  call chmdealloc('vpipf.src','VPIPF','IDMUIND',N3,crl=IDMUIND)
  call chmdealloc('vpipf.src','VPIPF','IMTXA',NN3,crl=IMTXA)
  call chmdealloc('vpipf.src','VPIPF','IMTXB',NN3,crl=IMTXB)
  call chmdealloc('vpipf.src','VPIPF','IVECE',N3,crl=IVECE)
  call chmdealloc('vpipf.src','VPIPF','IVV',N3,crl=IVV)
  call chmdealloc('vpipf.src','VPIPF','IINDXX',N3,crl=IINDXX)
  call chmdealloc('vpipf.src','VPIPF','IHSSN',NITPR9,crl=IHSSN)
  call chmdealloc('vpipf.src','VPIPF','INDTS',NITPR9,crl=INDTS)
  call chmdealloc('vpipf.src','VPIPF','IRDTS',NITPR27,crl=IRDTS)
  call chmdealloc('vpipf.src','VPIPF','IFTHTS',NITPR81,crl=IFTHTS)

  RETURN
END SUBROUTINE VPIPF
!
!
SUBROUTINE VPIPF2(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     ALP,N3,NDTS,RDTS,FTHTS,HSSN,EFIELD,TXWW,DMUIND,MTXA, &
     MTXB,VECE,VV,INDXX, &
     DD1,IUPT,QSECD &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES HESSIAN ELELEMENTS FOR VIBRATIONAL ANALYSI
  !     WITH POLARIZATION 
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !
  !     DMUIND - induced dipole for current iteration
  !     DDT - induced dipole for previous iteration
  !     EFIELD - permenent electric field
  !     EIND - induced electric field
  !     TXWW - rank 2 polar tensor
  !     IPOL - tag to mark an atom as a dipole center
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use exfunc
  use energym
  use stream
  use pipfm
  implicit none
  !
  INTEGER IFRSTA, NATOM, N3, IUPT(*)
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),EPOL,DD1(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,QSECD
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*),ALP(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) NDTS(3,3,*),RDTS(3,3,3,*),FTHTS(3,3,3,3,*),HSSN(3,3,*)
  real(chm_real) INDXX(*)
  real(chm_real) DMUIND(3,*),EFIELD(3,*),TXWW(6,*)
  real(chm_real) MTXA(N3,N3),MTXB(N3,N3),VECE(N3),VV(N3)
  !
  INTEGER I,J,JPR,NB,NPR,IADD,JJ,II
  INTEGER K1,K2,K3,K4,K5,KA1,KAI,KAJ
  INTEGER NBI,NBJ,NBI3,NBJ3
  real(chm_real) STGFLD,NDGFLD,HSKA,HSST,HSND,HSRD,HSFR,G1,G2
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IDXHS,IDXHS1,IDXHS2,IDXKAI,IDXKAJ

  !
  !     GET THE DIPOLE TENSOR: SECOND, THIRD AND FOURTH ORDER  
  !
  CALL FTHTSINT(IFRSTA,NATOM,JNB,INBLO,CG, &
       DX,DY,DZ,X,Y,Z,FTHTS,RDTS,NDTS,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)

  !
  !     GET THE INDUCED DIPOLES BY MATRIX INVERSTION
  !
  !
  CALL EPFINV2(IFRSTA,NATOM,JNB,INBLO,CG, &
       MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
       LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       EPOL,ALP,DMUIND,EFIELD,TXWW,MTXA,MTXB,VECE,VV, &
       INDXX,N3,.FALSE. &
       )


  DO I=IFRSTA,NATOM
     DO J=IFRSTA,NATOM
        IDXHS=(I-1)*NATOM+J
        DO K1=1,3
           DO K2=1,3
              IF(I.EQ.J) THEN
                 HSST=0.0D0
                 HSND=0.0D0
                 HSRD=0.0D0
                 STGFLD=0.0D0
                 DO KA1=IFRSTA,NATOM
                    IDXHS1=(I-1)*NATOM+KA1
                    DO K3=1,3
                       NDGFLD=0.0D0
                       STGFLD=STGFLD+RDTS(K1,K2,K3,IDXHS1)* &
                            DMUIND(K3,KA1)
                       NDGFLD=NDGFLD-CG(KA1)*RDTS(K3,K1,K2,IDXHS1)
                       DO K4=1,3
                          NDGFLD=NDGFLD+FTHTS(K3,K1,K2,K4,IDXHS1)* &
                               DMUIND(K4,KA1)
                       ENDDO
                       HSND=HSND-DMUIND(K3,I)*NDGFLD
                    ENDDO
                 ENDDO
                 HSST=-CG(I)*STGFLD
                 !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                 !     K1: LANDA, K2: SIGMA, K3: ALPHA
                 !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
              ELSE
                 HSST=0.0D0
                 HSND=0.0D0
                 HSRD=0.0D0
                 DO K3=1,3
                    HSST=HSST+RDTS(K1,K2,K3,IDXHS)*DMUIND(K3,J)
                    HSND=HSND-DMUIND(K3,I)*RDTS(K3,K1,K2,IDXHS)
                    DO K4=1,3
                       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                       !     K1: LANDA, K2: SIGMA, K3: ALPHA, K4: BETA
                       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                       HSRD=HSRD+DMUIND(K3,I) &
                            *FTHTS(K3,K1,K2,K4,IDXHS)*DMUIND(K4,J)
                    ENDDO
                 ENDDO
                 HSST=HSST*CG(I)
                 HSND=HSND*CG(J)                    
              ENDIF
              !
              !    CALCULATE THE FOURTH TERM
              !
              !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
              !     K1: LANDA, K2: SIGMA, K3: ALPHA, K4: BETA, K5: GAMA
              !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
              HSFR=0.0D0
              DO NBI=IFRSTA,NATOM
                 NBI3=3*NBI-3
                 IDXHS1=(I-1)*NATOM+NBI
                 DO NBJ=IFRSTA,NATOM
                    NBJ3=3*NBJ-3
                    IDXHS2=(NBJ-1)*NATOM+J
                    DO K3=1,3
                       DO K4=1,3
                          G1=0.0D0
                          G2=0.0D0
                          IF(I.EQ.NBI) THEN
                             DO KAI=IFRSTA,NATOM
                                IDXKAI=(I-1)*NATOM+KAI
                                DO K5=1,3
                                   G1=G1+RDTS(K1,K3,K5,IDXKAI)* &
                                        DMUIND(K5,KAI)
                                ENDDO
                                !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                                G1=G1-NDTS(K1,K3,IDXKAI)*CG(KAI)
                             ENDDO
                          ELSE
                             DO K5=1,3
                                G1=G1+DMUIND(K5,I) &
                                     *RDTS(K5,K1,K3,IDXHS1)
                                !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                             ENDDO
                             G1=G1+NDTS(K1,K3,IDXHS1)*CG(I)
                          ENDIF
                          IF(J.EQ.NBJ) THEN
                             DO KAJ=IFRSTA,NATOM
                                IDXKAJ=(J-1)*NATOM+KAJ
                                DO K5=1,3
                                   G2=G2+RDTS(K4,K2,K5,IDXKAJ)* &
                                        DMUIND(K5,KAJ)
                                ENDDO
                                G2=G2-NDTS(K4,K2,IDXKAJ)*CG(KAJ)
                             ENDDO
                          ELSE
                             DO K5=1,3
                                G2=G2-RDTS(K4,K2,K5,IDXHS2) &
                                     *DMUIND(K5,J)
                                !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                             ENDDO
                             G2=G2+NDTS(K4,K2,IDXHS2)*CG(J)
                          ENDIF
                          HSFR=HSFR-G1*MTXB(NBI3+K3,NBJ3+K4)*G2
                       ENDDO
                    ENDDO
                 ENDDO
              ENDDO
              HSSN(K1,K2,IDXHS)=(HSST+HSND+HSRD+HSFR)*CCELEC
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !  Update the second derivative array
  DO I=IFRSTA,NATOM 
     II=3*I-2
     DO J=I,NATOM
        IDXHS=(I-1)*NATOM+J
        JJ=3*J-2
        IF(I.EQ.J) THEN
           IADD=IUPT(II)+II
           DD1(IADD)=DD1(IADD)+HSSN(1,1,IDXHS)
           IADD=IUPT(II)+II+1
           DD1(IADD)=DD1(IADD)+HSSN(1,2,IDXHS)
           IADD=IUPT(II)+II+2
           DD1(IADD)=DD1(IADD)+HSSN(1,3,IDXHS)
           IADD=IUPT(II+1)+II+1
           DD1(IADD)=DD1(IADD)+HSSN(2,2,IDXHS)
           IADD=IUPT(II+1)+II+2
           DD1(IADD)=DD1(IADD)+HSSN(2,3,IDXHS)
           IADD=IUPT(II+2)+II+2
           DD1(IADD)=DD1(IADD)+HSSN(3,3,IDXHS)
        ELSE
           IADD=IUPT(II)+JJ
           DD1(IADD)=DD1(IADD)+HSSN(1,1,IDXHS)
           IADD=IUPT(II)+JJ+1
           DD1(IADD)=DD1(IADD)+HSSN(1,2,IDXHS)
           IADD=IUPT(II)+JJ+2
           DD1(IADD)=DD1(IADD)+HSSN(1,3,IDXHS)
           IADD=IUPT(II+1)+JJ
           DD1(IADD)=DD1(IADD)+HSSN(2,1,IDXHS)
           IADD=IUPT(II+1)+JJ+1
           DD1(IADD)=DD1(IADD)+HSSN(2,2,IDXHS)
           IADD=IUPT(II+1)+JJ+2
           DD1(IADD)=DD1(IADD)+HSSN(2,3,IDXHS)
           IADD=IUPT(II+2)+JJ
           DD1(IADD)=DD1(IADD)+HSSN(3,1,IDXHS)
           IADD=IUPT(II+2)+JJ+1
           DD1(IADD)=DD1(IADD)+HSSN(3,2,IDXHS)
           IADD=IUPT(II+2)+JJ+2
           DD1(IADD)=DD1(IADD)+HSSN(3,3,IDXHS)
        ENDIF
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE VPIPF2

SUBROUTINE FTHTSINT(IFRSTA,NATOM,JNB,INBLO,CG, &
     DX,DY,DZ,X,Y,Z,FTHTS,RDTS,NDTS,PFCTOF, &
     ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES INITIAL FOURTH ORDER TENSOR FOR THE 
  !     CALCULATION OF FORCE CONSTANT
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC,,,,,,,, - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability
  !     EZERO  - permanent electric field
  !     FTHTS  - fourth order dipole tensor
  !     PFCTOF - polariziation energy cut-off
  !     IAC,ITC - polarizability index
  !     NPDAMP - polarization damping option
  !     DPFAC - damping factor (gaussian width in Thole roh2)
  !     QPFEX - tag to exclude 1-4 polarization
  !----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use stream
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR  
#endif
  !
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) EZERO(3,NATOM)
  real(chm_real) NDTS(3,3,*),FTHTS(3,3,3,3,*),RDTS(3,3,3,*)
  !
  INTEGER I,J,JPR,NB,NPR,ITEMP
  INTEGER K1,K2,K3,K4,K5
  real(chm_real) RK1,RK2,RK3,RK4
  real(chm_real) DTA1,DTA2,DTA3,DTA4,DTA5,DTA6
  real(chm_real) XX,YY,ZZ,R1,R2,R3,R5,R7,R9,QR3I,QR3J,    &
       RR3,RR3XX,RR3YY,RR3ZZ
  !
  real(chm_real) ALP(*),DPFAC,U3,AU3,EXPAU3,A2U6,A3U9
  real(chm_real) FLMD3,FLMD5,FLMD7,FLMD9
  INTEGER IAC(*),ITC(*),NPDAMP,NITPR,IDXTSL,IDXTSU,I1,J1
  !
  real(chm_real) RIJ2,PFCTOF,PFCTOF2
  LOGICAL QPFEX
  PFCTOF2=PFCTOF*PFCTOF
  NITPR = NATOM*NATOM
  DO I = 1,NITPR
     DO K1 = 1,3
        DO K2 = 1,3
           NDTS(K1,K2,I) = 0.0D0
           DO K3 = 1,3
              RDTS(K1,K2,K3,I) = 0.0D0
              DO K4 = 1,3
                 FTHTS(K1,K2,K3,K4,I) = 0.0D0
              ENDDO
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  NB = 0
  ITEMP = 0

  ! DO BLOCK EXPANSION OF CODE TO IMPROVE EFFICIENCY
  IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.0) THEN
#undef PIPF_CTOF
#undef PIPF_DAMP
#include "vpipf.inc"
  ELSE IF(PFCTOF.GT.0.0D0 .AND. NPDAMP.EQ.0) THEN
#define PIPF_CTOF 1
#include "vpipf.inc"
#undef PIPF_CTOF     
  ELSE IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.1) THEN
#define PIPF_DAMP 1
#include "vpipf.inc"
  ELSE
#define PIPF_CTOF 1
#include "vpipf.inc"
#undef PIPF_CTOF
#undef PIPF_DAMP
  ENDIF
  RETURN
END SUBROUTINE FTHTSINT
#else /* (pipf_main)*/
SUBROUTINE NOPIPF
  RETURN
END SUBROUTINE NOPIPF
#endif /* (pipf_main)*/
