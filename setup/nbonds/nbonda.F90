SUBROUTINE NBONDA(NNNB,JNB,MAXJNB,INBLO,X,Y,Z, &
     INB14,IBLO14,CUTNB,WRNMIN,CMPLTD, &
#if KEY_MTS == 1
     NNMT1,MAXJM1,JNM1,INBLM1, &
     NNMT2,MAXJM2,JNM2,INBLM2, &
#endif
#if KEY_PERT == 1
     NNNBP,MAXJNP,JNBP,INBLOP, &
     NNNBR,MAXJNR,JNBR,INBLOR, &
     IPERT,INB14P,IBL14P, &
#endif
#if KEY_TSM == 1
     LTSM,REACLS,PRODLS, &
#endif
     RSCMX,RSCMY,RSCMZ, &
     RSXMAX,RSYMAX,RSZMAX,RSDISP, &
#if KEY_FOURD == 1
     RSFMAX,RSCMF, &
#endif
     ATSX,ATSY,ATSZ, &
     MAXNBTHOLE, NBTHOL, NBTHOLIJ, NBTHOLP, &
     NBTHOL1, NBTHOL2, NBTHOL3, PPNBTHOLP, &
     PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, THOLCUT)

  !-----------------------------------------------------------------------
  !     THIS ROUTINE CONSTRUCTS THE ATOM BASED NONBONDED LISTS
  !
  !     22-AUG-1981  By Bernard R. Brooks
  !     Overhauled (group lists removed) - BRB - October 25, 1996
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
#if KEY_MTS == 1
  use tbmts
#endif
#if KEY_PERT == 1
  use pert
#endif
  use exclm
  use gamess_fcm
  use psf
  use stream
  use timerm
#if KEY_PARALLEL == 1
  use parallel
#endif
#if KEY_REPLICA == 1
  use replica_mod
#endif
#if KEY_FOURD == 1
  use fourdm
#endif
#if KEY_PBOUND == 1
  use pbound
#endif
  use chutil,only:initia,atomid
  use machutil,only:die,timre,timrb
  !--- ##SE nbutil_module,only:qinlist
  implicit none
  !
  INTEGER NNNB,MAXJNB,JNB(*),INBLO(*)
  real(chm_real)  X(*),Y(*),Z(*)
  INTEGER INB14(*),IBLO14(*)
  real(chm_real)  CUTNB,WRNMIN
  LOGICAL CMPLTD
  logical,external :: qinlist
  real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
  real(chm_real)  RSXMAX(*),RSYMAX(*),RSZMAX(*)
  INTEGER RSDISP(*)
  real(chm_real)  ATSX(*),ATSY(*),ATSZ(*)
! Drude thole shielding
  INTEGER MAXNBTHOLE, NBTHOL, NBTHOLIJ(2,*), NBTHOLP, PPNBTHOLP
  INTEGER NBTHOL1(*), NBTHOL2(*), NBTHOL3(*)
  INTEGER PPNBTHOL1(*), PPNBTHOL2(*), PPNBTHOL3(*)
  real(chm_real) THOLCUT
  !
#if KEY_MTS == 1 /*mtsdecl*/
  INTEGER NNMT1,MAXJM1,JNM1(*),INBLM1(*)
  INTEGER NNMT2,MAXJM2,JNM2(*),INBLM2(*)
#endif /* (mtsdecl)*/
  !
#if KEY_PERT == 1 /*pertdecl*/
  INTEGER NNNBP,MAXJNP,JNBP(*),INBLOP(*)
  INTEGER NNNBR,MAXJNR,JNBR(*),INBLOR(*)
  INTEGER IPERT(*),INB14P(*),IBL14P(*)
  ! PERT local declarations
  INTEGER NXIP,NXIMXP
#if KEY_CHEMPERT == 1
  !sbcp   chem pert aux. variable
  integer iprtsu
#endif
#endif /* (pertdecl)  IF PERT*/
#if KEY_TSM == 1 /*tsmdecl*/
  LOGICAL LTSM
  INTEGER REACLS(*),PRODLS(*)
#endif /* (tsmdecl)*/
  !
#if KEY_PARALLEL == 1 /*pardecl*/
  INTEGER IMYNOD
#endif /* (pardecl)*/
  !
#if KEY_REPLICA == 1 /*repdecl*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  integer iRepNo, iRepID
#endif /* (repdecl)  REPLICA*/
  !
  INTEGER ITMP,JTMP
  LOGICAL LTMP, qErr
  !RCZ
  INTEGER I,J,IS,IQ,NAT,NGAT,IRS,NXI,NXIMAX
  INTEGER JRS,JS,JQ,IRST,JRST,INBX,IX14,IX14P,IGRPMX
  real(chm_real) CTNBSQ,WMINSQ
  real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
  real(chm_real) XD1,YD1,ZD1,R2,R,XI,YI,ZI,DXT,DYT,DZT,GRPMXS
  LOGICAL MOVEFG,QMOVE,DOIT
  CHARACTER(len=8) :: SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  !
#if KEY_FOURD == 1 /*4ddecl*/
  !     4D variable:
  real(chm_real) RSFMAX(*),RSCMF(*)
  real(chm_real) FD,FDI,FMIN,FMAX,FD1
#endif /* (4ddecl)*/
  !
#if KEY_PBOUND == 1 /*pbound*/
  real(chm_real) CORR
#endif /*     (pbound)*/
  integer nl0,nl1,nli
  !
  !
  !-----------------------------------------------------------------------
  !
  NBTHOLP = 0
  PPNBTHOLP = 0
  IF (TIMER > 0) CALL TIMRB
#if KEY_REPLICA == 1 /*repsetup*/
  IF (qRep) nRepXA = 0
#endif /* (repsetup)  REPLICA*/
  !
  CMPLTD=.FALSE.
  CTNBSQ=CUTNB*CUTNB
  WMINSQ=WRNMIN*WRNMIN
  QMOVE=.FALSE.
  !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
  GRPMXS=ZERO
  IGRPMX=1
  !
  ! Store the current atom configuration.
  ATSX(1:natom)=X(1:natom)
  ATSY(1:natom)=Y(1:natom)
  ATSZ(1:natom)=Z(1:natom)
  !
  ! Find geometric center for each group
  DO I=1,NGRP
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF(NAT <= 0) CALL DIE
     XMIN=X(IS)
     XMAX=XMIN
     YMIN=Y(IS)
     YMAX=YMIN
     ZMIN=Z(IS)
     ZMAX=ZMIN
#if KEY_FOURD == 1 /*4dset1*/
     IF(DIM4) THEN
        FMIN=FDIM(IS)
        FMAX=FMIN
     ENDIF
#endif /* (4dset1)*/
     IF(IMOVE(IS) > 0) QMOVE=.TRUE.
     DO J=IS+1,IQ
        XMIN=MIN(X(J),XMIN)
        YMIN=MIN(Y(J),YMIN)
        ZMIN=MIN(Z(J),ZMIN)
        XMAX=MAX(X(J),XMAX)
        YMAX=MAX(Y(J),YMAX)
        ZMAX=MAX(Z(J),ZMAX)
#if KEY_FOURD == 1 /*4dset2*/
        IF(DIM4) THEN
           FMIN=MIN(FDIM(J),FMIN)
           FMAX=MAX(FDIM(J),FMAX)
        ENDIF
#endif /* (4dset2)*/
        IF(IMOVE(J) > 0) QMOVE=.TRUE.
     ENDDO
     XD=HALF*(XMIN+XMAX)
     YD=HALF*(YMIN+YMAX)
     ZD=HALF*(ZMIN+ZMAX)
     ! Size of rectangular box surrounding group.
     RSXMAX(I)=XMAX-XD
     RSYMAX(I)=YMAX-YD
     RSZMAX(I)=ZMAX-ZD
     !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
     IF(RSXMAX(I) > GRPMXS) THEN
        GRPMXS=RSXMAX(I)
        IGRPMX=IS
     ENDIF
     IF(RSYMAX(I) > GRPMXS) THEN
        GRPMXS=RSYMAX(I)
        IGRPMX=IS
     ENDIF
     IF(RSZMAX(I) > GRPMXS) THEN
        GRPMXS=RSZMAX(I)
        IGRPMX=IS
     ENDIF
     !brb..07-FEB-99
     ! center of group defined by the box
     RSCMX(I)=XD
     RSCMY(I)=YD
     RSCMZ(I)=ZD
#if KEY_FOURD == 1 /*4dset4*/
     IF(DIM4) THEN
        FD=HALF*(FMIN+FMAX)
        RSFMAX(I)=FMAX-FD
        RSCMF(I)=FD
     ENDIF
#endif /* (4dset4)*/
  ENDDO
  !
  !brb..07-FEB-99 Add warning for disconnected electrostatic groups.
  GRPMXS=GRPMXS*TWO
  IF(GRPMXS > TWELVE) THEN
     IF(WRNLEV >= -1) WRITE(OUTU,137) GRPMXS,IGRPMX
137  FORMAT( &
          ' NBONDA>>  Maximum group spatial extent (12A) exceeded.',/ &
          '   Size is',F12.2,' Angstroms and starts with atom:',I8,/ &
          '   Please check group boundary definitions.')
     !yw...to run testcases         CALL DIEWRN(-1)
  ENDIF
  !brb..07-FEB-99
  !
  ! Now decide how to treat each residue pair using a rectangular
  ! search and store the disposition in rsdisp.
  !
  NNNB=0
  NGAT=0
#if KEY_MTS == 1 /*mtszero*/
  IF (QTBMTS) THEN
     NNMT1=0
     NNMT2=0
  ENDIF
#endif /* (mtszero)*/
#if KEY_PERT == 1 /*pertzero*/
  IF(QPERT) THEN
     NNNBR=0
     NNNBP=0
  ENDIF
#endif /* (pertzero)*/

!=======================================================================
!  Expand control section
!-------------------------------------------------------------------
! (disable expand when debug is active)
#if KEY_DEBUG == 1
#undef KEY_EXPAND
#endif

#if KEY_EXPAND == 1 && KEY_FOURD == 1

!-------------------------------------------------------------------
! Do DIM4 expansion of code
! ##EXPAND  F nofourd     .when. FOURD  EXPAND  (expand_dim4)
! ##PASS1   .not.EXPAND
  IF(DIM4) THEN

#undef KEY_EXPAND
#include "nbonda1.inc"
#define KEY_EXPAND 1

! ##PASS2   .not.FOURD nofourd
  ELSE

#undef KEY_FOURD
#define NBONDA_NOFOURD 1

#include "nbonda1.inc"

#define KEY_FOURD 1
#undef NBONDA_NOFOURD

! ##EXFIN
  ENDIF
! ##EXEND
! ##ENDEX    (expand_dim4)

#else /* KEY_EXPAND && KEY_FOURD */

#define NBONDA_F_FLAG 1
#define NBONDA_NOFOURD 1

#include "nbonda1.inc"

#undef NBONDA_F_FLAG
#undef NBONDA_NOFOURD

#endif /* KEY_EXPAND && KEY_FOURD */
!=======================================================================

  CMPLTD=.TRUE.
  !
245 FORMAT(' WARNING: ATOMS',4(1X,A), &
       ' AND',4(1X,A),' ONLY',F5.2,' A. APART')
  !
  ! Termination of the routine.
  !
#if KEY_MTS == 1
  IF(.not.(SLFG.OR.TBHY1)) then     ! GOTO 711
#endif
     !
     IF(PRNLEV >= 5) THEN
        WRITE(OUTU,720)
720     FORMAT(/' General atom nonbond list generation found:')
        ! Atom lists
        WRITE(OUTU,733) NNNB
733     FORMAT(I9,' ATOM PAIRS WERE FOUND FOR ATOM LIST')
      IF(NBTHOLP.GT.0)THEN
      write(outu,'(i9,a)') NBTHOLP, &
                       ' SPECIFIC THOLE SHIELDING PAIRS FOUND '
      ENDIF
#if KEY_PERT == 1 /*pertprint2*/
        IF(QPERT) THEN
           WRITE(OUTU,734) NNNBR
734        FORMAT(I9,' ATOM PAIRS WERE FOUND FOR REACTANT LIST')
           WRITE(OUTU,735) NNNBP
735        FORMAT(I9,' ATOM PAIRS WERE FOUND FOR PRODUCT  LIST')
        ENDIF
#endif /* (pertprint2)*/
        WRITE(OUTU,736) NGAT
736     FORMAT(I9,' GROUP PAIRS REQUIRED ATOM SEARCHES'/)
     ENDIF
     !
#if KEY_MTS == 1 /*mtsprint*/
  endif      !711 CONTINUE
#if KEY_PARALLEL == 0
  IF(PRNLEV >= 5) THEN
     IF (QTBMTS .AND. (SLFG .OR. TBHY1)) THEN
        WRITE(OUTU,720)
        IF(SLFG) THEN
           WRITE(OUTU,717) NNMT2
           WRITE(OUTU,716) NNMT1
717        FORMAT(' MTS> ',I9,' LONG-RANGE  ATOM PAIRS WERE FOUND')
716        FORMAT(' MTS> ',I9,' SHORT-RANGE ATOM PAIRS WERE FOUND')
        ENDIF
        IF(TBHY1) THEN
           WRITE(OUTU,718) NNMT1
718        FORMAT(I9,' ATOM PAIRS WERE FOUND FOR FAST MODE')
           WRITE(OUTU,719) NNMT2
719        FORMAT(I9,' ATOM PAIRS WERE FOUND FOR SLOW MODE')
        ENDIF
     ENDIF
  ENDIF
#endif
#endif /* (mtsprint)*/
  !
#if KEY_REPLICA == 1 /*repprint*/
  !# <caves>-Aug-4-1993 (Leo Caves)
  IF (PRNLEV >= 5.AND.qRep) THEN
     WRITE(OUTU,'(I9,A/)') nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
  ENDIF ! PRNLEV
#endif /* (repprint)  REPLICA*/
  IF (TIMER == 1) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,830) 'TOTAL TIME IN NBONDA'
830  FORMAT(1X,A)
     CALL TIMRE
     CALL TIMRB
  ENDIF
  !
  RETURN
END SUBROUTINE NBONDA

SUBROUTINE MKTHOLELIST(I,J,iAC, jAC, MAXNBTHOLE, NBTHOL, &
     NBTHOLIJ, NBTHOLP, NBTHOL1, NBTHOL2, NBTHOL3, R2)
!
  use chm_kinds
  use dimens_fcm
  use coord
  use stream
  implicit none
  integer I, J, iAC, jAC
  integer NBTHOL, NBTHOLIJ(2,*)
  integer MAXNBTHOLE
  integer NBTHOLP, NBTHOL1(*), NBTHOL2(*), NBTHOL3(*)
  real(chm_real) R2
! local variables
  integer k
  if(iAC.ge.jAC)then
     do k=1,NBTHOL
        if((iAC.eq.NBTHOLIJ(1,k)).and.(jAC.eq.NBTHOLIJ(2,k)))then
           NBTHOLP=NBTHOLP+1
           if(NBTHOLP.gt.MAXNBTHOLE)then
              CALL WRNDIE(-2,'<MKTHOLELIST>', &
                   'Number of Thole pairs exceeds dimension')
           endif
           NBTHOL1(NBTHOLP)= j
           NBTHOL2(NBTHOLP)= i
           NBTHOL3(NBTHOLP)= k
           if(PRNLEV.ge.7)then
              write(*,'(i5,a,7i5,2f12.4)') NBTHOLP,' (a) thole pair found ', &
                   i,j,iAc,jAC,k, &
                   NBTHOLIJ(1,k), NBTHOLIJ(2,k), sqrt(R2)
           endif
        endif
     enddo
  else
     do k=1,NBTHOL
        if((iAC.eq.NBTHOLIJ(2,k)).and.(jAC.eq.NBTHOLIJ(1,k)))then
           NBTHOLP=NBTHOLP+1
           if(NBTHOLP.gt.MAXNBTHOLE)then
              CALL WRNDIE(-2,'<MKTHOLELIST>', &
                   'Number of Thole pairs exceeds dimension')
           endif
           NBTHOL1(NBTHOLP)= i
           NBTHOL2(NBTHOLP)= j
           NBTHOL3(NBTHOLP)= k
           if(PRNLEV.ge.7)then
              write(*,'(i5,a,7i5,2f12.4)') NBTHOLP,' (b) thole pair found ', &
                   i,j,iAc,jAC,k, &
                   NBTHOLIJ(2,k), NBTHOLIJ(1,k), sqrt(R2)
           endif
        endif
      enddo
   endif

   RETURN
 END SUBROUTINE MKTHOLELIST
