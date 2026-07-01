#if KEY_RISM==0 /*rism_main*/

SUBROUTINE NULL_SOLVTION
  RETURN
end SUBROUTINE NULL_SOLVTION

#else /* (rism_main)*/

SUBROUTINE SOLVTION(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This subroutine parse the control command for the thermodynamic
  !     integration.  It is assumed that the correlation function are
  !     converged.  No verification is done about that.

  use dimens_fcm
  use memory
  use stream
  use string
  use number
  use rism
  use struc
  use distri
  use rism_control
  use fft
  use chm_kinds
  implicit none
  !     Command parser
  real(chm_real),allocatable,dimension(:) :: F1R
  real(chm_real),allocatable,dimension(:) :: F2R
  real(chm_real),allocatable,dimension(:) :: F3R
  real(chm_real),allocatable,dimension(:,:) :: PHIR
  real(chm_real),allocatable,dimension(:,:) :: W1
  real(chm_real),allocatable,dimension(:,:) :: W2
  real(chm_real),allocatable,dimension(:,:) :: DW
  real(chm_real),allocatable,dimension(:,:) :: PHIK2
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !     Local variables
  !     ---------------

  !     Solute label
  INTEGER UA,UB,IUA,IUB,UUAB,IOFF
  INTEGER IUNIT
  real(chm_real) CHMPOT,ENERG,CAVIT,TDS
  real(chm_real) RFIRST,RLAST
  LOGICAL QVERB,QCHM,QENER,QCAV,QFLUC,PLT2

  !     Local Heap management
  !!      INTEGER IOFF1,IOFF2

  !     Input-Output units for the distribution functions
  !     -------------------------------------------------

  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  SW(1)=GTRMF(COMLYN,COMLEN,'SW(1)',ONE)
  SW(2)=GTRMF(COMLYN,COMLEN,'SW(2)',ONE)
  SW(3)=GTRMF(COMLYN,COMLEN,'SW(3)',ONE)
  SW(4)=GTRMF(COMLYN,COMLEN,'SW(4)',ONE)

  IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
  IOFF=PUV(IUA)
  UA=1+DSITV+DSITU*(IUA-1)

  IF(CHECQUE(COMLYN,'HNC')) CLOS='HNC'
  IF(CHECQUE(COMLYN,'PY'))  CLOS='PY '
  IF(CHECQUE(COMLYN,'PY2')) CLOS='PY2'
  IF(CHECQUE(COMLYN,'EXT')) CLOS='EXT'
  QVERB=CHECQUE(COMLYN,'VERBO')
  QCHM =CHECQUE(COMLYN,'CHMPOT')
  QENER=CHECQUE(COMLYN,'ENER')
  QCAV =CHECQUE(COMLYN,'CAVIT')
  QFLUC=CHECQUE(COMLYN,'FLUC')
  PLT2 =CHECQUE(COMLYN,'PLT2')

  IF(PLT2)THEN
     RFIRST=GTRMF(COMLYN,COMLEN,'FROM',R(NFIRST))
     RLAST=GTRMF(COMLYN,COMLEN,'THRU',R(NPOINT))
  ENDIF

  IF(QCHM)THEN
     !     Solute chemical potential
     call chmalloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmalloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmalloc('solvation.src','SOLVTION','F3R',DVECT,crl=F3R)
     call chmalloc('solvation.src','SOLVTION','PHIR',DVECT,NPUV(IUA),crl=PHIR)
     !
     CALL CHEM(CLOS,ATNAM,ATNAM(UA),A(IOFF),B(IOFF),C(IOFF), &
          IPUSR(1,IOFF),PHIR, &
          IPGR(1,IOFF),IPCSR(1,IOFF), &
          NSITV,NSITU(IUA),NPUV(IUA),INTUV(1,1,IUA),KBT,RHO, &
          SW,IUNIT,QVERB, &
          CHMPOT,F1R,F2R,F3R, &
          PLT2,RFIRST,RLAST)
     !
     call chmdealloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmdealloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmdealloc('solvation.src','SOLVTION','F3R',DVECT,crl=F3R)
     call chmdealloc('solvation.src','SOLVTION','PHIR',DVECT,NPUV(IUA),crl=PHIR)
  ENDIF

  IF(QENER)THEN
     !     Solute-solvent interaction energy
     call chmalloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmalloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmalloc('solvation.src','SOLVTION','PHIR',DVECT,NPUV(IUA),crl=PHIR)
     !
     CALL SOLENER(A(IOFF),B(IOFF),C(IOFF), &
          IPUSR(1,IOFF),PHIR,IPGR(1,IOFF), &
          NSITV,NSITU(IUA),NPUV(IUA),INTUV(1,1,IUA),KBT,RHO,SW, &
          IUNIT,ENERG,F1R,F2R, &
          PLT2,RFIRST,RLAST,QVERB,ATNAM(UA))
     !
     call chmdealloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmdealloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmdealloc('solvation.src','SOLVTION','PHIR',DVECT,NPUV(IUA),crl=PHIR)
  ENDIF

  IF(QCAV)THEN
     !     Solvent-solvent cavity formation by the solute iua
     call chmalloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmalloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmalloc('solvation.src','SOLVTION','PHIR',DVECT,NPVV,crl=PHIR)
     !
     CALL CAVITY(A,B,C,IPUSR,PHIR, &
          IPDGR(1,NPRVV*(IUA-1)+1), &
          NSITV,NPVV,INTVV,KBT,RHO,SW,ADIE, &
          CAVIT,F1R,F2R,IUNIT)
     !
     call chmdealloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmdealloc('solvation.src','SOLVTION','F2R',DVECT,crl=F2R)
     call chmdealloc('solvation.src','SOLVTION','PHIR',DVECT,NPVV,crl=PHIR)
  ENDIF

  IF(QFLUC)THEN
     !     change in the chemical potential due to fluctuations
     !     in the structure
     call chmalloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmalloc('solvation.src','SOLVTION','W1',DVECT,NPU(IUA),crl=W1)
     call chmalloc('solvation.src','SOLVTION','W2',DVECT,NPU(IUA),crl=W2)
     call chmalloc('solvation.src','SOLVTION','DW',DVECT,NPU(IUA),crl=DW)
     call chmalloc('solvation.src','SOLVTION','PHIK2',DVECT,NPUV(IUA),crl=PHIK2)
     !
     CALL FLUC(CLOS,ATNAM(UA),SEGMID(UA),C(IOFF),W1, &
          W2,DW, &
          IPCSK(1,IOFF),IPXVVK, &
          PHIK2,X(UA),Y(UA),Z(UA), &
          X2(UA),Y2(UA),Z2(UA),NSITV,NSITU(IUA),NPUV(IUA), &
          INTV,INTU(1,1,IUA),INTUV(1,1,IUA), &
          KBT,RHO,SW,IUNIT,QVERB, &
          F1R,PLT2,RFIRST,RLAST)
     !
     call chmdealloc('solvation.src','SOLVTION','F1R',DVECT,crl=F1R)
     call chmdealloc('solvation.src','SOLVTION','W1',DVECT,NPU(IUA),crl=W1)
     call chmdealloc('solvation.src','SOLVTION','W2',DVECT,NPU(IUA),crl=W2)
     call chmdealloc('solvation.src','SOLVTION','DW',DVECT,NPU(IUA),crl=DW)
     call chmdealloc('solvation.src','SOLVTION','PHIK2',DVECT,NPUV(IUA),crl=PHIK2)
  ENDIF

  !     if all 3 quantities are calculated the entropy of solvation
  !     can be obtained da = de - Tds
  IF(QCHM.AND.QENER.AND.QCAV)THEN
     TDS=ENERG+CAVIT-CHMPOT
     WRITE(IUNIT,100) ENERG+CAVIT,TDS
100  FORMAT(' Solvation energy dE = ',F12.5,/,' Entropy TdS = ',F12.5)
  ENDIF

  RETURN
END SUBROUTINE SOLVTION

SUBROUTINE FLUC(CLOS,ATNAMU,SEGIDA,C,W1,W2,DW, &
     CSK,XVVK,PHIK,X1,Y1,Z1, &
     X2,Y2,Z2,NSITV,NSITU,NPUV,INTV,INTU,INTUV, &
     KBT,RHO,SW,IUNIT,QVERB, &
     F1R,PLT2,RFIRST,RLAST)
  !-----------------------------------------------------------------------
  !     This subroutine calculates the change in chemical potential
  !     that is caused by a change from a reference to a new structure
  !     of the solute.                    ---------      ---
  !     The change in chem. pot. can be approximated by an equation
  !     that uses the solvent susceptibility xvvk (invariant under
  !     structural changes of the solute in the limit of infinite
  !     dilution), the solute-solvent direct correlation function
  !     that corresponds to the reference solute structure and
  !     has resulted from a full RISM calculation, and the
  !     intramolecular matricies of the reference and new
  !     structure. Thus, a full RISM calculation of the direct
  !     correlation function cs(r) for the new structure is not
  !     needed in this approximation

  use fft
  use stream
  use number
  use rism
  use chm_kinds
  implicit none

  CHARACTER(len=3) CLOS
  CHARACTER(len=6) ATNAMU(*),SEGIDA(*)

  !     Site-site potential parameters
  real(chm_real) C(*)

  !     Distribution function
  real(chm_real) CSK(DVECT,*),XVVK(DVECT,*),PHIK(DVECT,*)
  real(chm_real) W1(DVECT,*),W2(DVECT,*),DW(DVECT,*)
  real(chm_real) X1(*),Y1(*),Z1(*),X2(*),Y2(*),Z2(*)

  !     Pointers
  INTEGER NSITV,NSITU,NPUV,INTU(DSITU,DSITU),INTUV(DSITU,DSITV)
  INTEGER INTV(DSITV,DSITV)

  !     Control variables
  real(chm_real) KBT,RHO(*),SW(*)
  INTEGER IUNIT
  LOGICAL QVERB,PLT2
  real(chm_real) RFIRST,RLAST

  !     Local variables
  real(chm_real) FACT,FLUCHMPOT
  real(chm_real) SUM1(DPAIR),SMTOT1
  real(chm_real) INTG3D,INTG1D
  EXTERNAL INTG3D,INTG1D
  LOGICAL QEND
  real(chm_real) F1R(*)
  real(chm_real) F2T1
  INTEGER IK,IP,I1,I2,I3,I4,I,J

  !     Construct the intramolecular site-site matrices for the
  !     two different conformations
  !     ---------------------------------------------------------
  CALL MKOMEG(X1,Y1,Z1,SEGIDA,NSITU,INTU,DSITU, &
       W1,W1,'NOINV')
  CALL MKOMEG(X2,Y2,Z2,SEGIDA,NSITU,INTU,DSITU, &
       W2,W2,'NOINV')
  DO IK=NFIRST,NPOINT
     DO I=1,NSITU
        DO J=1,I
           DW(IK,INTU(I,J))=W2(IK,INTU(I,J))-W1(IK,INTU(I,J))
           DW(IK,INTU(J,I))=DW(IK,INTU(I,J))
        ENDDO
     ENDDO
  ENDDO
  !
  WRITE(IUNIT,100)
100 FORMAT(/,6X,' Change in chemical potential',/)

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     WRITE(IUNIT,101) CLOS
101  FORMAT(' closure: ',A3)
     CALL MKPHIK(NPUV,C,ONE,SW(1),SW(3),KBT,PHIK)
     !     Calculate the  term -cuv*xvv*cvu*dwu'u , u<u'
     IP=0
     DO I1=1,NSITU
        DO I4=1,I1-1
           IP=IP+1
           DO IK=NFIRST,NPOINT
              F1R(IK)=ZERO
              DO I2=1,NSITV
                 DO I3=1,NSITV
                    !
                    ! NOTICE: XVVK IS MULTIPLIED BY RHO(J)
                    !
                    F1R(IK)=F1R(IK)+ &
                         (CSK(IK,INTUV(I1,I2))+PHIK(IK,INTUV(I1,I2)))* &
                         XVVK(IK,INTV(I2,I3))* &
                         (CSK(IK,INTUV(I4,I3))+PHIK(IK,INTUV(I4,I3)))* &
                         DW(IK,INTU(I4,I1))
                 ENDDO
              ENDDO
           ENDDO
           SUM1(IP)=INTG3D(F1R,NFIRST,NPOINT)
           SUM1(IP)=-KBT*TPM32*TPM32*SUM1(IP)
        ENDDO
     ENDDO

     IF(QVERB)THEN
        CALL VERBOS2(IUNIT,SUM1,NSITU,X1,Y1,Z1,X2,Y2,Z2, &
             ATNAMU)
     ENDIF

     FLUCHMPOT=0.0
     DO IK=1,IP
        FLUCHMPOT=FLUCHMPOT+SUM1(IK)
     ENDDO
     WRITE(IUNIT,109) FLUCHMPOT
109  FORMAT(/,' change in chemical potential due to ', &
          'structural changes =',F10.5)
  ENDIF

  RETURN
END SUBROUTINE FLUC

SUBROUTINE CHEM(CLOS,ATNAMV,ATNAMU,A,B,C, &
     USR,PHIR,GR,CSR,NSITV,NSITU,NPUV,INTUV, &
     KBT,RHO,SW,IUNIT,QVERB,CHMPOT,F1R,F2R,F3R, &
     PLT2,RFIRST,RLAST)
  !-----------------------------------------------------------------------
  !     This subroutine uses the HNC closed form of the thermodynamic
  !     integration and gives the chemical potential of the solute in
  !     the solvent.  It is assumed that the subroutine cycles has been
  !     called previously and that the calculation is converged.
  !     The solute can be one solvent molecule (see the call in SOLVATION).
  !     Reference: S.J. Singer and D. Chandler, Mol. Phys., 55, 621 (1985).

  use fft
  use stream
  use number
  use rism
  use chm_kinds
  implicit none

  CHARACTER(len=3) CLOS
  CHARACTER(len=6) ATNAMV(*),ATNAMU(*)

  !     Site-site potential parameters
  real(chm_real) A(*),B(*),C(*)

  !     Distribution function
  real(chm_real) USR(DVECT,*),PHIR(DVECT,*)
  real(chm_real) CSR(DVECT,*),GR(DVECT,*)

  !     Pointers
  INTEGER NSITV,NSITU,NPUV,INTUV(DSITU,DSITV)

  !     Control variables
  INTEGER IUNIT
  LOGICAL QVERB,PLT2
  real(chm_real) KBT,RHO(*),SW(*)
  real(chm_real) CHMPOT,F1R(*),F2R(*),F3R(*)
  real(chm_real) RFIRST,RLAST
  real(chm_real) INTG3D,INTG1D
  EXTERNAL INTG3D,INTG1D

  !     Local variables
  real(chm_real) RM1,RM6,RM12,FACT
  real(chm_real) SUM1(DPAIR),SUM2(DPAIR),SUM3(DPAIR),SUMTOT(DPAIR), &
       SMTOT1,SMTOT2,SMTOT3
  LOGICAL QEND
  real(chm_real) F2T1,F2T2,F2T3
  INTEGER I,J,IFIRST,ILAST,ICOUNT,IP,IR

  WRITE(IUNIT,100)
100 FORMAT(/,6X,'Chemical potential')

  IF(CLOS.EQ.'HNC')THEN
     !     ---------------------
     WRITE(IUNIT,101) CLOS
101  FORMAT(' closure: ',A3)
     CALL MKPHIR(NPUV,C,ONE,SW(1),SW(3),KBT,PHIR)
     DO IP=1,NPUV
        DO IR=NFIRST,NPOINT
           !     Calculate the 3 terms .5*h(r)**2 - Ctot - .5*h(r)*Ctot(r)
           F1R(IR)=HALF*KBT*(GR(IR,IP)-ONE)*(GR(IR,IP)-ONE)
           F2R(IR)=-KBT*CSR(IR,IP)
           F3R(IR)=-HALF*KBT*(GR(IR,IP)-ONE)*(CSR(IR,IP)+PHIR(IR,IP))
        ENDDO
        SUM1(IP)=INTG3D(F1R,NFIRST,NPOINT)
        SUM2(IP)=INTG3D(F2R,NFIRST,NPOINT)
        SUM3(IP)=INTG3D(F3R,NFIRST,NPOINT)
        SUMTOT(IP)=SUM1(IP)+SUM2(IP)+SUM3(IP)
     ENDDO

     IF(QVERB)THEN
        CALL VERBOS(IUNIT,RHO,SUMTOT,INTUV,NSITV,NSITU, &
             ATNAMV,ATNAMU)
     ENDIF

     SMTOT1=0.0
     SMTOT2=0.0
     SMTOT3=0.0
     DO I=1,NSITU
        DO J=1,NSITV
           SMTOT1=SMTOT1+SUM1(INTUV(I,J))*RHO(J)
           SMTOT2=SMTOT2+SUM2(INTUV(I,J))*RHO(J)
           SMTOT3=SMTOT3+SUM3(INTUV(I,J))*RHO(J)
        ENDDO
     ENDDO
     CHMPOT=SMTOT1+SMTOT2+SMTOT3
     WRITE(IUNIT,102) CHMPOT,SMTOT1,SMTOT2,SMTOT3
102  FORMAT(' chemical potential =',F12.5,/, &
          '      +1/2 H(R)H(R) =',F12.5,/, &
          '         - C(R)     =',F12.5,/, &
          '      -1/2 H(R)C(R) =',F12.5)

     IF(PLT2)THEN
        WRITE(IUNIT,'(/,1X,A)') 'Chem. pot. integrands in real space:'
        WRITE(IUNIT,150)
150     FORMAT(1X,'-----------------------------------',//, &
             9X,'r(i)',5X,'total*r**2',5X, &
             '1/2 HH',9X,'- C',7X,'-1/2 HC',/)
        IFIRST=0
        ILAST=0
        loop14:DO IR=NFIRST,NPOINT
           F1R(IR)=0.0
           F2T1=0.0
           F2T2=0.0
           F2T3=0.0
           IF((R(IR).GE.RFIRST).AND.(R(IR).LE.RLAST))THEN
              IF(IFIRST.EQ.0) IFIRST=IR
              ILAST=IR
              DO I=1,NSITU
                 DO J=1,NSITV
                    IP=INTUV(I,J)
                    F2R(1)=HALF*KBT*(GR(IR,IP)-ONE)*(GR(IR,IP)-ONE)
                    F2R(2)=-KBT*CSR(IR,IP)
                    F2R(3)=-HALF*KBT*(GR(IR,IP)-ONE)*(CSR(IR,IP)+PHIR(IR,IP))
                    F1R(IR)=F1R(IR)+(F2R(1)+F2R(2)+F2R(3))*RHO(J)
                    F2T1=F2T1+F2R(1)
                    F2T2=F2T2+F2R(2)
                    F2T3=F2T3+F2R(3)
                 ENDDO
              ENDDO
              WRITE(IUNIT,'(2X,5(F12.5,1X))') R(IR),F1R(IR)*R(IR)**2, &
                   F2T1,F2T2,F2T3
           ENDIF
        enddo loop14
        !     calculate the contribution of this segment
        !
        SMTOT1=INTG3D(F1R,IFIRST,ILAST)
        WRITE(IUNIT,'(/,1X,A,F12.5)') 'Contribution: ',SMTOT1
     ENDIF
     !
     !     else for the other closures
  ELSE
     !     ----
     WRITE(OUTU,111) CLOS
111  FORMAT(4X,'Chemical potential not calculated for closure', &
          2X,A4,//,4X,'try with HNC')
  ENDIF

  RETURN
END SUBROUTINE CHEM

SUBROUTINE VERBOS(IUNIT,RHO,SUMTOT,INTUV,NSITV,NSITU, &
     ATNAMV,ATNAMU)
  !-----------------------------------------------------------------------
  !     This subroutine breaks down the chemical potential to all
  !     site-site contributions
  use rism
  use chm_kinds
  implicit none
  INTEGER IUNIT
  real(chm_real) SUMTOT(*),RHO(*)
  INTEGER INTUV(DSITU,DSITV),NSITV,NSITU
  CHARACTER(len=6) ATNAMV(*),ATNAMU(*)

  !     Local variables
  real(chm_real) SUMV(DPAIR),SUMU(DPAIR),SUMVU,SUM
  CHARACTER(len=1) LINE(80)
  INTEGER NLINE,ILINE,I,J

  NLINE=16*(NSITV+1)
  LINE(1:nline)='-'

  DO I=1,NSITU
     SUMV(I)=0.0
     DO J=1,NSITV
        SUMV(I)=SUMV(I)+SUMTOT(INTUV(I,J))*RHO(J)
     ENDDO
  ENDDO

  DO I=1,NSITV
     SUMU(I)=0.0
     DO J=1,NSITU
        SUMU(I)=SUMU(I)+RHO(I)*SUMTOT(INTUV(J,I))
     ENDDO
  ENDDO

  !     Print the results
  !     -----------------

  WRITE(IUNIT,100) (ATNAMV(I),I=1,NSITV)
100 FORMAT(/,6X,25(12X,A4))
  SUM=0.0
  DO I=1,NSITU
     SUM=SUM+SUMV(I)
     WRITE(IUNIT,101) ATNAMU(I), &
          (RHO(J)*SUMTOT(INTUV(I,J)),J=1,NSITV),SUMV(I)
101  FORMAT(1X,A4,1X,25F16.5)
  ENDDO
  WRITE(IUNIT,'(6X,80A)') (LINE(ILINE),ILINE=1,NLINE)
  WRITE(IUNIT,102) (SUMU(J),J=1,NSITV),SUM
102 FORMAT(6X,25F16.5)
  RETURN
END SUBROUTINE VERBOS

SUBROUTINE VERBOS2(IUNIT,SUMTOT,NSITU,X1,Y1,Z1,X2,Y2,Z2, &
     ATNAMU)
  !-----------------------------------------------------------------------
  !     This subroutine breaks down the change in chemical potential
  !     due to structural changes to the various atom-atom contributions
  !     This information can be used to determine which atom pairs
  !     are mostly responsible for the change in chemical potential.

  use rism
  use chm_kinds
  implicit none
  INTEGER IUNIT
  real(chm_real) SUMTOT(*)
  real(chm_real) X1(*),Y1(*),Z1(*),X2(*),Y2(*),Z2(*)
  INTEGER NSITU
  CHARACTER(len=6) ATNAMU(*)

  !     Local variables
  real(chm_real) SUM,DIJ1,DIJ2,DIJ
  CHARACTER(len=1) LINE(80)
  INTEGER I,J,IP,NLINE,ILINE

  NLINE=50
  DO ILINE=1,NLINE
     LINE(ILINE)='-'
  ENDDO
  WRITE(IUNIT,105)
105 FORMAT(2X,'atom 1',3X,'atom 2',5X,'chem. pot. change',2X, &
       'distance change',/)


  !     Print the results
  !     -----------------

  IP=0
  SUM=0.0
  DO I=1,NSITU
     DO J=1,I-1
        IP=IP+1
        SUM=SUM+SUMTOT(IP)
        DIJ1=SQRT((X1(I)-X1(J))**2+(Y1(I)-Y1(J))**2+(Z1(I)-Z1(J))**2)
        DIJ2=SQRT((X2(I)-X2(J))**2+(Y2(I)-Y2(J))**2+(Z2(I)-Z2(J))**2)
        DIJ=DIJ2-DIJ1
        WRITE(IUNIT,107) I,ATNAMU(I),J,ATNAMU(J),SUMTOT(IP),DIJ
107     FORMAT(I3,1X,A4,1X,I3,1X,A4,4X,F16.5,2X,F16.5)
     ENDDO
  ENDDO
  WRITE(IUNIT,'(2X,80A)') (LINE(ILINE),ILINE=1,NLINE)
  WRITE(IUNIT,108) SUM
108 FORMAT(21X,F16.5)
  RETURN
END SUBROUTINE VERBOS2

SUBROUTINE SOLENER(A,B,C,USR,PHIR, &
     GR,NSITV,NSITU,NPUV,INTUV,KBT,RHO,SW,IUNIT, &
     ENERG,F1R,F2R,PLT2,RFIRST,RLAST,QVERB,ATNAMU)
  !-----------------------------------------------------------------------
  !     This subroutine calculates the interaction energy
  !     between the solute and the solvent. This energy is a part of
  !     the total energy of solvation of the solute inside the solvent.
  !     The other contribution to the solvation energy comes from the
  !     solvent responce and is calculated by the subroutine CAVITY
  !
  use fft
  use number
  use rism
  use chm_kinds
  implicit none

  real(chm_real) A(*),B(*),C(*),USR(DVECT,*),PHIR(DVECT,*),GR(DVECT,*)
  INTEGER NSITV,NSITU,INTUV(DSITU,DSITV),NPUV
  real(chm_real) KBT,SW(*),RHO(*)
  INTEGER IUNIT
  real(chm_real) ENERG,F1R(*),F2R(*)
  LOGICAL PLT2
  real(chm_real) RFIRST,RLAST
  LOGICAL QVERB
  CHARACTER(len=6) ATNAMU(*)
  real(chm_real) INTG3D
  EXTERNAL INTG3D

  !     Local variables
  real(chm_real) ELEC,CORE,ENERG1,CORE1,ELEC1
  real(chm_real) SMTOT1
  INTEGER I,J,IP,IR,IFIRST,ILAST

  ELEC=0.0
  CORE=0.0
  ENERG=0.0

  WRITE(IUNIT,100)
100 FORMAT(/,6X,'Interaction energy')
  IF(QVERB)THEN
     WRITE(IUNIT,'(A)') ' Site   <energy>     <elec>    <core>'
  ENDIF

  CALL MKUSR(NPUV,A,B,SW(1),SW(2),KBT,USR)
  CALL MKPHIR(NPUV,C,ONE,SW(1),SW(3),KBT,PHIR)

  loop10:DO I=1,NSITU
     DO J=1,NSITV
        IP=INTUV(I,J)
        IF(J.EQ.1)THEN
           DO IR=NFIRST,NPOINT
              F1R(IR)= KBT*GR(IR,IP)*USR(IR,IP)*RHO(J)
              F2R(IR)=-KBT*GR(IR,IP)*PHIR(IR,IP)*RHO(J)
           ENDDO
        ELSE
           DO IR=NFIRST,NPOINT
              F1R(IR)=F1R(IR)+KBT*GR(IR,IP)*USR(IR,IP)*RHO(J)
              F2R(IR)=F2R(IR)-KBT*GR(IR,IP)*PHIR(IR,IP)*RHO(J)
           ENDDO
        ENDIF
     ENDDO
     CORE1=INTG3D(F1R,NFIRST,NPOINT)
     ELEC1=INTG3D(F2R,NFIRST,NPOINT)
     ENERG1=ELEC1+CORE1
     IF(QVERB)THEN
        WRITE(IUNIT,102) ATNAMU(I),ENERG1,ELEC1,CORE1
102     FORMAT(1X,A,3F12.5)
     ENDIF
     ELEC=ELEC+ELEC1
     CORE=CORE+CORE1
     ENERG=ENERG+ENERG1
  enddo loop10
  WRITE(IUNIT,101) ELEC,CORE,ENERG
101 FORMAT(/,' electrostatics ',F12.5,/, &
       ' core forces    ',F12.5,/, &
       ' total          ',F12.5)

  IF(PLT2)THEN
     IFIRST=0
     ILAST=0
     WRITE(IUNIT,'(/,1X,A)') &
          'interaction energy integrands in real space:'
     WRITE(IUNIT,150)
150  FORMAT(1X,'----------------------------------------------',//, &
          8X,'r(i)',6X,'total*r**2',7X,'elec',7X,'core',/)
     DO IR=NFIRST,NPOINT
        F1R(IR)=0.0
        F2R(1)=0.0
        F2R(2)=0.0
        IF((R(IR).GE.RFIRST).AND.(R(IR).LE.RLAST))THEN
           IF(IFIRST.EQ.0) IFIRST=IR
           ILAST=IR
           DO I=1,NSITU
              DO J=1,NSITV
                 IP=INTUV(I,J)
                 F2R(1)=F2R(1)-KBT*PHIR(IR,IP)*GR(IR,IP)*RHO(J)
                 F2R(2)=F2R(2)+KBT*USR(IR,IP)*GR(IR,IP)*RHO(J)
              ENDDO
           ENDDO
           F1R(IR)=F2R(1)+F2R(2)
           WRITE(IUNIT,'(2X,5(F12.5,1X))') R(IR),F1R(IR)*R(IR)**2, &
                F2R(1),F2R(2)
        ENDIF
     ENDDO
     !
     !     calculate the contribution of this segment
     !
     SMTOT1=INTG3D(F1R,IFIRST,ILAST)
     WRITE(IUNIT,'(/,1X,A,F12.5)') 'Contribution: ',SMTOT1
  ENDIF

  RETURN
END SUBROUTINE SOLENER

SUBROUTINE CAVITY(A,B,C,USR,PHIR,DGR, &
     NSITV,NPVV,INTVV,KBT,RHO,SW,ADIE,CAVIT,F1R,F2R,IUNIT)
  !-----------------------------------------------------------------------
  !     This subroutine calculates the energy due to cavity formation
  !     in the solvent that accompagnies solvation.
  use fft
  use number
  use rism
  use chm_kinds
  implicit none

  real(chm_real) A(*),B(*),C(*),USR(DVECT,*),PHIR(DVECT,*),DGR(DVECT,*)
  INTEGER NSITV,NPVV,INTVV(DSITV,DSITV)
  real(chm_real) KBT,RHO(*),SW(*),ADIE
  real(chm_real) CAVIT,F1R(*),F2R(*)
  INTEGER IUNIT
  real(chm_real) INTG3D
  EXTERNAL INTG3D

  !     Local variables
  real(chm_real) ELEC,CORE
  INTEGER IR,I,J,IP

  WRITE(IUNIT,100)
100 FORMAT(/,6X,'Cavity formation energy')

  CALL MKUSR(NPVV,A,B,SW(1),SW(2),KBT,USR)
  CALL MKPHIR(NPVV,C,ONE,SW(1),SW(3),KBT,PHIR)

  DO IR=NFIRST,NPOINT
     F1R(IR)=0.0
     F2R(IR)=0.0
     DO I=1,NSITV
        DO J=1,NSITV
           IP=INTVV(I,J)
           F1R(IR)=F1R(IR)+KBT*DGR(IR,IP)*USR(IR,IP)*RHO(I)*RHO(J)*HALF
           F2R(IR)=F2R(IR)-KBT*DGR(IR,IP)*PHIR(IR,IP)*RHO(I)*RHO(J)*HALF
        ENDDO
     ENDDO
  ENDDO
  CORE=INTG3D(F1R,NFIRST,NPOINT)
  ELEC=INTG3D(F2R,NFIRST,NPOINT)
  CAVIT=ELEC+CORE
  WRITE(IUNIT,101) ELEC,CORE,CAVIT
101 FORMAT(' electrostatics ',F12.5,/, &
       ' core forces    ',F12.5,/, &
       ' total          ',F12.5)
  RETURN
END SUBROUTINE CAVITY

#endif /* (rism_main)*/


