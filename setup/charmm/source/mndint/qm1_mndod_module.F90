module qm1_mndod
  use chm_kinds
  use number
  use qm1_constant
  !
  contains

#if KEY_MNDO97==1 /*mndo97*/
  subroutine reppd_qmqm(NI,NJ,R,RI,CORE_mat,W,LIMIJ,LIMKL)
  !
  ! Two-center two-electron repulsion integrals and two-center one-electron
  ! attractions in local coordinates. (General version for MNDO/d and AM1/d).
  ! 
  ! The two-electron integrals are returned as W(LIMKL,LIMIJ) using standard
  ! pair indices ij=(i*(i-1))/2+j for addressing.  Note that the orbital order
  ! for integral evaluation applied:
  ! AO     1     2     3     4     5     6     7     8     9
  ! L      0     1     1     1     2     2     2     2     2
  ! M      0     0     1    -1     0     1    -1     2    -2
  ! TYPE   S    PZ    PX    PY   DZ2   DXZ   DYZ DX2Y2   DXY
  ! 
  ! INPUT 
  ! NI       atomic number of atom A (orbitals i and j, pairs ij).
  ! NJ       atomic number of atom B (orbitals k and l, pairs kl).
  ! R        interatomic distance (au).
  ! RI       local two-electron integrals for SP basis.
  ! CORE     local one-electron integrals for SP basis.
  ! LIMIJ    column dimension of W. 1 for S, 10 for SP, 45 for SPD. 
  ! LIMKL    row    dimension of W. 1 for S, 10 for SP, 45 for SPD.
  !      both LIMIJ and LIMKL are iorbs*(iorbs-1)/2+iorbs and
  !                               jorbs*(jorbs-1)/2+jorbs.
  ! 
  ! On output
  ! CORE_mat complete set of local one-electron integrals.
  ! W        complete set of local two-electron integrals.
  !
  !!!! for MM charges
  !!!! Convention: ni=0 or nj=0 denotes an external point charge without basis
  !!!!             orbitals.  The charge is 1 atomic unit. Thus, the values of
  !!!!             DD(i,0) and PO(i,0) are defined to be zero.
  ! 
  ! charge separations DD(6,LMZ) and additive terms PO(9,LMZ) are from parameters.  
  ! inedx of DD and PO : SS 1, SP 2, PP 3, SD 4, PD 5, DD 6.
  ! multipole          :  L=0,  L=1,  L=2,  L=2,  L=1,  L=2.
  ! special index of PO: PP 7, DD 8.
  ! multipole          :  L=0,  L=0.
  ! for atomic core    : additive term PO(9,ni)
  !
  ! Note that DD and PO are defined in accordance with the TCA92 paper, with the
  ! following exception:  DD(4,n) and DD(6,n) contain an extra factor of SQRT2
  ! to smpliify the calcualtions involving the square quadrupoles (L=2).  such a
  ! factor would also be helpful in DD(3,n), but would conflict with the code
  ! for the TCA77 multipoles in subroutine REPP.  therefore, the extra factor
  ! SQRT2 is included explicitly for DD(3,n) in this routine (see below).
  !
  ! The coefficients relating analytical and point-charge multipole mements, see
  ! eq. (17) and Table 2 of TCA paper, are given below as parameter statements
  ! (variables CLM..) where the digits in CLM.. refer to the standard pair index
  ! (see above).  the current definitions include the signs of the coefficients
  ! that are missing in Table 2 of the TCA paper.  only those nonzero coefficients 
  ! are defined which are needed and whose absolute value is not equal to 1.
  !
  use qm1_parameters, only : CORE,DD,PO

  implicit none

  integer :: NI,NJ,LIMIJ,LIMKL
  real(chm_real):: W(LIMKL,LIMIJ),RI(22),CORE_mat(10,2)
  real(chm_real):: R

  ! local variables:
  real(chm_real),parameter :: CLM3  = 0.13333333333333D+01,  &
                              CLM6  =-0.66666666666667D+00,  &
                              CLM10 =-0.66666666666667D+00,  &
                              CLM11 = 0.11547005383793D+01,  &
                              CLM12 = 0.11547005383793D+01,  &
                              CLM13 =-0.57735026918963D+00,  &
                              CLM15 = 0.13333333333333D+01,  &
                              CLM20 = 0.57735026918963D+00,  &
                              CLM21 = 0.66666666666667D+00,  &
                              CLM28 = 0.66666666666667D+00,  &
                              CLM33 =-0.11547005383793D+01,  &
                              CLM36 =-0.13333333333333D+01,  &
                              CLM45 =-0.13333333333333D+01,  &
                              SQRT2 = 0.14142135623731D+01,  &
                              small = 1.0D-06
  real(chm_real),parameter :: r_SQRT2 = one/SQRT2
  integer, parameter :: LCW(10)=(/ 1,2,3,6,11,12,15,18,21,36/)

  real(chm_real):: X(405),XM(14),factor
  integer       :: LX(6,6)
  integer       :: I,J,K,L,N,IJ,KL,LIJ,IFIRST,ILAST,NC,NE,LE,LIJMAX,LKLMAX,ICMAX
  real(chm_real):: R2,SIG,AA,AB,CORENC,DA,QA,DB,QB,QAS,QBS,DA2,  &
                   ADD,ADDAS,ADDBS,ADDSS,RMDA2,RPDA2,XYXY,ZZZZ,  &
                   DABP,DABM,DABP2,DABM2,RPAB2,RMAB2,RPDB2,RMDB2


  ! initialization.
  w(1:LIMKL,1:LIMIJ)=zero
  if(LIMIJ == 1) then        ! iorbs=1
     LIJMAX = 1
  else if(LIMIJ == 10) then  ! iorbs=4
     LIJMAX = 3
  else                       ! iorbs=9
     LIJMAX = 6
  end if
  if(LIMKL == 1) then        ! jorbs=1
     LKLMAX = 1
  else if(LIMKL == 10) then  ! jorbs=4
     LKLMAX = 3
  else                       ! jorbs=9
     LKLMAX = 6
  end if
  ! at least one of LIJMAX and LKLMAX should be "6".

  ! distance between point charges.
  R2     = R*R
  I      = 0
  loopLIJ1: do LIJ=1,LIJMAX
     DA     = DD(LIJ,ni)
     QA     = PO(LIJ,ni)
     if(LIJ == 3) then
        QAS = PO(7,ni)
        DA  = DA*SQRT2
     else if(LIJ == 6) then
        QAS = PO(8,ni)
     else
        QAS = PO(1,ni)
     end if
     DA2    = DA*DA
     RMDA2  = (R-DA)**2
     RPDA2  = (R+DA)**2

     !(SS,KL), LIJ=1.
     if(LIJ == 1) then
        if(LKLMAX > 3) then
           ! KL=DS.
           LX(4,LIJ) = I
           DB     = DD(4,NJ)
           QB     = PO(4,NJ)
           ADD    = (QA+QB)**2
           ! L1=0, L2=2, M=0, LABEL=6.
           X(I+1) = (R-DB)**2+ADD
           X(I+2) = R2+DB**2+ADD
           X(I+3) = (R+DB)**2+ADD
           I      = I+3
           ! KL=DP.
           LX(5,LIJ) = I
           DB     = DD(5,NJ)
           QB     = PO(5,NJ)
           ADD    = (QA+QB)**2
           ! L1=0, L2=1, M=0, LABEL=3.
           X(I+1) = (R+DB)**2+ADD
           X(I+2) = (R-DB)**2+ADD
           I      = I+2
           ! KL=DD.
           LX(6,LIJ) = I
           DB     = DD(6,NJ)
           QB     = PO(6,NJ)
           QBS    = PO(8,NJ)
           ADD    = (QA+QB)**2
           ! L1=0, L2=0, M=0, LABEL=1.
           X(I+1) = R2+(QA+QBS)**2
           ! L1=0, L2=2, M=0, LABEL=6.
           X(I+2) = (R-DB)**2+ADD
           X(I+3) = R2+DB**2+ADD
           X(I+4) = (R+DB)**2+ADD
           I      = I+4
        end if
     ! end of LIJ.eq.1

     ! (PS,KL), LIJ=2, OR (DP,KL), LIJ=5.
     else if(LIJ == 2 .or. LIJ == 5) then
        if(LIJ == 5) then
           ! KL=SS.
           LX(1,LIJ) = I
           QB     = PO(1,NJ)
           ADD    = (QA+QB)**2
           ! L1=1, L2=0, M=0, LABEL=2.
           X(I+1) = RPDA2+ADD
           X(I+2) = RMDA2+ADD
           I      = I+2
           if(LKLMAX > 1) then
              ! KL=PS.
              LX(2,LIJ) = I
              DB     = DD(2,NJ)
              QB     = PO(2,NJ)
              ADD    = (QA+QB)**2
              DABP   = DA+DB
              DABM   = DA-DB
              ! L1=1, L2=1, M=0, LABEL=4.
              X(I+1) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
              X(I+2) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
              X(I+3) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
              X(I+4) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
              ! L1=1, L2=1, M=1, LABEL=5.
              X(I+5) = R2 + DABM**2 + ADD  ! R2+(DA-DB)**2+ADD
              X(I+6) = R2 + DABP**2 + ADD  ! R2+(DA+DB)**2+ADD
              I      = I+6
              ! KL=PP.
              LX(3,LIJ) = I
              DB     = DD(3,NJ)*SQRT2
              QB     = PO(3,NJ)
              QBS    = PO(7,NJ)
              ADD    = (QA+QB)**2
              ADDBS  = (QA+QBS)**2
              DABP   = DA+DB
              DABM   = DA-DB
              ! L1=1, L2=0, M=0, LABEL=2.
              X(I+1) = RPDA2+ADDBS
              X(I+2) = RMDA2+ADDBS
              ! L1=1, L2=2, M=0, LABEL=8.
              X(I+3) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
              X(I+4) = RMDA2+DB**2 + ADD
              X(I+5) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
              X(I+6) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
              X(I+7) = RPDA2+DB**2+ADD
              X(I+8) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
              ! L1=1, L2=2, M=1, LABEL=11.
              AB     = DB*r_SQRT2  ! /SQRT2
              RPAB2  = (R+AB)**2
              RMAB2  = (R-AB)**2
              DABP2  = (DA+AB)**2
              DABM2  = (DA-AB)**2
              X(I+9) = RMAB2 + DABM2 + ADD ! (R-AB)**2+(DA-AB)**2+ADD
              X(I+10)= RPAB2 + DABM2 + ADD ! (R+AB)**2+(DA-AB)**2+ADD
              X(I+11)= RMAB2 + DABP2 + ADD ! (R-AB)**2+(DA+AB)**2+ADD
              X(I+12)= RPAB2 + DABP2 + ADD ! (R+AB)**2+(DA+AB)**2+ADD
              I      = I+12
           end if
        end if   ! for LIJ.eq.5
        if(LKLMAX > 3) then
           ! KL=DS.
           LX(4,LIJ) = I
           DB     = DD(4,NJ)
           QB     = PO(4,NJ)
           ADD    = (QA+QB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=1, L2=2, M=0, LABEL=8.
           X(I+1) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
           X(I+2) = RMDA2+DB**2 + ADD
           X(I+3) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
           X(I+4) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
           X(I+5) = RPDA2+DB**2 + ADD
           X(I+6) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
           ! L1=1, L2=2, M=1, LABEL=11.
           AB     = DB*r_SQRT2  ! /SQRT2
           RPAB2  = (R+AB)**2
           RMAB2  = (R-AB)**2
           DABP2  = (DA+AB)**2
           DABM2  = (DA-AB)**2
           X(I+7) = RMAB2 + DABM2 + ADD ! (R-AB)**2+(DA-AB)**2+ADD
           X(I+8) = RPAB2 + DABM2 + ADD ! (R+AB)**2+(DA-AB)**2+ADD
           X(I+9) = RMAB2 + DABP2 + ADD ! (R-AB)**2+(DA+AB)**2+ADD
           X(I+10)= RPAB2 + DABP2 + ADD ! (R+AB)**2+(DA+AB)**2+ADD
           I      = I+10
           ! KL=DP.
           LX(5,LIJ) = I
           DB     = DD(5,NJ)
           QB     = PO(5,NJ)
           ADD    = (QA+QB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=1, L2=1, M=0, LABEL=4.
           X(I+1) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
           X(I+2) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
           X(I+3) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
           X(I+4) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
           ! L1=1, L2=1, M=1, LABEL=5.
           X(I+5) = R2 + DABM**2 + ADD ! R2+(DA-DB)**2+ADD
           X(I+6) = R2 + DABP**2 + ADD ! R2+(DA+DB)**2+ADD
           I      = I+6
           ! KL=DD.
           LX(6,LIJ) = I
           DB     = DD(6,NJ)
           QB     = PO(6,NJ)
           QBS    = PO(8,NJ)
           ADD    = (QA+QB)**2
           ADDBS  = (QA+QBS)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=1, L2=0, M=0, LABEL=2.
           X(I+1) = RPDA2+ADDBS
           X(I+2) = RMDA2+ADDBS
           ! L1=1, L2=2, M=0, LABEL=8.
           X(I+3) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
           X(I+4) = RMDA2+DB**2+ADD
           X(I+5) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
           X(I+6) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
           X(I+7) = RPDA2+DB**2+ADD
           X(I+8) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
           ! L1=1, L2=2, M=1, LABEL=11.
           AB     = DB*r_SQRT2  ! /SQRT2
           RPAB2  = (R+AB)**2
           RMAB2  = (R-AB)**2
           DABP2  = (DA+AB)**2
           DABM2  = (DA-AB)**2
           X(I+9) = RMAB2 + DABM2 + ADD ! (R-AB)**2+(DA-AB)**2+ADD
           X(I+10)= RPAB2 + DABM2 + ADD ! (R+AB)**2+(DA-AB)**2+ADD
           X(I+11)= RMAB2 + DABP2 + ADD ! (R-AB)**2+(DA+AB)**2+ADD
           X(I+12)= RPAB2 + DABP2 + ADD ! (R+AB)**2+(DA+AB)**2+ADD
           I      = I+12
        end if
     ! end of LIJ.eq.2 .or. LIJ.eq.5

     ! (PP,KL), LIJ=3, OR (DD,KL), LIJ=6.
     else if(LIJ == 3 .or. LIJ == 6) then
        if(LIJ == 6) then
           ! KL=SS.
           LX(1,LIJ) = I
           QB     = PO(1,NJ)
           ADD    = (QA+QB)**2
           ADDAS  = (QAS+QB)**2
           ! L1=0, L2=0, M=0, LABEL=1.
           X(I+1) = R2+ADDAS
           ! L1=2, L2=0, M=0, LABEL=7.
           X(I+2) = RMDA2+ADD
           X(I+3) = R2+DA2+ADD
           X(I+4) = RPDA2+ADD
           I      = I+4
           if(LKLMAX > 1) then
              ! KL=PS.
              LX(2,LIJ) = I
              DB     = DD(2,NJ)
              QB     = PO(2,NJ)
              ADD    = (QA+QB)**2
              ADDAS  = (QAS+QB)**2
              DABP   = DA+DB
              DABM   = DA-DB
              RPDB2  = (R+DB)**2
              RMDB2  = (R-DB)**2
              ! L1=0, L2=1, M=0, LABEL=3.
              X(I+1) = RPDB2 + ADDAS     ! (R+DB)**2+ADDAS
              X(I+2) = RMDB2 + ADDAS     ! (R-DB)**2+ADDAS
              ! L1=2, L2=1, M=0, LABEL=9.
              X(I+3) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
              X(I+4) = RMDB2 + DA2 + ADD ! (R-DB)**2+DA2+ADD
              X(I+5) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
              X(I+6) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
              X(I+7) = RPDB2 + DA2 + ADD ! (R+DB)**2+DA2+ADD
              X(I+8) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
              ! L1=2, L2=1, M=1, LABEL=12.
              AA     = DA*r_SQRT2  ! /SQRT2
              RPAB2  = (R+AA)**2
              RMAB2  = (R-AA)**2
              DABP2  = (AA+DB)**2
              DABM2  = (AA-DB)**2
              X(I+9) = RPAB2 + DABM2 + ADD ! (R+AA)**2+(AA-DB)**2+ADD
              X(I+10)= RMAB2 + DABM2 + ADD ! (R-AA)**2+(AA-DB)**2+ADD
              X(I+11)= RPAB2 + DABP2 + ADD ! (R+AA)**2+(AA+DB)**2+ADD
              X(I+12)= RMAB2 + DABP2 + ADD ! (R-AA)**2+(AA+DB)**2+ADD
              I      = I+12
              ! KL=PP.
              LX(3,LIJ) = I
              DB     = DD(3,NJ)*SQRT2
              QB     = PO(3,NJ)
              QBS    = PO(7,NJ)
              ADD    = (QA+QB)**2
              ADDAS  = (QAS+QB)**2
              ADDBS  = (QA+QBS)**2
              ADDSS  = (QAS+QBS)**2
              RPDB2  = (R+DB)**2
              RMDB2  = (R-DB)**2
              DABP   = DA+DB
              DABM   = DA-DB
              ! L1=0, L2=0, M=0, LABEL=1.
              X(I+1) = R2+ADDSS
              ! L1=0, L2=2, M=0, LABEL=6.
              X(I+2) = RMDB2   +ADDAS  ! (R-DB)**2+ADDAS
              X(I+3) = R2+DB**2+ADDAS
              X(I+4) = RPDB2   +ADDAS  ! (R+DB)**2+ADDAS
              ! L1=2, L2=0, M=0, LABEL=7.
              X(I+5) = RMDA2+ADDBS
              X(I+6) = R2+DA2+ADDBS
              X(I+7) = RPDA2+ADDBS
              ! L1=2, L2=2, M=0, LABEL=10.
              X(I+8) = (R-DABP)**2  + ADD  ! (R-DA-DB)**2+ADD
              X(I+9) = (R-DABM)**2  + ADD  ! (R-DA+DB)**2+ADD
              X(I+10)= (R+DABM)**2  + ADD  ! (R+DA-DB)**2+ADD
              X(I+11)= (R+DABP)**2  + ADD  ! (R+DA+DB)**2+ADD
              X(I+12)= RMDA2 + DB**2+ ADD
              X(I+13)= RMDB2 + DA2  + ADD  ! (R-DB)**2+DA2+ADD
              X(I+14)= RPDA2 + DB**2+ ADD
              X(I+15)= RPDB2 + DA2  + ADD  ! (R+DB)**2+DA2+ADD
              X(I+16)= R2 + DABM**2 + ADD  ! R2+(DA-DB)**2+ADD
              X(I+17)= R2 + DABP**2 + ADD  ! R2+(DA+DB)**2+ADD
              X(I+18)= R2+DA2+DB**2+ADD
              ! L1=2, L2=2, M=1, LABEL=13.
              AA     = DA*r_SQRT2  ! /SQRT2
              AB     = DB*r_SQRT2  ! /SQRT2
              DABP   = AA+AB
              DABM   = AA-AB
              DABP2  = DABP**2  ! (AA+AB)**2
              DABM2  = DABM**2  ! (AA-AB)**2
              X(I+19)= (R+DABM)**2 + DABM2 + ADD ! (R+AA-AB)**2+(AA-AB)**2+ADD
              X(I+20)= (R+DABP)**2 + DABM2 + ADD ! (R+AA+AB)**2+(AA-AB)**2+ADD
              X(I+21)= (R-DABP)**2 + DABM2 + ADD ! (R-AA-AB)**2+(AA-AB)**2+ADD
              X(I+22)= (R-DABM)**2 + DABM2 + ADD ! (R-AA+AB)**2+(AA-AB)**2+ADD
              X(I+23)= (R+DABM)**2 + DABP2 + ADD ! (R+AA-AB)**2+(AA+AB)**2+ADD
              X(I+24)= (R+DABP)**2 + DABP2 + ADD ! (R+AA+AB)**2+(AA+AB)**2+ADD
              X(I+25)= (R-DABP)**2 + DABP2 + ADD ! (R-AA-AB)**2+(AA+AB)**2+ADD
              X(I+26)= (R-DABM)**2 + DABP2 + ADD ! (R-AA+AB)**2+(AA+AB)**2+ADD
              I      = I+26
           end if
        end if    ! LIJ.eq.6 
        if(LKLMAX > 3) then
           ! KL=DS.
           LX(4,LIJ) = I
           DB     = DD(4,NJ)
           QB     = PO(4,NJ)
           ADD    = (QA+QB)**2
           ADDAS  = (QAS+QB)**2
           RPDB2  = (R+DB)**2
           RMDB2  = (R-DB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=0, L2=2, M=0, LABEL=6.
           X(I+1) = RMDB2   + ADDAS     ! (R-DB)**2+ADDAS
           X(I+2) = R2+DB**2+ ADDAS
           X(I+3) = RPDB2   + ADDAS     ! (R+DB)**2+ADDAS
           ! L1=2, L2=2, M=0, LABEL=10.
           X(I+4) = (R-DABP)**2 + ADD   ! (R-DA-DB)**2+ADD
           X(I+5) = (R-DABM)**2 + ADD   ! (R-DA+DB)**2+ADD
           X(I+6) = (R+DABM)**2 + ADD   ! (R+DA-DB)**2+ADD
           X(I+7) = (R+DABP)**2 + ADD   ! (R+DA+DB)**2+ADD
           X(I+8) = RMDA2+DB**2 + ADD
           X(I+9) = RMDB2 + DA2 + ADD   ! (R-DB)**2+DA2+ADD
           X(I+10)= RPDA2+DB**2 + ADD
           X(I+11)= RPDB2 + DA2 + ADD   ! (R+DB)**2+DA2+ADD
           X(I+12)= R2 + DABM**2+ ADD   ! R2+(DA-DB)**2+ADD
           X(I+13)= R2 + DABP**2+ ADD   ! R2+(DA+DB)**2+ADD
           X(I+14)= R2+DA2+DB**2+ ADD
           ! L1=2, L2=2, M=1, LABEL=13.
           AA     = DA*r_SQRT2  ! /SQRT2
           AB     = DB*r_SQRT2  ! /SQRT2
           DABP   = AA+AB
           DABM   = AA-AB
           DABP2  = DABP**2  ! (AA+AB)**2
           DABM2  = DABM**2  ! (AA-AB)**2
           X(I+15)= (R+DABM)**2 + DABM2 + ADD ! (R+AA-AB)**2+(AA-AB)**2+ADD
           X(I+16)= (R+DABP)**2 + DABM2 + ADD ! (R+AA+AB)**2+(AA-AB)**2+ADD
           X(I+17)= (R-DABP)**2 + DABM2 + ADD ! (R-AA-AB)**2+(AA-AB)**2+ADD
           X(I+18)= (R-DABM)**2 + DABM2 + ADD ! (R-AA+AB)**2+(AA-AB)**2+ADD
           X(I+19)= (R+DABM)**2 + DABP2 + ADD ! (R+AA-AB)**2+(AA+AB)**2+ADD
           X(I+20)= (R+DABP)**2 + DABP2 + ADD ! (R+AA+AB)**2+(AA+AB)**2+ADD
           X(I+21)= (R-DABP)**2 + DABP2 + ADD ! (R-AA-AB)**2+(AA+AB)**2+ADD
           X(I+22)= (R-DABM)**2 + DABP2 + ADD ! (R-AA+AB)**2+(AA+AB)**2+ADD
           I      = I+22
           ! KL=DP.
           LX(5,LIJ) = I
           DB     = DD(5,NJ)
           QB     = PO(5,NJ)
           ADD    = (QA+QB)**2
           ADDAS  = (QAS+QB)**2
           RPDB2  = (R+DB)**2
           RMDB2  = (R-DB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=0, L2=1, M=0, LABEL=3.
           X(I+1) = RPDB2 + ADDAS ! (R+DB)**2+ADDAS
           X(I+2) = RMDB2 + ADDAS ! (R-DB)**2+ADDAS
           ! L1=2, L2=1, M=0, LABEL=9.
           X(I+3) = (R-DABP)**2 + ADD   ! (R-DA-DB)**2+ADD
           X(I+4) = RMDB2 + DA2 + ADD   ! (R-DB)**2+DA2+ADD
           X(I+5) = (R+DABM)**2 + ADD   ! (R+DA-DB)**2+ADD
           X(I+6) = (R-DABM)**2 + ADD   ! (R-DA+DB)**2+ADD
           X(I+7) = RPDB2 + DA2 + ADD   ! (R+DB)**2+DA2+ADD
           X(I+8) = (R+DABP)**2 + ADD   ! (R+DA+DB)**2+ADD
           ! L1=2, L2=1, M=1, LABEL=12.
           AA     = DA*r_SQRT2  ! /SQRT2
           RPAB2  = (R+AA)**2
           RMAB2  = (R-AA)**2
           DABP2  = (AA+DB)**2
           DABM2  = (AA-DB)**2
           X(I+9) = RPAB2 + DABM2 + ADD ! (R+AA)**2+(AA-DB)**2+ADD
           X(I+10)= RMAB2 + DABM2 + ADD ! (R-AA)**2+(AA-DB)**2+ADD
           X(I+11)= RPAB2 + DABP2 + ADD ! (R+AA)**2+(AA+DB)**2+ADD
           X(I+12)= RMAB2 + DABP2 + ADD ! (R-AA)**2+(AA+DB)**2+ADD
           I      = I+12
           ! KL=DD.
           LX(6,LIJ) = I
           DB     = DD(6,NJ)
           QB     = PO(6,NJ)
           QBS    = PO(8,NJ)
           ADD    = (QA+QB)**2
           ADDAS  = (QAS+QB)**2
           ADDBS  = (QA+QBS)**2
           ADDSS  = (QAS+QBS)**2
           RPDB2  = (R+DB)**2
           RMDB2  = (R-DB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=0, L2=0, M=0, LABEL=1.
           X(I+1) = R2+ADDSS
           ! L1=0, L2=2, M=0, LABEL=6.
           X(I+2) = RMDB2   + ADDAS  ! (R-DB)**2+ADDAS
           X(I+3) = R2+DB**2+ ADDAS
           X(I+4) = RPDB2   + ADDAS  ! (R+DB)**2+ADDAS
           ! L1=2, L2=0, M=0, LABEL=7.
           X(I+5) = RMDA2+ADDBS
           X(I+6) = R2+DA2+ADDBS
           X(I+7) = RPDA2+ADDBS
           ! L1=2, L2=2, M=0, LABEL=10.
           X(I+8) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
           X(I+9) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
           X(I+10)= (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
           X(I+11)= (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
           X(I+12)= RMDA2+DB**2 + ADD
           X(I+13)= RMDB2 + DA2 + ADD ! (R-DB)**2+DA2+ADD
           X(I+14)= RPDA2+DB**2 + ADD
           X(I+15)= RPDB2 + DA2 + ADD ! (R+DB)**2+DA2+ADD
           X(I+16)= R2+ DABM**2 + ADD ! R2+(DA-DB)**2+ADD
           X(I+17)= R2+ DABP**2 + ADD ! R2+(DA+DB)**2+ADD
           X(I+18)= R2+DA2+DB**2 + ADD
           ! L1=2, L2=2, M=1, LABEL=13.
           AA     = DA*r_SQRT2  ! /SQRT2
           AB     = DB*r_SQRT2  ! /SQRT2
           DABP   = AA+AB
           DABM   = AA-AB
           DABP2  = DABP**2  ! (AA+AB)**2
           DABM2  = DABM**2  ! (AA-AB)**2
           X(I+19)= (R+DABM)**2 + DABM2 + ADD  ! (R+AA-AB)**2+(AA-AB)**2+ADD
           X(I+20)= (R+DABP)**2 + DABM2 + ADD  ! (R+AA+AB)**2+(AA-AB)**2+ADD
           X(I+21)= (R-DABP)**2 + DABM2 + ADD  ! (R-AA-AB)**2+(AA-AB)**2+ADD
           X(I+22)= (R-DABM)**2 + DABM2 + ADD  ! (R-AA+AB)**2+(AA-AB)**2+ADD
           X(I+23)= (R+DABM)**2 + DABP2 + ADD  ! (R+AA-AB)**2+(AA+AB)**2+ADD
           X(I+24)= (R+DABP)**2 + DABP2 + ADD  ! (R+AA+AB)**2+(AA+AB)**2+ADD
           X(I+25)= (R-DABP)**2 + DABP2 + ADD  ! (R-AA-AB)**2+(AA+AB)**2+ADD
           X(I+26)= (R-DABM)**2 + DABP2 + ADD  ! (R-AA+AB)**2+(AA+AB)**2+ADD
           I      = I+26
        end if
     ! end of LIJ.eq.3 .or. LIJ.eq.6

     ! (DS,KL), LIJ=4.
     else if(LIJ == 4) then
        ! KL=SS.
        LX(1,LIJ) = I
        QB     = PO(1,NJ)
        ADD    = (QA+QB)**2
        ! L1=2, L2=0, M=0, LABEL=7.
        X(I+1) = RMDA2+ADD
        X(I+2) = R2+DA2+ADD
        X(I+3) = RPDA2+ADD
        I      = I+3
        if(LKLMAX > 1) then
           ! KL=PS.
           LX(2,LIJ) = I
           DB     = DD(2,NJ)
           QB     = PO(2,NJ)
           ADD    = (QA+QB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=2, L2=1, M=0, LABEL=9.
           X(I+1) = (R-DABP)**2 + ADD  ! (R-DA-DB)**2+ADD
           X(I+2) = (R-DB)**2+DA2+ADD
           X(I+3) = (R+DABM)**2 + ADD  ! (R+DA-DB)**2+ADD
           X(I+4) = (R-DABM)**2 + ADD  ! (R-DA+DB)**2+ADD
           X(I+5) = (R+DB)**2+DA2+ADD
           X(I+6) = (R+DABP)**2 + ADD  ! (R+DA+DB)**2+ADD
           ! L1=2, L2=1, M=1, LABEL=12.
           AA     = DA*r_SQRT2  ! /SQRT2
           RPAB2  = (R+AA)**2
           RMAB2  = (R-AA)**2
           DABP2  = (AA+DB)**2
           DABM2  = (AA-DB)**2
           X(I+7) = RPAB2 + DABM2 + ADD ! (R+AA)**2+(AA-DB)**2+ADD
           X(I+8) = RMAB2 + DABM2 + ADD ! (R-AA)**2+(AA-DB)**2+ADD
           X(I+9) = RPAB2 + DABP2 + ADD ! (R+AA)**2+(AA+DB)**2+ADD
           X(I+10)= RMAB2 + DABP2 + ADD ! (R-AA)**2+(AA+DB)**2+ADD
           I      = I+10
           ! KL=PP.
           LX(3,LIJ) = I
           DB     = DD(3,NJ)*SQRT2
           QB     = PO(3,NJ)
           QBS    = PO(7,NJ)
           ADD    = (QA+QB)**2
           ADDBS  = (QA+QBS)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=2, L2=0, M=0, LABEL=7.
           X(I+1) = RMDA2+ADDBS
           X(I+2) = R2+DA2+ADDBS
           X(I+3) = RPDA2+ADDBS
           ! L1=2, L2=2, M=0, LABEL=10.
           X(I+4) = (R-DABP)**2 + ADD ! (R-DA-DB)**2+ADD
           X(I+5) = (R-DABM)**2 + ADD ! (R-DA+DB)**2+ADD
           X(I+6) = (R+DABM)**2 + ADD ! (R+DA-DB)**2+ADD
           X(I+7) = (R+DABP)**2 + ADD ! (R+DA+DB)**2+ADD
           X(I+8) = RMDA2+DB**2 + ADD
           X(I+9) = (R-DB)**2+DA2+ADD
           X(I+10)= RPDA2+DB**2 + ADD
           X(I+11)= (R+DB)**2+DA2+ADD
           X(I+12)= R2+ DABM**2 + ADD ! R2+(DA-DB)**2+ADD
           X(I+13)= R2+ DABP**2 + ADD ! R2+(DA+DB)**2+ADD
           X(I+14)= R2+DA2+DB**2+ ADD
           ! L1=2, L2=2, M=1, LABEL=13.
           AA     = DA*r_SQRT2  ! /SQRT2
           AB     = DB*r_SQRT2  ! /SQRT2
           DABP   = AA+AB
           DABM   = AA-AB
           DABP2  = DABP**2  ! (AA+AB)**2
           DABM2  = DABM**2  ! (AA-AB)**2
           X(I+15)= (R+DABM)**2 + DABM2 + ADD  ! (R+AA-AB)**2+(AA-AB)**2+ADD
           X(I+16)= (R+DABP)**2 + DABM2 + ADD  ! (R+AA+AB)**2+(AA-AB)**2+ADD
           X(I+17)= (R-DABP)**2 + DABM2 + ADD  ! (R-AA-AB)**2+(AA-AB)**2+ADD
           X(I+18)= (R-DABM)**2 + DABM2 + ADD  ! (R-AA+AB)**2+(AA-AB)**2+ADD
           X(I+19)= (R+DABM)**2 + DABP2 + ADD  ! (R+AA-AB)**2+(AA+AB)**2+ADD
           X(I+20)= (R+DABP)**2 + DABP2 + ADD  ! (R+AA+AB)**2+(AA+AB)**2+ADD
           X(I+21)= (R-DABP)**2 + DABP2 + ADD  ! (R-AA-AB)**2+(AA+AB)**2+ADD
           X(I+22)= (R-DABM)**2 + DABP2 + ADD  ! (R-AA+AB)**2+(AA+AB)**2+ADD
           I      = I+22
        end if
        if(LKLMAX > 3) then
           ! KL=DS.
           LX(4,LIJ) = I
           DB     = DD(4,NJ)
           QB     = PO(4,NJ)
           ADD    = (QA+QB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=2, L2=2, M=0, LABEL=10.
           X(I+1) = (R-DABP)**2 + ADD  ! (R-DA-DB)**2+ADD
           X(I+2) = (R-DABM)**2 + ADD  ! (R-DA+DB)**2+ADD
           X(I+3) = (R+DABM)**2 + ADD  ! (R+DA-DB)**2+ADD
           X(I+4) = (R+DABP)**2 + ADD  ! (R+DA+DB)**2+ADD
           X(I+5) = RMDA2+DB**2 + ADD
           X(I+6) = (R-DB)**2+DA2+ADD
           X(I+7) = RPDA2+DB**2 + ADD
           X(I+8) = (R+DB)**2+DA2+ADD
           X(I+9) = R2+ DABM**2 + ADD  ! R2+(DA-DB)**2+ADD
           X(I+10)= R2+ DABP**2 + ADD  ! R2+(DA+DB)**2+ADD
           X(I+11)= R2+DA2+DB**2+ ADD
           ! L1=2, L2=2, M=1, LABEL=13.
           AA     = DA*r_SQRT2  ! /SQRT2
           AB     = DB*r_SQRT2  ! /SQRT2
           DABP   = AA+AB
           DABM   = AA-AB
           DABP2  = DABP**2  ! (AA+AB)**2
           DABM2  = DABM**2  ! (AA-AB)**2
           X(I+12)= (R+DABM)**2 + DABM2 + ADD  ! (R+AA-AB)**2+(AA-AB)**2+ADD
           X(I+13)= (R+DABP)**2 + DABM2 + ADD  ! (R+AA+AB)**2+(AA-AB)**2+ADD
           X(I+14)= (R-DABP)**2 + DABM2 + ADD  ! (R-AA-AB)**2+(AA-AB)**2+ADD
           X(I+15)= (R-DABM)**2 + DABM2 + ADD  ! (R-AA+AB)**2+(AA-AB)**2+ADD
           X(I+16)= (R+DABM)**2 + DABP2 + ADD  ! (R+AA-AB)**2+(AA+AB)**2+ADD
           X(I+17)= (R+DABP)**2 + DABP2 + ADD  ! (R+AA+AB)**2+(AA+AB)**2+ADD
           X(I+18)= (R-DABP)**2 + DABP2 + ADD  ! (R-AA-AB)**2+(AA+AB)**2+ADD
           X(I+19)= (R-DABM)**2 + DABP2 + ADD  ! (R-AA+AB)**2+(AA+AB)**2+ADD
           I      = I+19
           ! KL=DP.
           LX(5,LIJ) = I
           DB     = DD(5,NJ)
           QB     = PO(5,NJ)
           ADD    = (QA+QB)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=2, L2=1, M=0, LABEL=9.
           X(I+1) = (R-DABP)**2 + ADD   ! (R-DA-DB)**2+ADD
           X(I+2) = (R-DB)**2+DA2+ADD
           X(I+3) = (R+DABM)**2 + ADD   ! (R+DA-DB)**2+ADD
           X(I+4) = (R-DABM)**2 + ADD   ! (R-DA+DB)**2+ADD
           X(I+5) = (R+DB)**2+DA2+ADD
           X(I+6) = (R+DABP)**2 + ADD   ! (R+DA+DB)**2+ADD
           ! L1=2, L2=1, M=1, LABEL=12.
           AA     = DA*r_SQRT2  ! /SQRT2
           RPAB2  = (R+AA)**2
           RMAB2  = (R-AA)**2
           DABP2  = (AA+DB)**2
           DABM2  = (AA-DB)**2
           X(I+7) = RPAB2 + DABM2 + ADD  ! (R+AA)**2+(AA-DB)**2+ADD
           X(I+8) = RMAB2 + DABM2 + ADD  ! (R-AA)**2+(AA-DB)**2+ADD
           X(I+9) = RPAB2 + DABP2 + ADD  ! (R+AA)**2+(AA+DB)**2+ADD
           X(I+10)= RMAB2 + DABP2 + ADD  ! (R-AA)**2+(AA+DB)**2+ADD
           I      = I+10
           ! KL=DD.
           LX(6,LIJ) = I
           DB     = DD(6,NJ)
           QB     = PO(6,NJ)
           QBS    = PO(8,NJ)
           ADD    = (QA+QB)**2
           ADDBS  = (QA+QBS)**2
           DABP   = DA+DB
           DABM   = DA-DB
           ! L1=2, L2=0, M=0, LABEL=7.
           X(I+1) = RMDA2+ADDBS
           X(I+2) = R2+DA2+ADDBS
           X(I+3) = RPDA2+ADDBS
           ! L1=2, L2=2, M=0, LABEL=10.
           X(I+4) = (R-DABP)**2 + ADD  ! (R-DA-DB)**2+ADD
           X(I+5) = (R-DABM)**2 + ADD  ! (R-DA+DB)**2+ADD
           X(I+6) = (R+DABM)**2 + ADD  ! (R+DA-DB)**2+ADD
           X(I+7) = (R+DABP)**2 + ADD  ! (R+DA+DB)**2+ADD
           X(I+8) = RMDA2+DB**2+ADD
           X(I+9) = (R-DB)**2+DA2+ADD
           X(I+10)= RPDA2+DB**2+ADD
           X(I+11)= (R+DB)**2+DA2+ADD
           X(I+12)= R2+ DABM**2 + ADD   ! R2+(DA-DB)**2+ADD
           X(I+13)= R2+ DABP**2 + ADD   ! R2+(DA+DB)**2+ADD
           X(I+14)= R2+DA2+DB**2+ADD
           ! L1=2, L2=2, M=1, LABEL=13.
           AA     = DA*r_SQRT2  ! /SQRT2
           AB     = DB*r_SQRT2  ! /SQRT2
           DABP   = AA+AB
           DABM   = AA-AB
           DABP2  = DABP**2  ! (AA+AB)**2
           DABM2  = DABM**2  ! (AA-AB)**2
           X(I+15)= (R+DABM)**2 + DABM2 + ADD  ! (R+AA-AB)**2+(AA-AB)**2+ADD
           X(I+16)= (R+DABP)**2 + DABM2 + ADD  ! (R+AA+AB)**2+(AA-AB)**2+ADD
           X(I+17)= (R-DABP)**2 + DABM2 + ADD  ! (R-AA-AB)**2+(AA-AB)**2+ADD
           X(I+18)= (R-DABM)**2 + DABM2 + ADD  ! (R-AA+AB)**2+(AA-AB)**2+ADD
           X(I+19)= (R+DABM)**2 + DABP2 + ADD  ! (R+AA-AB)**2+(AA+AB)**2+ADD
           X(I+20)= (R+DABP)**2 + DABP2 + ADD  ! (R+AA+AB)**2+(AA+AB)**2+ADD
           X(I+21)= (R-DABP)**2 + DABP2 + ADD  ! (R-AA-AB)**2+(AA+AB)**2+ADD
           X(I+22)= (R-DABM)**2 + DABP2 + ADD  ! (R-AA+AB)**2+(AA+AB)**2+ADD
           I      = I+22
        end if
     end if     ! LIJ.eq.4 
  end do  loopLIJ1

  ! calculate inverse distances:  numerator ONE implies atomic units.
  !                                         EV          conversion to eV.
  ! it could be done after the following do loop at the level of the
  ! two-electron integrals W(LIMKL,LIMIJ).
  ilast = i
  X(1:ilast)   = eV/SQRT(X(1:ilast))

  !
  ! evaluate semiempirical multipole-multipole interactions XM(label). 
  ! evaluate the unique integrals W(KL,IJ) from XM(label).
  ! define the remaining symmetry-related integrals W(KL,IJ).
  loopLIJ2: do LIJ=1,LIJMAX
     ! (SS,KL), LIJ=1.
     if (LIJ == 1) then 
        if(LKLMAX > 3) then
           ! KL=DS.
           I        = LX(4,LIJ)
           XM(6)    =(X(I+1)-X(I+2)*TWO+X(I+3))*PT25
           W(11, 1) = XM( 6) * CLM11
           ! KL=DP.
           I        = LX(5,LIJ)
           XM(3)    =(X(I+1)-X(I+2))*PT5
           W(12, 1) = XM( 3) * CLM12
           W(18, 1) = XM( 3)
           W(25, 1) = W(18, 1)
           ! KL=DD.
           I     = LX(6,LIJ)
           XM(1) =  X(I+1)
           XM(6) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
           W(15, 1) = XM( 1) +XM( 6) * CLM15
           W(21, 1) = XM( 1) +XM( 6) * CLM21
           W(36, 1) = XM( 1) +XM( 6) * CLM36
           W(28, 1) = W(21, 1)
           W(45, 1) = W(36, 1)
        end if
     ! end of LIJ .eq. 1 

     ! (PS,KL), LIJ=2.
     else if(LIJ == 2) then
        if(LKLMAX > 3) then
           ! KL=DS.
           I        = LX(4,LIJ)
           XM(8)    = (X(I+1)-X(I+2)*TWO+X(I+3)-X(I+4)+X(I+5)*TWO-X(I+6))*0.125d0  ! /8.0D0
           XM(11)   =-(X(I+7)-X(I+8)-X(I+9)+X(I+10))*PT25
           W(11, 2) = XM( 8) * CLM11
           W(16, 4) = XM(11)
           W(22, 7) = W(16, 4)
           ! KL=DP.
           I        = LX(5,LIJ)
           XM(4)    = (X(I+1)+X(I+2)-X(I+3)-X(I+4))*PT25
           XM(5)    = (X(I+5)-X(I+6))*PT5
           W(12, 2) = XM( 4) * CLM12
           W(18, 2) = XM( 4)
           W(13, 4) = XM( 5) * CLM13
           W(17, 4) = XM( 5)
           W(31, 4) = XM( 5)
           W(25, 2) = W(18, 2)
           W(40, 4) = W(31, 4)
           W(14, 7) = W(13, 4)
           W(23, 7) = W(17, 4)
           W(32, 7) =-W(31, 4)
           W(39, 7) = W(31, 4)
           ! KL=DD.
           I        = LX(6,LIJ)
           XM(2)    =-(X(I+1)-X(I+2))*PT5
           XM(8)    = (X(I+3)-X(I+4)*TWO+X(I+5)-X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  ! /8.0D0
           XM(11)   =-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W(15, 2) = XM( 2) +XM( 8) * CLM15
           W(21, 2) = XM( 2) +XM( 8) * CLM21
           W(36, 2) = XM( 2) +XM( 8) * CLM36
           W(20, 4) = XM(11) * CLM20
           W(34, 4) = XM(11)
           W(28, 2) = W(21, 2)
           W(45, 2) = W(36, 2)
           W(43, 4) = W(34, 4)
           W(26, 7) = W(20, 4)
           W(35, 7) =-W(34, 4)
           W(42, 7) = W(34, 4)
        end if
     ! end LIJ.eq.2

     ! (PP,KL), LIJ=3.
     else if (LIJ == 3) then 
        if(LKLMAX > 3) then
           ! KL=DS.
           I   = LX(4,LIJ)
           XM(6) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
           ZZZZ  =  X(I+4)+X(I+5)+X(I+6)+X(I+7) - TWO*(X(I+8)+X(I+9)+X(I+10)+X(I+11)-X(I+12)-X(I+13))
           XYXY  =  X(I+12)+X(I+13)-X(I+14)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+15)-X(I+16)-X(I+17)+X(I+18)-X(I+19)+X(I+20)+X(I+21)-X(I+22))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(11, 3) = XM( 6) * CLM11 +XM(10) * CLM3*CLM11
           W(16, 5) = XM(13)
           W(11, 6) = XM( 6) * CLM11 +XM(10) * CLM6*CLM11
           W(29, 6) = XM(14)
           W(22, 8) = W(16, 5)
           W(37, 9) = W(29, 6)
           W(11,10) = W(11, 6)
           W(29,10) =-W(29, 6)
           ! KL=DP.
           I   = LX(5,LIJ)
           XM(3) = (X(I+1)-X(I+2))*PT5
           XM(9) =-(X(I+3)-X(I+4)*TWO+X(I+5)-X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  !/8.0D0
           XM(12)=-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W(12, 3) = XM( 3) * CLM12 +XM( 9) * CLM3*CLM12
           W(18, 3) = XM( 3) +XM( 9) * CLM3
           W(13, 5) = XM(12) * CLM13
           W(17, 5) = XM(12)
           W(31, 5) = XM(12)
           W(12, 6) = XM( 3) * CLM12 +XM( 9) * CLM6*CLM12
           W(18, 6) = XM( 3) +XM( 9) * CLM6
           W(25, 6) = XM( 3) +XM( 9) * CLM6
           W(25, 3) = W(18, 3)
           W(40, 5) = W(31, 5)
           W(14, 8) = W(13, 5)
           W(23, 8) = W(17, 5)
           W(32, 8) =-W(31, 5)
           W(39, 8) = W(31, 5)
           W(19, 9) = W(30, 6)
           W(24, 9) = W(30, 6)
           W(38, 9) = W(30, 6)
           W(12,10) = W(12, 6)
           W(18,10) = W(25, 6)
           W(25,10) = W(18, 6)
           W(30,10) =-W(30, 6)
           ! KL=DD.
           I   = LX(6,LIJ)
           XM(1) =  X(I+1)
           XM(6) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
           XM(7) = (X(I+5)-X(I+6)*TWO+X(I+7))*PT25
           ZZZZ  =  X(I+8)+X(I+9)+X(I+10)+X(I+11) - TWO*(X(I+12)+X(I+13)+X(I+14)+X(I+15)-X(I+16)-X(I+17))
           XYXY  =  X(I+16)+X(I+17)-X(I+18)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+19)-X(I+20)-X(I+21)+X(I+22) -X(I+23)+X(I+24)+X(I+25)-X(I+26))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(15, 3) = XM( 1)+XM( 6)*CLM15+XM( 7)*CLM3 +XM(10)*CLM3*CLM15
           W(21, 3) = XM( 1)+XM( 6)*CLM21+XM( 7)*CLM3 +XM(10)*CLM3*CLM21
           W(36, 3) = XM( 1)+XM( 6)*CLM36+XM( 7)*CLM3 +XM(10)*CLM3*CLM36
           W(20, 5) = XM(13) * CLM20
           W(34, 5) = XM(13)
           W(15, 6) = XM( 1)+XM( 6)*CLM15+XM( 7)*CLM6 +XM(10)*CLM6*CLM15
           W(21, 6) = XM( 1)+XM( 6)*CLM21+XM( 7)*CLM6 +XM(10)*CLM6*CLM21+XM(14)
           W(28, 6) = XM( 1)+XM( 6)*CLM28+XM( 7)*CLM6 +XM(10)*CLM6*CLM28-XM(14)
           W(33, 6) = XM(14) * CLM33
           W(36, 6) = XM( 1)+XM( 6)*CLM36+XM( 7)*CLM6 +XM(10)*CLM6*CLM36
           W(27, 9) = XM(14)
           W(28, 3) = W(21, 3)
           W(45, 3) = W(36, 3)
           W(43, 5) = W(34, 5)
           W(45, 6) = W(36, 6)
           W(26, 8) = W(20, 5)
           W(35, 8) =-W(34, 5)
           W(42, 8) = W(34, 5)
           W(41, 9) = W(33, 6)
           W(15,10) = W(15, 6)
           W(21,10) = W(28, 6)
           W(28,10) = W(21, 6)
           W(33,10) =-W(33, 6)
           W(36,10) = W(36, 6)
           W(45,10) = W(36, 6)
        end if
     ! end of LIJ.eq.3

     ! (DS,KL), LIJ=4.
     else if (LIJ == 4) then
        ! KL=SS.
        I      = LX(1,LIJ)
        XM(7) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
        W( 1,11) = XM( 7) * CLM11
        if(LKLMAX > 1) then
           ! KL=PS.
           I   = LX(2,LIJ)
           XM(9) =-(X(I+1)-X(I+2)*TWO+X(I+3)-X(I+4)+X(I+5)*TWO-X(I+6))*0.125d0  !/8.0D0
           XM(12)=-(X(I+7)-X(I+8)-X(I+9)+X(I+10))*PT25
           W( 2,11) = XM( 9) * CLM11
           W( 4,16) = XM(12)
           W( 7,22) = W( 4,16)
           ! KL=PP.
           I   = LX(3,LIJ)
           XM(7) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
           ZZZZ  =  X(I+4)+X(I+5)+X(I+6)+X(I+7) - TWO*(X(I+8)+X(I+9)+X(I+10)+X(I+11)-X(I+12)-X(I+13))
           XYXY  =  X(I+12)+X(I+13)-X(I+14)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+15)-X(I+16)-X(I+17)+X(I+18)-X(I+19)+X(I+20)+X(I+21)-X(I+22))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W( 3,11) = XM( 7) * CLM11 +XM(10) * CLM11*CLM3
           W( 6,11) = XM( 7) * CLM11 +XM(10) * CLM11*CLM6
           W( 5,16) = XM(13)
           W( 6,29) = XM(14)
           W(10,11) = W( 6,11)
           W( 8,22) = W( 5,16)
           W(10,29) =-W( 6,29)
           W( 9,37) = W( 6,29)
        end if
        if(LKLMAX > 3) then
           ! KL=DS.
           I   = LX(4,LIJ)
           ZZZZ  =  X(I+1)+X(I+2)+X(I+3)+X(I+4) - TWO*(X(I+5)+X(I+6)+X(I+7)+X(I+8)-X(I+9)-X(I+10))
           XYXY  =  X(I+9)+X(I+10)-X(I+11)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+12)-X(I+13)-X(I+14)+X(I+15)-X(I+16)+X(I+17)+X(I+18)-X(I+19))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(11,11) = XM(10) * CLM11*CLM11
           W(16,16) = XM(13)
           W(29,29) = XM(14)
           W(22,22) = W(16,16)
           W(37,37) = W(29,29)
           ! KL=DP.
           I   = LX(5,LIJ)
           XM(9) =-(X(I+1)-X(I+2)*TWO+X(I+3)-X(I+4)+X(I+5)*TWO-X(I+6))*0.125d0  !/8.0D0
           XM(12)=-(X(I+7)-X(I+8)-X(I+9)+X(I+10))*PT25
           W(12,11) = XM( 9) * CLM11*CLM12
           W(18,11) = XM( 9) * CLM11
           W(13,16) = XM(12) * CLM13
           W(17,16) = XM(12)
           W(31,16) = XM(12)
           W(25,11) = W(18,11)
           W(40,16) = W(31,16)
           W(14,22) = W(13,16)
           W(23,22) = W(17,16)
           W(32,22) =-W(31,16)
           W(39,22) = W(31,16)
           W(25,29) =-W(18,29)
           W(30,29) = W(18,29)
           W(19,37) = W(18,29)
           W(24,37) = W(18,29)
           W(38,37) = W(18,29)
           ! KL=DD.
           I   = LX(6,LIJ)
           XM(7) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
           ZZZZ  =  X(I+4)+X(I+5)+X(I+6)+X(I+7) - TWO*(X(I+8)+X(I+9)+X(I+10)+X(I+11)-X(I+12)-X(I+13))
           XYXY  =  X(I+12)+X(I+13)-X(I+14)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+15)-X(I+16)-X(I+17)+X(I+18)-X(I+19)+X(I+20)+X(I+21)-X(I+22))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(15,11) = XM( 7) * CLM11 +XM(10) * CLM11*CLM15
           W(21,11) = XM( 7) * CLM11 +XM(10) * CLM11*CLM21
           W(36,11) = XM( 7) * CLM11 +XM(10) * CLM11*CLM36
           W(20,16) = XM(13) * CLM20
           W(34,16) = XM(13)
           W(21,29) = XM(14)
           W(33,29) = XM(14) * CLM33
           W(28,11) = W(21,11)
           W(45,11) = W(36,11)
           W(43,16) = W(34,16)
           W(26,22) = W(20,16)
           W(35,22) =-W(34,16)
           W(42,22) = W(34,16)
           W(28,29) =-W(21,29)
           W(27,37) = W(21,29)
           W(41,37) = W(33,29)
        end if
     ! end of LIJ.eq.4

     ! (DP,KL), LIJ=5.
     else if (LIJ == 5) then 
        ! KL=SS.
        I      = LX(1,LIJ)
        XM(2) =-(X(I+1)-X(I+2))*PT5
        W( 1,12) = XM( 2) * CLM12
        W( 1,18) = XM( 2)
        W( 1,25) = W( 1,18)
        if(LKLMAX > 1) then
           ! KL=PS.
           I   = LX(2,LIJ)
           XM(4) = (X(I+1)+X(I+2)-X(I+3)-X(I+4))*PT25
           XM(5) = (X(I+5)-X(I+6))*PT5
           W( 2,12) = XM( 4) * CLM12
           W( 4,13) = XM( 5) * CLM13
           W( 4,17) = XM( 5)
           W( 2,18) = XM( 4)
           W( 4,31) = XM( 5)
           W( 7,14) = W( 4,13)
           W( 7,23) = W( 4,17)
           W( 2,25) = W( 2,18)
           W( 7,32) =-W( 4,31)
           W( 7,39) = W( 4,31)
           W( 4,40) = W( 4,31)
           ! KL=PP.
           I   = LX(3,LIJ)
           XM(2) =-(X(I+1)-X(I+2))*PT5
           XM(8) = (X(I+3)-X(I+4)*TWO+X(I+5)-X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  !/8.0D0
           XM(11)=-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W( 3,12) = XM( 2) * CLM12 +XM( 8) * CLM12*CLM3
           W( 6,12) = XM( 2) * CLM12 +XM( 8) * CLM12*CLM6
           W( 5,13) = XM(11) * CLM13
           W( 5,17) = XM(11)
           W( 3,18) = XM( 2) +XM( 8) * CLM3
           W( 6,18) = XM( 2) +XM( 8) * CLM6
           W(10,18) = XM( 2) +XM( 8) * CLM10
           W( 5,31) = XM(11)
           W(10,12) = W( 6,12)
           W( 8,14) = W( 5,13)
           W( 8,23) = W( 5,17)
           W( 9,24) = W( 9,19)
           W( 3,25) = W( 3,18)
           W( 6,25) = W(10,18)
           W(10,25) = W( 6,18)
           W( 6,30) = W( 9,19)
           W(10,30) =-W( 9,19)
           W( 8,32) =-W( 5,31)
           W( 9,38) = W( 9,19)
           W( 8,39) = W( 5,31)
           W( 5,40) = W( 5,31)
        end if
        if(LKLMAX > 3) then
           ! KL=DS.
           I   = LX(4,LIJ)
           XM(8) = (X(I+1)-X(I+2)*TWO+X(I+3)-X(I+4)+X(I+5)*TWO-X(I+6))*0.125d0  !/8.0D0
           XM(11)=-(X(I+7)-X(I+8)-X(I+9)+X(I+10))*PT25
           W(11,12) = XM( 8) * CLM12*CLM11
           W(16,13) = XM(11) * CLM13
           W(16,17) = XM(11)
           W(11,18) = XM( 8) * CLM11
           W(16,31) = XM(11)
           W(22,14) = W(16,13)
           W(37,19) = W(29,18)
           W(22,23) = W(16,17)
           W(37,24) = W(29,18)
           W(11,25) = W(11,18)
           W(29,25) =-W(29,18)
           W(29,30) = W(29,18)
           W(22,32) =-W(16,31)
           W(37,38) = W(29,18)
           W(22,39) = W(16,31)
           W(16,40) = W(16,31)
           ! KL=DP.
           I   = LX(5,LIJ)
           XM(4) = (X(I+1)+X(I+2)-X(I+3)-X(I+4))*PT25
           XM(5) = (X(I+5)-X(I+6))*PT5
           W(12,12) = XM( 4) * CLM12*CLM12
           W(18,12) = XM( 4) * CLM12
           W(13,13) = XM( 5) * CLM13*CLM13
           W(17,13) = XM( 5) * CLM13
           W(31,13) = XM( 5) * CLM13
           W(13,17) = XM( 5) * CLM13
           W(17,17) = XM( 5)
           W(31,17) = XM( 5)
           W(12,18) = XM( 4) * CLM12
           W(18,18) = XM( 4)
           W(25,18) = XM( 4)
           W(13,31) = XM( 5) * CLM13
           W(17,31) = XM( 5)
           W(31,31) = XM( 5)
           W(40,31) = XM( 5)
           W(25,12) = W(18,12)
           W(40,13) = W(31,13)
           W(14,14) = W(13,13)
           W(23,14) = W(17,13)
           W(32,14) =-W(31,13)
           W(39,14) = W(31,13)
           W(40,17) = W(31,17)
           W(19,19) = W(30,18)
           W(24,19) = W(30,18)
           W(38,19) = W(30,18)
           W(14,23) = W(13,17)
           W(23,23) = W(17,17)
           W(32,23) =-W(31,17)
           W(39,23) = W(31,17)
           W(19,24) = W(30,18)
           W(24,24) = W(30,18)
           W(38,24) = W(30,18)
           W(12,25) = W(12,18)
           W(18,25) = W(25,18)
           W(25,25) = W(18,18)
           W(30,25) =-W(30,18)
           W(18,30) = W(30,18)
           W(25,30) =-W(30,18)
           W(30,30) = W(30,18)
           W(14,32) =-W(13,31)
           W(23,32) =-W(17,31)
           W(32,32) = W(31,31)
           W(39,32) =-W(40,31)
           W(19,38) = W(30,18)
           W(24,38) = W(30,18)
           W(38,38) = W(30,18)
           W(14,39) = W(13,31)
           W(23,39) = W(17,31)
           W(32,39) =-W(40,31)
           W(39,39) = W(31,31)
           W(13,40) = W(13,31)
           W(17,40) = W(17,31)
           W(31,40) = W(40,31)
           W(40,40) = W(31,31)
           ! KL=DD.
           I   = LX(6,LIJ)
           XM(2) =-(X(I+1)-X(I+2))*PT5
           XM(8) = (X(I+3)-X(I+4)*TWO+X(I+5)-X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  !/8.0D0
           XM(11)=-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W(15,12) = XM( 2) * CLM12 +XM( 8) * CLM12*CLM15
           W(21,12) = XM( 2) * CLM12 +XM( 8) * CLM12*CLM21
           W(36,12) = XM( 2) * CLM12 +XM( 8) * CLM12*CLM36
           W(20,13) = XM(11) * CLM13*CLM20
           W(34,13) = XM(11) * CLM13
           W(20,17) = XM(11) * CLM20
           W(34,17) = XM(11)
           W(15,18) = XM( 2) +XM( 8) * CLM15
           W(21,18) = XM( 2) +XM( 8) * CLM21
           W(28,18) = XM( 2) +XM( 8) * CLM28
           W(36,18) = XM( 2) +XM( 8) * CLM36
           W(20,31) = XM(11) * CLM20
           W(34,31) = XM(11)
           W(43,31) = XM(11)
           W(28,12) = W(21,12)
           W(45,12) = W(36,12)
           W(43,13) = W(34,13)
           W(26,14) = W(20,13)
           W(35,14) =-W(34,13)
           W(42,14) = W(34,13)
           W(43,17) = W(34,17)
           W(45,18) = W(36,18)
           W(41,19) = W(33,18)
           W(26,23) = W(20,17)
           W(35,23) =-W(34,17)
           W(42,23) = W(34,17)
           W(27,24) = W(27,19)
           W(41,24) = W(33,18)
           W(15,25) = W(15,18)
           W(21,25) = W(28,18)
           W(28,25) = W(21,18)
           W(33,25) =-W(33,18)
           W(36,25) = W(36,18)
           W(45,25) = W(36,18)
           W(21,30) = W(27,19)
           W(28,30) =-W(27,19)
           W(33,30) = W(33,18)
           W(26,32) =-W(20,31)
           W(35,32) = W(34,31)
           W(42,32) =-W(43,31)
           W(27,38) = W(27,19)
           W(41,38) = W(33,18)
           W(26,39) = W(20,31)
           W(35,39) =-W(43,31)
           W(42,39) = W(34,31)
           W(20,40) = W(20,31)
           W(34,40) = W(43,31)
           W(43,40) = W(34,31)
        end if
     ! end of LIJ.eq.5

     ! (DD,KL), LIJ=6.
     else if (LIJ == 6) then
        ! KL=SS.
        I      = LX(1,LIJ)
        XM(1) =  X(I+1)
        XM(7) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
        W( 1,15) = XM( 1) +XM( 7) * CLM15
        W( 1,21) = XM( 1) +XM( 7) * CLM21
        W( 1,36) = XM( 1) +XM( 7) * CLM36
        W( 1,28) = W( 1,21)
        W( 1,45) = W( 1,36)
        if(LKLMAX > 1) then
           ! KL=PS.
           I   = LX(2,LIJ)
           XM(3) = (X(I+1)-X(I+2))*PT5
           XM(9) =-(X(I+3)-X(I+4)*TWO+X(I+5)-X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  !/8.0D0
           XM(12)=-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W( 2,15) = XM( 3) +XM( 9) * CLM15
           W( 4,20) = XM(12) * CLM20
           W( 2,21) = XM( 3) +XM( 9) * CLM21
           W( 4,34) = XM(12)
           W( 2,36) = XM( 3) +XM( 9) * CLM36
           W( 7,26) = W( 4,20)
           W( 2,28) = W( 2,21)
           W( 7,35) =-W( 4,34)
           W( 7,42) = W( 4,34)
           W( 4,43) = W( 4,34)
           W( 2,45) = W( 2,36)
           ! KL=PP.
           I   = LX(3,LIJ)
           XM(1) =  X(I+1)
           XM(6) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
           XM(7) = (X(I+5)-X(I+6)*TWO+X(I+7))*PT25
           ZZZZ  =  X(I+8)+X(I+9)+X(I+10)+X(I+11) - TWO*(X(I+12)+X(I+13)+X(I+14)+X(I+15)-X(I+16)-X(I+17))
           XYXY  =  X(I+16)+X(I+17)-X(I+18)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+19)-X(I+20)-X(I+21)+X(I+22) -X(I+23)+X(I+24)+X(I+25)-X(I+26))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W( 3,15) = XM( 1)+XM( 6)*CLM3+XM( 7) *CLM15 +XM(10)*CLM15*CLM3
           W( 6,15) = XM( 1)+XM( 6)*CLM6+XM( 7) *CLM15 +XM(10)*CLM15*CLM6
           W( 5,20) = XM(13) * CLM20
           W( 3,21) = XM( 1)+XM( 6)*CLM3+XM( 7) *CLM21 +XM(10)*CLM21*CLM3
           W( 6,21) = XM( 1)+XM( 6)*CLM6+XM( 7) *CLM21 +XM(10)*CLM21*CLM6 +XM(14)
           W(10,21) = XM( 1)+XM( 6)*CLM10+XM( 7)*CLM21 +XM(10)*CLM21*CLM10-XM(14)
           W( 9,27) = XM(14)
           W( 6,33) = XM(14) * CLM33
           W( 5,34) = XM(13)
           W( 3,36) = XM( 1)+XM( 6)*CLM3+XM( 7)*CLM36 +XM(10)*CLM36*CLM3
           W( 6,36) = XM( 1)+XM( 6)*CLM6+XM( 7)*CLM36 +XM(10)*CLM36*CLM6
           W(10,15) = W( 6,15)
           W( 8,26) = W( 5,20)
           W( 3,28) = W( 3,21)
           W( 6,28) = W(10,21)
           W(10,28) = W( 6,21)
           W(10,33) =-W( 6,33)
           W( 8,35) =-W( 5,34)
           W(10,36) = W( 6,36)
           W( 9,41) = W( 6,33)
           W( 8,42) = W( 5,34)
           W( 5,43) = W( 5,34)
           W( 3,45) = W( 3,36)
           W( 6,45) = W( 6,36)
           W(10,45) = W( 6,36)
        end if
        if(LKLMAX > 3) then
           ! KL=DS.
           I   = LX(4,LIJ)
           XM(6) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
           ZZZZ  =  X(I+4)+X(I+5)+X(I+6)+X(I+7) - TWO*(X(I+8)+X(I+9)+X(I+10)+X(I+11)-X(I+12)-X(I+13))
           XYXY  =  X(I+12)+X(I+13)-X(I+14)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+15)-X(I+16)-X(I+17)+X(I+18) -X(I+19)+X(I+20)+X(I+21)-X(I+22))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(11,15) = XM( 6) * CLM11 +XM(10) * CLM15*CLM11
           W(16,20) = XM(13) * CLM20
           W(11,21) = XM( 6) * CLM11 +XM(10) * CLM21*CLM11
           W(29,21) = XM(14)
           W(29,33) = XM(14) * CLM33
           W(16,34) = XM(13)
           W(11,36) = XM( 6) * CLM11 +XM(10) * CLM36*CLM11
           W(22,26) = W(16,20)
           W(37,27) = W(29,21)
           W(11,28) = W(11,21)
           W(29,28) =-W(29,21)
           W(22,35) =-W(16,34)
           W(37,41) = W(29,33)
           W(22,42) = W(16,34)
           W(16,43) = W(16,34)
           W(11,45) = W(11,36)
           ! KL=DP.
           I   = LX(5,LIJ)
           XM(3) = (X(I+1)-X(I+2))*PT5
           XM(9) =-(X(I+3)-X(I+4)*TWO+X(I+5) -X(I+6)+X(I+7)*TWO-X(I+8))*0.125d0  !/8.0D0
           XM(12)=-(X(I+9)-X(I+10)-X(I+11)+X(I+12))*PT25
           W(12,15) = XM( 3) * CLM12 +XM( 9) * CLM15*CLM12
           W(18,15) = XM( 3) +XM( 9) * CLM15
           W(13,20) = XM(12) * CLM20*CLM13
           W(17,20) = XM(12) * CLM20
           W(31,20) = XM(12) * CLM20
           W(12,21) = XM( 3) * CLM12 +XM( 9) * CLM21*CLM12
           W(18,21) = XM( 3) +XM( 9) * CLM21
           W(25,21) = XM( 3) +XM( 9) * CLM21
           W(13,34) = XM(12) * CLM13
           W(17,34) = XM(12)
           W(31,34) = XM(12)
           W(40,34) = XM(12)
           W(12,36) = XM( 3) * CLM12 +XM( 9) * CLM36*CLM12
           W(18,36) = XM( 3) +XM( 9) * CLM36
           W(25,15) = W(18,15)
           W(40,20) = W(31,20)
           W(14,26) = W(13,20)
           W(23,26) = W(17,20)
           W(32,26) =-W(31,20)
           W(39,26) = W(31,20)
           W(19,27) = W(30,21)
           W(24,27) = W(30,21)
           W(38,27) = W(30,21)
           W(12,28) = W(12,21)
           W(18,28) = W(25,21)
           W(25,28) = W(18,21)
           W(30,28) =-W(30,21)
           W(25,33) =-W(18,33)
           W(30,33) = W(18,33)
           W(14,35) =-W(13,34)
           W(23,35) =-W(17,34)
           W(32,35) = W(31,34)
           W(39,35) =-W(40,34)
           W(25,36) = W(18,36)
           W(19,41) = W(18,33)
           W(24,41) = W(18,33)
           W(38,41) = W(18,33)
           W(14,42) = W(13,34)
           W(23,42) = W(17,34)
           W(32,42) =-W(40,34)
           W(39,42) = W(31,34)
           W(13,43) = W(13,34)
           W(17,43) = W(17,34)
           W(31,43) = W(40,34)
           W(40,43) = W(31,34)
           W(12,45) = W(12,36)
           W(18,45) = W(18,36)
           W(25,45) = W(18,36)
           ! KL=DD.
           I   = LX(6,LIJ)
           XM(1) =  X(I+1)
           XM(6) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
           XM(7) = (X(I+5)-X(I+6)*TWO+X(I+7))*PT25
           ZZZZ  =  X(I+8)+X(I+9)+X(I+10)+X(I+11) - TWO*(X(I+12)+X(I+13)+X(I+14)+X(I+15)-X(I+16)-X(I+17))
           XYXY  =  X(I+16)+X(I+17)-X(I+18)*TWO
           XM(10)= (ZZZZ-XYXY)*0.0625d0   ! /16.0D0
           XM(13)= (X(I+19)-X(I+20)-X(I+21)+X(I+22) -X(I+23)+X(I+24)+X(I+25)-X(I+26))*0.125d0  !/8.0D0
           XM(14)= XYXY*PT25
           W(15,15) = XM( 1)+XM( 6)*CLM15+XM( 7)*CLM15 +XM(10)*CLM15*CLM15
           W(21,15) = XM( 1)+XM( 6)*CLM21+XM( 7)*CLM15 +XM(10)*CLM15*CLM21   
           W(36,15) = XM( 1)+XM( 6)*CLM36+XM( 7)*CLM15 +XM(10)*CLM15*CLM36
           W(20,20) = XM(13) * CLM20*CLM20
           W(34,20) = XM(13) * CLM20
           W(15,21) = XM( 1)+XM( 6)*CLM15+XM( 7)*CLM21 +XM(10)*CLM21*CLM15
           W(21,21) = XM( 1)+XM( 6)*CLM21+XM( 7)*CLM21 +XM(10)*CLM21*CLM21 +XM(14)
           W(28,21) = XM( 1)+XM( 6)*CLM28+XM( 7)*CLM21 +XM(10)*CLM21*CLM28 -XM(14)
           W(33,21) = XM(14) * CLM33
           W(36,21) = XM( 1)+XM( 6)*CLM36+XM( 7)*CLM21 +XM(10)*CLM21*CLM36
           W(27,27) = XM(14)
           W(21,33) = XM(14) * CLM33
           W(33,33) = XM(14) * CLM33*CLM33
           W(20,34) = XM(13) * CLM20
           W(34,34) = XM(13)
           W(43,34) = XM(13)
           W(15,36) = XM( 1)+XM( 6)*CLM15+XM( 7)*CLM36 +XM(10)*CLM36*CLM15
           W(21,36) = XM( 1)+XM( 6)*CLM21+XM( 7)*CLM36 +XM(10)*CLM36*CLM21
           W(36,36) = XM( 1)+XM( 6)*CLM36+XM( 7)*CLM36 +XM(10)*CLM36*CLM36
           W(45,36) = XM( 1)+XM( 6)*CLM45+XM( 7)*CLM36 +XM(10)*CLM36*CLM45
           W(28,15) = W(21,15)
           W(45,15) = W(36,15)
           W(43,20) = W(34,20)
           W(45,21) = W(36,21)
           W(26,26) = W(20,20)
           W(35,26) =-W(34,20)
           W(42,26) = W(34,20)
           W(41,27) = W(33,21)
           W(15,28) = W(15,21)
           W(21,28) = W(28,21)
           W(28,28) = W(21,21)
           W(33,28) =-W(33,21)
           W(36,28) = W(36,21)
           W(45,28) = W(36,21)
           W(28,33) =-W(21,33)
           W(26,35) =-W(20,34)
           W(35,35) = W(34,34)
           W(42,35) =-W(43,34)
           W(28,36) = W(21,36)
           W(27,41) = W(21,33)
           W(41,41) = W(33,33)
           W(26,42) = W(20,34)
           W(35,42) =-W(43,34)
           W(42,42) = W(34,34)
           W(20,43) = W(20,34)
           W(34,43) = W(43,34)
           W(43,43) = W(34,34)
           W(15,45) = W(15,36)
           W(21,45) = W(21,36)
           W(28,45) = W(21,36)
           W(36,45) = W(45,36)
           W(45,45) = W(36,36)
        end if
     end if   ! end of LIJ.eq.6 
  end do loopLIJ2

  !
  ! include two-electron integrals involving only s and p orbitals.
  ! Currently, local integrals RI(22) from subroutine REPP.
  W( 1, 1) = RI(1)
  if(LKLMAX > 1) then
     W( 2, 1) = RI( 5)
     W( 3, 1) = RI(11)
     W( 6, 1) = RI(12)
     W(10, 1) = RI(12)
  end if
  if(LIJMAX > 1) then
     W( 1, 2) = RI( 2)
     W( 1, 3) = RI( 3)
     W( 1, 6) = RI( 4)
     W( 1,10) = RI( 4)
  end if
  if(LIJMAX > 1 .and. LKLMAX > 1) then
     W( 2, 2) = RI( 6)
     W( 3, 2) = RI(13)
     W( 6, 2) = RI(14)
     W(10, 2) = RI(14)
     W( 2, 3) = RI( 8)
     W( 3, 3) = RI(16)
     W( 6, 3) = RI(18)
     W(10, 3) = RI(18)
     W( 4, 4) = RI( 7)
     W( 5, 4) = RI(15)
     W( 4, 5) = RI(10)
     W( 5, 5) = RI(20)
     W( 2, 6) = RI( 9)
     W( 3, 6) = RI(17)
     W( 6, 6) = RI(19)
     W(10, 6) = RI(21)
     W( 7, 7) = RI( 7)
     W( 8, 7) = RI(15)
     W( 7, 8) = RI(10)
     W( 8, 8) = RI(20)
     W( 9, 9) = RI(22)
     W( 2,10) = RI( 9)
     W( 3,10) = RI(17)
     W( 6,10) = RI(21)
     W(10,10) = RI(19)
  end if
  
  !
  ! core-electron attraction integrals.
  ! CORE_mat(i,1) - electrons (ij) at atom A and core of atom B (nc=nj).
  ! CORE_mat(i,2) - electrons (kl) at atom B and core of atom A (nc=ni).
  !
  ! index i in CORE_mat(i,1) and CORE_mat(i,2).
  ! (S S )=1, (S P )=2, (P P )=3, (P+P+)=4
  ! (D S )=5, (D P )=6, (D D )=7, (D+P+)=8, (D+D+)=9, (D#D#)=10.
  ! notation: P-SIGMA=P , P-PI=P+, D-SIGMA=D , D-PI=D+, D-DELTA=D#.
  !
  ! loop over the two possible cases.
  ! NC   atomic number of atom carrying the core.
  ! NE   atomic number of atom carrying the electrons.
  ! LE   limit for L pair index, 1 for S, 3 for SP, 6 for SPD.
  ! SIG  sign factor for integrals, always positive except for interations between (ij) at 
  !      atom A involving one P-sigma or D-Pi orbitail and the core of atom B (case N=1).
  !
  ifirst = 8  
  loopN: do n=1,2
     if(n == 1) then
        NC  = nj
        NE  = ni
        LE  = LIJMAX
        SIG =-one
     else
        NC  = ni
        NE  = nj
        LE  = LKLMAX
        SIG = one
     end if
     ! check for external point charge:
     !if(NE == 0) cycle loopN

     !if(NC == 0) then  ! mm atom
     !   COREnc = one
     !else
        COREnc = CORE(nc)
     !end if
     ! additive term for core.
     QA     = PO(9,NC)
     !
     ! check for neglect of penetration integrals. if the additive terms for SS (PO(1,nc)) 
     ! and the core (PO(9,nc)) are equal, the core-electron attraction integrals are 
     ! expressed in terms of the two-electron integrals (SS,KL) and (IJ,SS). 
     ! LCW(i) is the standard pair index corressponding to i in CORE_mat(i,n).
     if(abs(QA-PO(1,nc)) < small) then  ! for atoms iorbs<=4, PO(9,NC)=PO(1,NC)
        if(le == 1) then                 !           see fill_qm_parameters.
           icmax = 1                     ! therefore, for the current n, if it
        else if(le == 3) then            ! belongs here, the other n should be
           icmax = 4                     ! the norbs=9 (LE=6).
        else
           icmax = 10
        end if
        if(n == 1) then
           CORE_mat(1:icmax,1) = -COREnc*W(1,LCW(1:icmax))
        else
           CORE_mat(1:icmax,2) = -COREnc*W(LCW(1:icmax),1)
        end if
     else
        ! evaluate relevant distances between point charges.
        !!!if(le > 3) then  ! it it comes here, le should gt 3.
        ! DS.
        DB     = DD(4,NE)
        QB     = PO(4,NE)
        ADD    = (QA+QB)**2
        ! L1=0, L2=2, M=0, LABEL=6.
        X(8)   = (R-DB)**2+ADD
        X(9)   = R2+DB**2+ADD
        X(10)  = (R+DB)**2+ADD
        ! DP.
        DB     = DD(5,NE)
        QB     = PO(5,NE)
        ADD    = (QA+QB)**2
        ! L1=0, L2=1, M=0, LABEL=3.
        X(11)  = (R+DB)**2+ADD
        X(12)  = (R-DB)**2+ADD
        ! DD.
        DB     = DD(6,NE)
        QB     = PO(6,NE)
        QBS    = PO(8,NE)
        ADD    = (QA+QB)**2
        ! L1=0, L2=0, M=0, LABEL=1.
        X(13)  = R2+(QA+QBS)**2
        ! L1=0, L2=2, M=0, LABEL=6.
        X(14)  = (R-DB)**2+ADD
        X(15)  = R2+DB**2+ADD
        X(16)  = (R+DB)**2+ADD
        ilast  = 16
        !!!end if
        ! calculate inverse distances. the numerator is chosen such that the nuclear charges 
        ! are included already at this point, as well as the conversion to eV.
        factor = -COREnc*eV
        X(ifirst:ilast)   = factor/SQRT(X(ifirst:ilast))

        ! evaluate semiempirical multipole-multipole interactions XM(label).
        ! evalaute the unique core-electron attraction integrals Core_mat(I,N) from XM(label).
        !!!if(le > 3) then
        ! DS.
        XM(6) = (X(8)-X(9)*TWO+X(10))*PT25
        CORE_mat(5,N) = XM(6) * CLM11
        ! DP.
        XM(3) = (X(11)-X(12))*PT5
        CORE_mat(6,N) = XM(3) * CLM12 * SIG
        CORE_mat(8,N) = XM(3) * SIG
        ! DD.
        XM(1) =  X(13)
        XM(6) = (X(14)-X(15)*TWO+X(16))*PT25
        CORE_mat(7,N) = XM(1) +XM(6) * CLM15
        CORE_mat(9,N) = XM(1) +XM(6) * CLM21
        CORE_mat(10,N)= XM(1) +XM(6) * CLM36
        !!!end if
     end if
  end do loopN

  return
  end subroutine reppd_qmqm



  subroutine reppd_qmmm(NI,NJ,R,RI,CORE_mat,W,LIMIJ,LIMKL)
  !
  ! for MM atoms: 
  ! NJ=0, 
  ! LIMKL=1, 
  !
  ! Two-center two-electron repulsion integrals and two-center one-electron
  ! attractions in local coordinates. (General version for MNDO/d and AM1/d).
  ! 
  ! INPUT 
  ! NI       atomic number of atom A (orbitals i and j, pairs ij).
  ! NJ       atomic number of atom B (orbitals k and l, pairs kl). = 0, for mm atom
  ! R        interatomic distance (au).
  ! RI       local two-electron integrals for SP basis.
  ! CORE     local one-electron integrals for SP basis.
  ! LIMIJ    column dimension of W. 1 for S, 10 for SP, 45 for SPD; iorbs*(iorbs+1)/2
  ! LIMKL    row    dimension of W. 1 for S, 10 for SP, 45 for SPD.
  ! 
  ! On output
  ! CORE_mat complete set of local one-electron integrals.
  ! W        complete set of local two-electron integrals.
  !
  ! for MM charges
  ! Convention: ni=0 or nj=0 denotes an external point charge without basis
  !             orbitals.  The charge is 1 atomic unit. Thus, the values of
  !             DD(i,0) and PO(i,0) are defined to be zero.
  ! 
  use qm1_parameters, only : CORE,DD,PO
      
  implicit none

  integer :: NI,NJ,LIMIJ,LIMKL
  real(chm_real):: W(LIMIJ),RI(22),CORE_mat(10,2)
  real(chm_real):: R

  ! local variables:
  real(chm_real),parameter :: CLM3  = 0.13333333333333D+01,  &
                              CLM6  =-0.66666666666667D+00,  &
                              CLM10 =-0.66666666666667D+00,  &
                              CLM11 = 0.11547005383793D+01,  &
                              CLM12 = 0.11547005383793D+01,  &
                              CLM13 =-0.57735026918963D+00,  &
                              CLM15 = 0.13333333333333D+01,  &
                              CLM20 = 0.57735026918963D+00,  &
                              CLM21 = 0.66666666666667D+00,  &
                              CLM28 = 0.66666666666667D+00,  &
                              CLM33 =-0.11547005383793D+01,  &
                              CLM36 =-0.13333333333333D+01,  &
                              CLM45 =-0.13333333333333D+01,  &
                              SQRT2 = 0.14142135623731D+01,  &
                              small = 1.0D-06
  real(chm_real),parameter :: r_SQRT2 = one/SQRT2
  integer, parameter :: LCW(10)=(/ 1,2,3,6,11,12,15,18,21,36/)

  real(chm_real):: X(405),XM(14),factor
  integer       :: LX(6)
  integer       :: I,J,K,L,N,IJ,KL,LIJ,IFIRST,ILAST,NC,NE,LE,LIJMAX,LKLMAX,ICMAX
  real(chm_real):: R2,SIG,AA,AB,CORENC,DA,QA,DB,QB,QAS,QBS,DA2,  &
                   ADD,ADDAS,ADDBS,ADDSS,RMDA2,RPDA2,XYXY,ZZZZ


  ! initialization.
  ! since LIMKL=1 and NJ=0, LIMIJ and NI should be 45 (iorbs=9).
  w(1:LIMIJ)=zero
  LIJMAX = 6  ! limij.ne.1 .or. ne.10
  LKLMAX = 1  ! limkl.eq.1
  ! at least one of LIJMAX and LKLMAX should be "6".

  ! distance between point charges.
  R2  = R*R
  I   = 0
  QB  = PO(1,NJ)
  loopLIJ1: do LIJ=4,LIJMAX  ! 1,LIJMAX
     DA     = DD(LIJ,ni)
     QA     = PO(LIJ,ni)
     !
     DA2    = DA*DA
     RMDA2  = (R-DA)**2
     RPDA2  = (R+DA)**2
     ADD    = (QA+QB)**2

     !if(LIJ.eq.6) then
     !   QAS = PO(8,ni)
     !else
     !   QAS = PO(1,ni)
     !end if

     ! for LIJ<=3, nothing belongs here.
     ! (DS,KL), LIJ=4.
     if(LIJ.eq.4) then
        ! KL=SS.
        LX(LIJ)= I
        !QB     = PO(1,NJ)
        !ADD    = (QA+QB)**2
        ! L1=2, L2=0, M=0, LABEL=7.
        X(I+1) = RMDA2+ADD
        X(I+2) = R2+DA2+ADD
        X(I+3) = RPDA2+ADD
        I      = I+3
     ! end of LJI.eq.4

     ! (DP,KL), LIJ=5.
     else if(LIJ.eq.5) then
        ! KL=SS.
        LX(LIJ)= I
        !QB     = PO(1,NJ)
        !ADD    = (QA+QB)**2
        ! L1=1, L2=0, M=0, LABEL=2.
        X(I+1) = RPDA2+ADD
        X(I+2) = RMDA2+ADD
        I      = I+2
     ! end of LIJ.eq.5

     ! (DD,KL), LIJ=6.
     else if(LIJ.eq.6) then
        ! KL=SS.
        LX(LIJ)= I
        !QB     = PO(1,NJ)
        !ADD    = (QA+QB)**2
        !ADDAS  = (QAS+QB)**2
        ADDAS  = (PO(8,ni)+QB)**2
        ! L1=0, L2=0, M=0, LABEL=1.
        X(I+1) = R2+ADDAS
        ! L1=2, L2=0, M=0, LABEL=7.
        X(I+2) = RMDA2+ADD
        X(I+3) = R2+DA2+ADD
        X(I+4) = RPDA2+ADD
        I      = I+4
     end if     ! LIJ.eq.6 
  end do  loopLIJ1

  ! calculate inverse distances:  numerator ONE implies atomic units.
  !                                         EV          conversion to eV.
  ! it could be done after the following do loop at the level of the
  ! two-electron integrals W(LIMKL,LIMIJ).
  ilast = i
  X(1:ilast)   = eV/SQRT(X(1:ilast))

  !
  ! evaluate semiempirical multipole-multipole interactions XM(label). 
  ! evaluate the unique integrals W(KL,IJ) from XM(label).
  ! define the remaining symmetry-related integrals W(KL,IJ).
  loopLIJ2: do LIJ=4,LIJMAX
     ! (DS,KL), LIJ=4.
     if (LIJ.eq.4) then
        ! KL=SS.
        I      = LX(LIJ)
        XM(7) = (X(I+1)-X(I+2)*TWO+X(I+3))*PT25
        W(11) = XM( 7) * CLM11
     ! end of LIJ.eq.4

     ! (DP,KL), LIJ=5.
     else if (LIJ.eq.5) then 
        ! KL=SS.
        I      = LX(LIJ)
        XM(2) =-(X(I+1)-X(I+2))*PT5
        W(12) = XM( 2) * CLM12
        W(18) = XM( 2)
        W(25) = XM( 2)
     ! end of LIJ.eq.5

     ! (DD,KL), LIJ=6.
     else if (LIJ.eq.6) then
        ! KL=SS.
        I      = LX(LIJ)
        XM(1) =  X(I+1)
        XM(7) = (X(I+2)-X(I+3)*TWO+X(I+4))*PT25
        W(15) = XM( 1) +XM( 7) * CLM15
        W(21) = XM( 1) +XM( 7) * CLM21
        W(36) = XM( 1) +XM( 7) * CLM36
        W(28) = W(21)
        W(45) = W(36)
     end if   ! end of LIJ.eq.6 
  end do loopLIJ2

  !
  ! include two-electron integrals involving only s and p orbitals.
  ! Currently, local integrals RI(22) from subroutine REPP.
  W( 1) = RI(1)
  if(LIJMAX.gt.1) then
     W( 2) = RI( 2)
     W( 3) = RI( 3)
     W( 6) = RI( 4)
     W(10) = RI( 4)
  end if
  
  !
  ! core-electron attraction integrals.
  ! CORE_mat(i,1) - electrons (ij) at atom A and core of atom B (nc=nj).
  ! CORE_mat(i,2) - electrons (kl) at atom B and core of atom A (nc=ni).
  !
  ! index i in CORE_mat(i,1) and CORE_mat(i,2).
  ! (S S )=1, (S P )=2, (P P )=3, (P+P+)=4
  ! (D S )=5, (D P )=6, (D D )=7, (D+P+)=8, (D+D+)=9, (D#D#)=10.
  ! notation: P-SIGMA=P , P-PI=P+, D-SIGMA=D , D-PI=D+, D-DELTA=D#.
  !
  ! loop over the two possible cases.
  ! NC   atomic number of atom carrying the core.
  ! NE   atomic number of atom carrying the electrons.
  ! LE   limit for L pair index, 1 for S, 3 for SP, 6 for SPD.
  ! SIG  sign factor for integrals, always positive except for interations between (ij) at 
  !      atom A involving one P-sigma or D-Pi orbitail and the core of atom B (case N=1).
  !
  ! since nj=0 (mm atom), n=2 will exit right away. so, no need of the do-loop.
  ! only do n=1 case
  ifirst = 8  
  N   = 1      ! for do-loop.
  !NC  = nj     ! = 0
  NE  = ni     !
  LE  = LIJMAX ! = 6
  SIG =-one
  COREnc = one ! as NC==0
  QA     = PO(9,nj)   ! additive term for core.
  !
  ! check for neglect of penetration integrals. if the additive terms for SS (PO(1,nc)) 
  ! and the core (PO(9,nc)) are equal, the core-electron attraction integrals are 
  ! expressed in terms of the two-electron integrals (SS,KL) and (IJ,SS). 
  ! LCW(i) is the standard pair index corressponding to i in CORE_mat(i,n).
  if(abs(QA-PO(1,nj)).lt.small) then  ! for atoms iorbs<=4, PO(9,NC)=PO(1,NC)
     !if(le.eq.1) then                 !           see fill_qm_parameters.
     !   icmax = 1                     ! therefore, for the current n, if it
     !else if(le.eq.3) then            ! belongs here, the other n should be
     !   icmax = 4                     ! the norbs=9 (LE=6).
     !else
     icmax = 10
     !end if
     !if(n.eq.1) then
     CORE_mat(1:icmax,1) = -COREnc*W(LCW(1:icmax))
     !else
     !   CORE_mat(1:icmax,2) = -COREnc*W(LCW(1:icmax))
     !end if
  else
  ! the following will be unlikely happening, as PO(1:9,for mm atom) is zero,
  ! but keep here, just in case.

     ! evaluate relevant distances between point charges.
     !!!if(le.gt.3) then  ! it it comes here, le should gt 3.
     ! DS.
     DB     = DD(4,NE)
     QB     = PO(4,NE)
     ADD    = (QA+QB)**2
     ! L1=0, L2=2, M=0, LABEL=6.
     X(8)   = (R-DB)**2+ADD
     X(9)   = R2+DB**2+ADD
     X(10)  = (R+DB)**2+ADD
     ! DP.
     DB     = DD(5,NE)
     QB     = PO(5,NE)
     ADD    = (QA+QB)**2
     ! L1=0, L2=1, M=0, LABEL=3.
     X(11)  = (R+DB)**2+ADD
     X(12)  = (R-DB)**2+ADD
     ! DD.
     DB     = DD(6,NE)
     QB     = PO(6,NE)
     QBS    = PO(8,NE)
     ADD    = (QA+QB)**2
     ! L1=0, L2=0, M=0, LABEL=1.
     X(13)  = R2+(QA+QBS)**2
     ! L1=0, L2=2, M=0, LABEL=6.
     X(14)  = (R-DB)**2+ADD
     X(15)  = R2+DB**2+ADD
     X(16)  = (R+DB)**2+ADD
     ilast  = 16
     !!!!end if
     ! calculate inverse distances. the numerator is chosen such that the nuclear charges 
     ! are included already at this point, as well as the conversion to eV.
     factor = -COREnc*eV
     X(ifirst:ilast)   = factor/SQRT(X(ifirst:ilast))

     ! evaluate semiempirical multipole-multipole interactions XM(label).
     ! evalaute the unique core-electron attraction integrals Core_mat(I,N) from XM(label).
     !!!!if(le.gt.3) then
     ! DS.
     XM(6) = (X(8)-X(9)*TWO+X(10))*PT25
     CORE_mat(5,1) = XM(6) * CLM11
     ! DP.
     XM(3) = (X(11)-X(12))*PT5
     CORE_mat(6,1) = XM(3) * CLM12 * SIG
     CORE_mat(8,1) = XM(3) * SIG
     ! DD.
     XM(1) =  X(13)
     XM(6) = (X(14)-X(15)*TWO+X(16))*PT25
     CORE_mat( 7,1)= XM(1) +XM(6) * CLM15
     CORE_mat( 9,1)= XM(1) +XM(6) * CLM21
     CORE_mat(10,1)= XM(1) +XM(6) * CLM36
     !!!end if
  end if

  return
  end subroutine reppd_qmmm



  subroutine rotd(W,YY,LIMIJ,LIMKL)
  !
  ! transformation of two-electron two-center integrals (SPD basis) from the
  ! local to the molecular coordinate system. A two-step procedure is used.
  !
  !use number, only : zero
  use qm1_parameters, only : R2CENT

  implicit none

  integer :: LIMIJ,LIMKL
  real(chm_real):: W(LIMKL,LIMIJ),YY(15,45)

  ! local variables
  integer :: i,j,ij,k,KL
  real(chm_real):: wrepp
  !
  real(chm_real),parameter :: vsmall=1.0d-10
  !
  integer, parameter:: limy=45, limv=45
  real(chm_real):: V(limv,limv)
  !
  ! MET assigns a pair type (1 SS, 2 PS, 3 PP, 4 DS, 5 DP, 6 DD)
  ! MET(kl) to each standard pair index kl=indx(k)+l.
  integer,parameter:: MET(45)=(/1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, &
                                4, 5, 5, 5, 6, 6, 4, 5, 5, 5, 6, 6, 6, 4, 5, &
                                5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6/)
  !
  ! map the array index belw.
  integer, parameter:: ii1(3)=(/2,4,7/),                       &
                       ii2(6)=(/3,5,6,8,9,10/),                &
                       ii3(5)=(/11,16,22,29,37/),              &
                       ii4(15)=(/12,13,14,17,18,19,23,24,25,30,31,32,38,39,40/), &
                       ii5(15)=(/15,20,21,26,27,28,33,34,35,36,41,42,43,44,45/)

  !
  ! definition of indices for intermediate integrals V(kl,ij),
  ! final integrals W(kl,ij), and transformation matrix YY(m,kl).
  ! ij    standard pair index.
  ! kl    standard pair index.
  ! met   consecutive index (up to 1 for kl=SS, 3 for kl=PS, 6 for kl=PP, 
  !                          5 for kl=DS, 15 for kl=PD and kl=DD).
  ! first step - transformation for indices kl in (kl,ij).
  ! 
  V(1:LIMKL,1:LIMIJ)=zero
  !
  ! loop over outer two indices ij (not transformed).
  do ij=1,limij
     ! loop over inner two indices kl (transformed).
     do kl=1,limkl
        if(R2CENT(kl,ij)) then
           wrepp    = w(kl,ij)
           w(kl,ij) = zero
           select case(MET(kl))
             case (1)
               V(1,ij)  =  wrepp
             case (2)
               V(ii1(1:3) ,ij)= V(ii1(1:3) ,ij)+ YY(1:3 ,kl)*wrepp
             case (3)
               V(ii2(1:6) ,ij)= V(ii2(1:6) ,ij)+ YY(1:6 ,kl)*wrepp
             case (4)
               V(ii3(1:5) ,ij)= V(ii3(1:5) ,ij)+ YY(1:5 ,kl)*wrepp
             case (5)
               V(ii4(1:15),ij)= V(ii4(1:15),ij)+ YY(1:15,kl)*wrepp
             case (6) 
               V(ii5(1:15),ij)= V(ii5(1:15),ij)+ YY(1:15,kl)*wrepp
           end select
        end if
     end do
  end do

  !
  !probably, i can change the indexing of W, i.e., 
  !          swap between row and column. 
  ! if so, the indexing should be also changed in W2MAT.

  !
  ! second step - transformation for outer two indices.
  !
  ! loop over outer two indices ij (transformed).
  do ij=1,limij
     ! loop over inner two indices kl (not transformed).
     select case(MET(ij))
       case (1)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,1) = V(kl,ij)
         end do
       case (2)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,ii1(1:3)) =w(kl,ii1(1:3)) +YY(1:3 ,ij)*V(kl,ij)
         end do
       case (3)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,ii2(1:6)) =w(kl,ii2(1:6)) +YY(1:6 ,ij)*V(kl,ij)
         end do
       case (4)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,ii3(1:5)) =w(kl,ii3(1:5)) +YY(1:5 ,ij)*V(kl,ij)
         end do
       case (5)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,ii4(1:15))=w(kl,ii4(1:15))+YY(1:15,ij)*V(kl,ij)
         end do
       case (6)
         do kl=1,limkl
           if(abs(V(kl,ij)).gt.vsmall) w(kl,ii5(1:15))=w(kl,ii5(1:15))+YY(1:15,ij)*V(kl,ij)
         end do
     end select
  end do

  return
  end subroutine rotd
#endif /*mndo97*/
!
end module qm1_mndod
