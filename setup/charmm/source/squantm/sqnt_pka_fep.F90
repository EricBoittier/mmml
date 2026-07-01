#if KEY_SQUANTM==1 /*mainsquatn*/
      SUBROUTINE ChkInternal_pKa(HQM,QPKNOANG,QPKNODIH,QPKNOIMP)
!-----------------------------------------------------------------------
!     Go over Bonded terms and find the bond terms related to
!     H atoms to be annihilated thru pKa perturbation to be
!     replaced with Dummy atoms only having bond terms.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
!
  use consta
  use string
  use stream
  use psf
  use squantm
! for parallel run
#if KEY_PARALLEL==1
  use parallel  
#endif

      implicit none

      Integer HQM
      LOGICAL QPKNOANG,QPKNODIH,QPKNOIMP

      Integer MM,I,J,K,L,Ncount

! bond
      Ncount=0
      If(NBOND.GT.0) Then
         Do MM=1,NBOND
            I=IB(MM)
            J=JB(MM)
            If(I.eq.HQM.or.J.eq.HQM) then
               Ncount=Ncount+1
               If(Ncount.le.MAXBNDQ) HBND(Ncount)=MM
            End if
         End do
 100     If(Ncount.Gt.MAXBNDQ) &
            Call Wrndie(-1,'<ChkInternal>', &
           'Number of Bonds is out of range for pKa perturbation.')
      End if
      NUMBND=Ncount

! angle
      Ncount=0
      If(.not.QPKNOANG) then
         If(NTHETA.GT.0) Then
            Do MM=1,NTHETA
               I=IT(MM)
               J=JT(MM)
               K=KT(MM)
               If(I.eq.HQM.or.K.eq.HQM) then
                  Ncount=Ncount+1
                  If(Ncount.le.MAXANGQ) HANG(Ncount)=MM
               End if
            End do
 200        If(Ncount.Gt.MAXANGQ) &
               Call Wrndie(-1,'<ChkInternal>', &
           'Number of Angles is out of range for pKa perturbation.')
         End if
      End if
      NUMANG=Ncount
      

! dihedrals
      Ncount=0
      If(.not.QPKNODIH) then
         If(NPHI.GT.0) Then
            Do MM=1,NPHI
               I=IP(MM)
               J=JP(MM)
               K=KP(MM)
               L=LP(MM)
               If(I.eq.HQM.or.L.eq.HQM.or.J.eq.HQM.or.K.eq.HQM) then
                  Ncount=Ncount+1
                  If(Ncount.le.MAXDIHQ) HDIH(Ncount)=MM
               End if
            End do
 300        If(Ncount.Gt.MAXDIHQ) &
               Call Wrndie(-1,'<ChkInternal>', &
           'Number of Dihedrals is out of range for pKa perturbation.')
         End if
      End if
      NUMDIH=Ncount

! improper
      Ncount=0
      If(.not.QPKNOIMP) then
         If(NIMPHI.GT.0) Then
            Do MM=1,NIMPHI
               I=IM(MM)
               J=JM(MM)
               K=KM(MM)
               L=LM(MM)
               If(I.eq.HQM.or.L.eq.HQM.or.J.eq.HQM.or.K.eq.HQM) then
                  Ncount=Ncount+1
                  If(Ncount.le.MAXDIHQ) HIMP(Ncount)=MM
               End if
            End do
 400        If(Ncount.Gt.MAXDIHQ) &
               Call Wrndie(-1,'<ChkInternal>', &
      'Number of Improper angles is out of range for pKa perturbation.')

         End if
      End if
      NUMIMP=Ncount

! setup for internal energy calculations: parameters copying
      Call PKAINTE_SETUP


! printing output information.
      If(PRNLEV.GE.2) then
         WRITE(6,500)
         
         write(6,510)'Number of bonds:           ',NUMBND
         if(NUMBND.gt.0) then
            do mm=1,NUMBND
               i = HBND(mm)
               write(6,'(I6,2X,I6)')IB(i),JB(i)
            end do
         else
            write(6,520)
         end if
         write(6,*)'   '
         
         write(6,510)'Number of angles:          ',NUMANG
         if(NUMANG.gt.0) then
            do mm=1,NUMANG
               i = HANG(mm)
               write(6,'(I6,2(2x,I6))')IT(i),JT(i),KT(i)
            end do
         else
            write(6,520)
         end if
         write(6,*)'   '         

         write(6,510)'Number of dihedrals:       ',NUMDIH
         if(NUMDIH.gt.0) then
            do mm=1,NUMDIH
               i = HDIH(mm)
               write(6,'(I6,3(2x,I6))')IP(i),JP(i),KP(i),LP(i)
            end do
         else
            write(6,520)
         end if
         write(6,*)'   '

         write(6,510)'Number of improper angles: ',NUMIMP
         if(NUMIMP.gt.0) then
            do mm=1,NUMIMP
               i = HIMP(mm)
               write(6,'(I6,3(2x,I6))')IM(i),JM(i),KM(i),LM(i)
            end do
         else
            write(6,520)
         end if
         WRITE(6,500)
      End if
 500  FORMAT('------------------------------------------------')
 510  FORMAT('ChkInternal_PKA-FEP> ',A,I4)
 520  FORMAT(6X,'NONE.')

      Return
      END SUBROUTINE

      SUBROUTINE PKAINTE_SETUP
!-----------------------------------------------------------------------
!     Setup to compute internal valence energy terms
!     connecting Dummy H atom for pKa perturbation.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
!
  use stream
  use psf
  use code
  use param
!
  use squantm
!
      implicit none

      INTEGER MM,I,J,IC,II,ITH,IPHI
!
! all routines assume Scalar Fast routine version
! also, Urey-Bradley terms will be ignored...Unluckily.. : (

! bond energy
      If (NBOND.GT.0) then
         Do II=1,NUMBND
            MM        =HBND(II)
            IC        =ICB(MM)
            ICBQ(II)  =IC
            RBND(1,II)=CBB(IC)
            RBND(2,II)=CBC(IC)
         End do
      End if

! angle energy
      If (NTHETA.GT.0) then
          Do II=1,NUMANG
             ITH       =HANG(II)
             IC        =ICT(ITH)
             ICTQ(II)  =IC
             RANG(1,II)=CTB(IC)
             RANG(2,II)=CTC(IC)
          End do
      End if

! dihedral energy
      If (NPHI.GT.0) then
         Do II=1,NUMDIH
            IPHI      =HDIH(II)
            IC        =ICP(IPHI)
            ICPQ(II)  =IC
            IDIH(II)  =CPD(IC)
            RDIH(1,II)=CPCOS(IC)
            RDIH(2,II)=CPSIN(IC)
            RDIH(3,II)=CPC(IC)
         End do
      End if

! improper dihedral energy
      If (NIMPHI.GT.0) then
         Do II=1,NUMIMP
            IPHI       =HIMP(II)
            IC         =ICI(IPHI)
            ICIQ(II)   =IC
            IIMP(II)   =CID(IC)
            RIMPQ(1,II)=CICOS(IC)
            RIMPQ(2,II)=CISIN(IC)
            RIMPQ(3,II)=CIB(IC)
            RIMPQ(4,II)=CIC(IC)
         End do
      End if

      Return
      END SUBROUTINE

      SUBROUTINE PKAINTE(EVALN,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     Call each routine to compute internal valence energy terms
!     connecting Dummy H atom for pKa perturbation.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
!
  use psf
  use squantm

      implicit none

      real(chm_real) EVALN,Prefct,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
      real(chm_real) EBND,EANG,EDIH,EIMP

! all routines assume Scalar Fast routine version
! also, Urey-Bradley terms will be ignored...Unluckily.. : (
      EBND=ZERO
      EANG=ZERO
      EDIH=ZERO
      EIMP=ZERO

! bond energy
      If(NUMBND.GT.0) &
          Call EBONDFS_PKA(EBND,Prefct,X,Y,Z,DX,DY,DZ)

! angle energy
      If(NUMANG.GT.0) &
          Call EANGLFS_PKA(EANG,Prefct,X,Y,Z,DX,DY,DZ)

! dihedral energy
      If(NUMDIH.GT.0) &
          Call EPHIFS_PKA(EDIH,Prefct,X,Y,Z,DX,DY,DZ)

! improper dihedral energy
      If(NUMIMP.GT.0) &
          Call EIPHIFS_PKA(EIMP,Prefct,X,Y,Z,DX,DY,DZ)

      EVALN=EBND+EANG+EDIH+EIMP

      Return
      END SUBROUTINE

      SUBROUTINE PKAINTE_HMM(EVALN,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     Call each routine to compute internal valence energy terms
!     connecting Dummy H atom for pKa perturbation.
!     This is to annihilate H atom of MM angle, dihedral, and
!     improper angles.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
!
  use psf
  use squantm

      implicit none

      real(chm_real) EVALN(2),Prefct,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
      real(chm_real) EBND,EANG,EDIH,EIMP

! all routines assume Scalar Fast routine version
! also, Urey-Bradley terms will be ignored...Unluckily.. : (
      EBND=ZERO
      EANG=ZERO
      EDIH=ZERO
      EIMP=ZERO

! bond energy
      If(NUMBND.GT.0) &
          Call EBONDFS_PKA(EBND,Prefct,X,Y,Z,DX,DY,DZ)

! angle energy
      If(NUMANG.GT.0) &
          Call EANGLFS_PKA(EANG,Prefct,X,Y,Z,DX,DY,DZ)

! dihedral energy
      If(NUMDIH.GT.0) &
          Call EPHIFS_PKA(EDIH,Prefct,X,Y,Z,DX,DY,DZ)

! improper dihedral energy
      If(NUMIMP.GT.0) &
          Call EIPHIFS_PKA(EIMP,Prefct,X,Y,Z,DX,DY,DZ)

! bond will not be perturbed, and angle/dihedral/improper will be
! perturbed to be zero, thus separating the energy terms.
      EVALN(1)=EBND
      EVALN(2)=EANG+EDIH+EIMP

      Return
      END SUBROUTINE

      SUBROUTINE EBONDFS_PKA(EB,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     calculates bond energies and forces as fast as possible
!     Fast SCALAR version
!     14-JUL-1983, Bernard R. Brooks
!     31-Aug-1991, Youngdo Won
!-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
!
  use stream
!
  use psf
  use code
  use squantm
!
      implicit none

      real(chm_real) EB,Prefct
      real(chm_real) X(*),Y(*),Z(*), DX(*),DY(*),DZ(*)
!
      real(chm_real) RXYZ(3),S,R,DB,DF,DXYZ(3),RCBC,RCBB
      INTEGER MM,I,J,IC,II
!
!
      EB=ZERO
      IF (NBOND.EQ.0) RETURN
!
      DO II=1,NUMBND
         MM=HBND(II)
         I=IB(MM)
         J=JB(MM)

         If (I.GT.0) then
           IC  = 1
           RCBB=RBND(1,II)
           RCBC=RBND(2,II)

           if(IC.ne.0 .and. RCBC.ne.ZERO) then
              RXYZ(1)=X(I)-X(J)
              RXYZ(2)=Y(I)-Y(J)
              RXYZ(3)=Z(I)-Z(J)
              S=SQRT(RXYZ(1)*RXYZ(1)+RXYZ(2)*RXYZ(2)+RXYZ(3)*RXYZ(3))

              if(S.GT.ZERO) then
                 R =TWO/S
                 DB=S-RCBB
                 DF=RCBC*DB
                 EB=EB+DF*DB
!
                 DF     =DF*R
                 DXYZ(1)=RXYZ(1)*DF*Prefct
                 DXYZ(2)=RXYZ(2)*DF*Prefct
                 DXYZ(3)=RXYZ(3)*DF*Prefct
                 DX(I)  =DX(I)+DXYZ(1)
                 DY(I)  =DY(I)+DXYZ(2)
                 DZ(I)  =DZ(I)+DXYZ(3)
                 DX(J)  =DX(J)-DXYZ(1)
                 DY(J)  =DY(J)-DXYZ(2)
                 DZ(J)  =DZ(J)-DXYZ(3)
!
               End if
            End if
         End if
      END DO

      EB = EB
      RETURN
      END

      SUBROUTINE EANGLFS_PKA(ET,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     calculates bond angles and bond angle energies as fast as possible
!     Fast SCALAR version
!     14-JUL-1983, Bernard R. Brooks
!     31-Aug-1991, Youngdo Won
!-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
!
  use psf
  use code
  use stream
  use consta
  use squantm
!
      implicit none

      real(chm_real) ET,Prefct
      real(chm_real) X(*),Y(*),Z(*), DX(*),DY(*),DZ(*)
!
      real(chm_real) DRI(3),DRJ(3),DRIR(3),DRJR(3),DTRI(3),DTRJ(3)
      real(chm_real) RI,RJ,RIR,RJR
      real(chm_real) CST,AT,DA,DF,RCTB,RCTC
      INTEGER NWARN,ITH,I,J,K,IC,II
!
      ET=ZERO
      NWARN=0
      IF (NTHETA.EQ.0) RETURN
!
      DO II=1,NUMANG
         ITH=HANG(II)
         I=IT(ITH)
         J=JT(ITH)
         K=KT(ITH)

         IC = 1

         If(I.gt.0 .and. IC.ne.0) then
            RCTB=RANG(1,II)
            RCTC=RANG(2,II)

            DRI(1)=X(I)-X(J)
            DRI(2)=Y(I)-Y(J)
            DRI(3)=Z(I)-Z(J)

            DRJ(1)=X(K)-X(J)
            DRJ(2)=Y(K)-Y(J)
            DRJ(3)=Z(K)-Z(J)

            RI=SQRT(DRI(1)*DRI(1)+DRI(2)*DRI(2)+DRI(3)*DRI(3))
            RJ=SQRT(DRJ(1)*DRJ(1)+DRJ(2)*DRJ(2)+DRJ(3)*DRJ(3))

            If(RI.GT.ZERO .and. RJ.GT.ZERO) then
               RIR=ONE/RI
               RJR=ONE/RJ

               DRIR(1)=DRI(1)*RIR
               DRIR(2)=DRI(2)*RIR
               DRIR(3)=DRI(3)*RIR

               DRJR(1)=DRJ(1)*RJR
               DRJR(2)=DRJ(2)*RJR
               DRJR(3)=DRJ(3)*RJR
               CST=DRIR(1)*DRJR(1)+DRIR(2)*DRJR(2)+DRIR(3)*DRJR(3)
!
               IF (ABS(CST).GE.COSMAX) THEN
                  CST=SIGN(COSMAX,CST)
                  NWARN=NWARN+1
                  IF ((NWARN.LE.5.AND.WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
                       WRITE(OUTU,445) ITH,I,J,K
               ENDIF
 445  FORMAT(' EANGLFS> Warning: Angle',I5,' is almost linear.', &
              /' Derivatives may be affected for atoms:',3I5)
!
               AT=ACOS(CST)
               DA=AT-RCTB
               DF=RCTC*DA
!
               ET= ET+DF*DA
               DF=-TWO*DF/SIN(AT)
!
               DTRI(1)=RIR*(DRJR(1)-CST*DRIR(1))
               DTRI(2)=RIR*(DRJR(2)-CST*DRIR(2))
               DTRI(3)=RIR*(DRJR(3)-CST*DRIR(3))

               DTRJ(1)=RJR*(DRIR(1)-CST*DRJR(1))
               DTRJ(2)=RJR*(DRIR(2)-CST*DRJR(2))
               DTRJ(3)=RJR*(DRIR(3)-CST*DRJR(3))
!
               DRI(1)=DF*DTRI(1)*Prefct
               DRI(2)=DF*DTRI(2)*Prefct
               DRI(3)=DF*DTRI(3)*Prefct
               DRJ(1)=DF*DTRJ(1)*Prefct
               DRJ(2)=DF*DTRJ(2)*Prefct
               DRJ(3)=DF*DTRJ(3)*Prefct

               DX(I) =DX(I)+DRI(1)
               DX(K) =DX(K)+DRJ(1)
               DX(J) =DX(J)-DRI(1)-DRJ(1)

               DY(I) =DY(I)+DRI(2)
               DY(K) =DY(K)+DRJ(2)
               DY(J) =DY(J)-DRI(2)-DRJ(2)

               DZ(I) =DZ(I)+DRI(3)
               DZ(K) =DZ(K)+DRJ(3)
               DZ(J) =DZ(J)-DRI(3)-DRJ(3)

            End if
         End if
      END DO

      ET = ET
!
      RETURN
      END

      SUBROUTINE EPHIFS_PKA(EP,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     Fast SCALAR version dihedral energy and force routine.
!     Dihedral energy terms are expressed as a function of PHI.
!     This avoids all problems as dihedrals become planar.
!     The functional form is:
!
!     E = K*(1+COS(n*Phi-Phi0))
!     Where:
!     n IS A POSITIVE INTEGER COS(n(Phi-Phi0) AND SIN(n(Phi-Phi0)
!            ARE CALCULATED BY RECURENCE TO AVOID LIMITATION ON n
!     K IS THE FORCE CONSTANT IN kcal/mol/rad
!     Phi0/n IS A MAXIMUM IN ENERGY.
!
!     The parameter of the routine is:
!     EP: Diheral Energy returned
!     Data come form param.f90
!
!     31-Aug-1991, Youngdo Won
!
!     New formulation by
!           Arnaud Blondel    1994
!
!-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
!
  use code
  use consta
  use psf
  use stream
!
  use squantm

      implicit none

      real(chm_real) EP,Prefct
      real(chm_real) X(*),Y(*),Z(*), DX(*),DY(*),DZ(*)
!
      real(chm_real)  FR(3),GR(3),HR(3),AR(3),BR(3),DFR(3),DGR(3),DHR(3)
      real(chm_real)  RA2,RB2,RG,RGR,RA2R,RB2R,RABR,CT,ST
      real(chm_real)  GAA,GBB,FG,HG,FGA,HGB
      real(chm_real)  E,DF,E1,DF1,DDF1,ARG
      real(chm_real)  RCPCOS,RCPSIN,RCPC
      INTEGER NWARN,IPHI,I,J,K,L,IC,IPER,NPER,II,ICPD
      LOGICAL LREP
!
      real(chm_real)  RXMIN,RXMIN2
      PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
!
!     Initialize the torsion energy to zero
      EP=ZERO
      IF (NPHI.EQ.0) RETURN
      NWARN=0
!
      DO II=1,NUMDIH
         IPHI= HDIH(II)
         I   = IP(IPHI)
         J   = JP(IPHI)
         K   = KP(IPHI)
         L   = LP(IPHI)

         ICPD  =IDIH(II)
         RCPCOS=RDIH(1,II)
         RCPSIN=RDIH(2,II)
         RCPC  =RDIH(3,II)

         IC = 1

         If(IC.NE.0 .or. ICPD.NE.0) then
!  F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
            FR(1)=X(I)-X(J)
            FR(2)=Y(I)-Y(J)
            FR(3)=Z(I)-Z(J)
            GR(1)=X(J)-X(K)
            GR(2)=Y(J)-Y(K)
            GR(3)=Z(J)-Z(K)
            HR(1)=X(L)-X(K)
            HR(2)=Y(L)-Y(K)
            HR(3)=Z(L)-Z(K)
! A=F^G, B=H^G.
            AR(1)=FR(2)*GR(3)-FR(3)*GR(2)
            AR(2)=FR(3)*GR(1)-FR(1)*GR(3)
            AR(3)=FR(1)*GR(2)-FR(2)*GR(1)
            BR(1)=HR(2)*GR(3)-HR(3)*GR(2)
            BR(2)=HR(3)*GR(1)-HR(1)*GR(3)
            BR(3)=HR(1)*GR(2)-HR(2)*GR(1)
!
            RA2=AR(1)*AR(1)+AR(2)*AR(2)+AR(3)*AR(3)
            RB2=BR(1)*BR(1)+BR(2)*BR(2)+BR(3)*BR(3)
            RG =SQRT(GR(1)*GR(1)+GR(2)*GR(2)+GR(3)*GR(3))
! Warnings have been simplified.
            If((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
               NWARN=NWARN+1
               IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
                    WRITE(OUTU,20) IPHI,I,J,K,L
   20    FORMAT(' EPHIFS: WARNING.  dihedral',I5,' is almost linear.'/ &
                 ' derivatives may be affected for atoms:',4I5)
            Else
!
               RGR =ONE/RG
               RA2R=ONE/RA2
               RB2R=ONE/RB2
               RABR=SQRT(RA2R*RB2R)
! CT=cos(phi)
               CT=(AR(1)*BR(1)+AR(2)*BR(2)+AR(3)*BR(3))*RABR
!
! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
               ST=RG*RABR*(AR(1)*HR(1)+AR(2)*HR(2)+AR(3)*HR(3))
!
!     Energy and derivative contributions.
               E =ZERO
               DF=ZERO

 30            CONTINUE

               IPER=ICPD
               If(IPER.GE.0) then
                  LREP=.FALSE.
               Else
                  LREP=.TRUE.
                  IPER=-IPER
               End if
!
               E1  =ONE
               DF1 =ZERO
               DDF1=ZERO
!alculation of cos(n*phi-phi0) and sin(n*phi-phi0).
               Do NPER=1,IPER
                  DDF1=E1*CT-DF1*ST
                  DF1 =E1*ST+DF1*CT
                  E1  =DDF1
               End do
               E1 = E1*RCPCOS +DF1*RCPSIN
               DF1= DF1*RCPCOS-DDF1*RCPSIN
               DF1=-IPER*DF1
               E1 = ONE+E1
!brb...03-Jul-2004 Zero-period dihedral bugfix
               IF(IPER.EQ.0) E1=ONE
!
               ARG=RCPC
               E  =E+ARG*E1
               DF =DF+ARG*DF1
!
               IF(LREP) THEN
                  IC=IC+1
                  GOTO 30
               ENDIF
!
!     Cumulate the energy
               EP=EP+E
!
!     Compute derivatives wrt catesian coordinates.
!
! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
!  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
               FG  = FR(1)*GR(1)+FR(2)*GR(2)+FR(3)*GR(3)
               HG  = HR(1)*GR(1)+HR(2)*GR(2)+HR(3)*GR(3)

               RA2R= DF*RA2R
               RB2R= DF*RB2R
               FGA = FG*RA2R*RGR
               HGB = HG*RB2R*RGR
               GAA = RA2R*RG
               GBB = RB2R*RG
! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
               DFR(1)=-GAA*AR(1)*Prefct
               DFR(2)=-GAA*AR(2)*Prefct
               DFR(3)=-GAA*AR(3)*Prefct
               DGR(1)= FGA*AR(1)*Prefct - HGB*BR(1)*Prefct
               DGR(2)= FGA*AR(2)*Prefct - HGB*BR(2)*Prefct
               DGR(3)= FGA*AR(3)*Prefct - HGB*BR(3)*Prefct
               DHR(1)= GBB*BR(1)*Prefct
               DHR(2)= GBB*BR(2)*Prefct
               DHR(3)= GBB*BR(3)*Prefct
! Distribute over Ri.
               DX(I)=DX(I)+DFR(1)
               DY(I)=DY(I)+DFR(2)
               DZ(I)=DZ(I)+DFR(3)
               DX(J)=DX(J)-DFR(1)+DGR(1)
               DY(J)=DY(J)-DFR(2)+DGR(2)
               DZ(J)=DZ(J)-DFR(3)+DGR(3)
               DX(K)=DX(K)-DHR(1)-DGR(1)
               DY(K)=DY(K)-DHR(2)-DGR(2)
               DZ(K)=DZ(K)-DHR(3)-DGR(3)
               DX(L)=DX(L)+DHR(1)
               DY(L)=DY(L)+DHR(2)
               DZ(L)=DZ(L)+DHR(3)
!
            End if
         End if
      END DO

      EP = EP
!
      RETURN
      END

      SUBROUTINE EIPHIFS_PKA(EIP,Prefct,X,Y,Z,DX,DY,DZ)
!-----------------------------------------------------------------------
!     Fast SCALAR version improper torsion energy and force routine.
!
!     For improper dihedrals, the energy term is given by:
!     E = K*(Phi-phi0)**2
!     WHERE
!     K IS THE FORCE CONSTANT IN kcal/mol/rad
!     Phi0 IS THE MINIMUM IN ENERGY.
!
!     The parameter of the routine is:
!     EPI: Diheral Energy returned
!     Data come form param.f90
!
!     31-Aug-1991, Youngdo Won
!
!     New formulation by
!           Arnaud Blondel    1994
!
!-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
!
  use psf
  use code
  use consta
  use stream
!
  use squantm
!
      implicit none

      real(chm_real) EIP,Prefct
      real(chm_real) X(*),Y(*),Z(*), DX(*),DY(*),DZ(*)
!
      real(chm_real) FR(3),GR(3),HR(3),AR(3),BR(3),DFR(3),DGR(3),DHR(3)
      real(chm_real) RA2,RB2,RG,RGR,RA2R,RB2R,RABR,CT,AP,ST,CA,SA
      real(chm_real) GAA,GBB,FG,HG,FGA,HGB
      real(chm_real) E,DF,RCICOS,RCISIN,RCIB,RCIC
      INTEGER NWARN,NWARNX,IPHI,I,J,K,L,IC,II,ICID
!
      real(chm_real)  RXMIN,RXMIN2
      PARAMETER (RXMIN=0.005D0,RXMIN2=0.000025D0)
!
      EIP=ZERO
      IF (NIMPHI.EQ.0) RETURN
      NWARN=0
      NWARNX=0
!
      DO II=1,NUMIMP
         IPHI=HIMP(II)
         I=IM(IPHI)
         J=JM(IPHI)
         K=KM(IPHI)
         L=LM(IPHI)

         ICID  =IIMP(II)
         RCICOS=RIMPQ(1,II)
         RCISIN=RIMPQ(2,II)
         RCIB  =RIMPQ(3,II)
         RCIC  =RIMPQ(4,II)

         IC = 1

         If (IC.ne.0) then

#if KEY_OPLS==0
            If (ICID.NE.0) then
              CALL WRNDIE(-3,'<EIPHIFS>', &
              'Bad periodicity value for improper dihedral angles.')
            Else
#endif 
! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
               FR(1)=X(I)-X(J)
               FR(2)=Y(I)-Y(J)
               FR(3)=Z(I)-Z(J)
               GR(1)=X(J)-X(K)
               GR(2)=Y(J)-Y(K)
               GR(3)=Z(J)-Z(K)
               HR(1)=X(L)-X(K)
               HR(2)=Y(L)-Y(K)
               HR(3)=Z(L)-Z(K)
! A=F^G, B=H^G.
               AR(1)=FR(2)*GR(3)-FR(3)*GR(2)
               AR(2)=FR(3)*GR(1)-FR(1)*GR(3)
               AR(3)=FR(1)*GR(2)-FR(2)*GR(1)
               BR(1)=HR(2)*GR(3)-HR(3)*GR(2)
               BR(2)=HR(3)*GR(1)-HR(1)*GR(3)
               BR(3)=HR(1)*GR(2)-HR(2)*GR(1)
!
               RA2=AR(1)*AR(1)+AR(2)*AR(2)+AR(3)*AR(3)
               RB2=BR(1)*BR(1)+BR(2)*BR(2)+BR(3)*BR(3)
               RG =SQRT(GR(1)*GR(1)+GR(2)*GR(2)+GR(3)*GR(3))
! Warnings have been simplified.
               If((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2) &
                                 .OR.(RG.LE.RXMIN)  ) then
                  NWARN=NWARN+1
                  if((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) &
                      WRITE(OUTU,20) IPHI,I,J,K,L
 20    FORMAT(' EIPHIFS: WARNING.  dihedral',I5,' is almost linear.'/ &
              ' derivatives may be affected for atoms:',4I5)
               Else
!
                  RGR = ONE/RG
                  RA2R= ONE/RA2
                  RB2R= ONE/RB2
                  RABR= SQRT(RA2R*RB2R)
! CT=cos(phi)
                  CT=(AR(1)*BR(1)+AR(2)*BR(2)+AR(3)*BR(3))*RABR
!
! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
                  ST=RG*RABR*(AR(1)*HR(1)+AR(2)*HR(2)+AR(3)*HR(3))
!
!alcul of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
                  CA=CT*RCICOS+ST*RCISIN
                  SA=ST*RCICOS-CT*RCISIN
                  if (CA.GT.PTONE ) then
                     AP=ASIN(SA)
                  else
                     AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
! Warning is now triggered at deltaphi=84.26...deg (used to be 90).
                     NWARNX=NWARNX+1
                     if((NWARNX.LE.5 .AND. WRNLEV.GE.5) &
                                     .OR. (WRNLEV.GE.6)) then
                        WRITE(OUTU,80) IPHI,AP*RADDEG,RCIB*RADDEG, &
                                       I,J,K,L
 80    FORMAT(' EPHI: WARNING. bent improper torsion angle', &
              ' is '//'far ','from minimum for;'/3X,' IPHI=',I5, &
              '  with deltaPHI=',F9.4,' MIN=',F9.4,' ATOMS:',4I5)
                     end if
                  end if
!
                  DF=RCIC*AP
                  E=DF*AP
                  DF=TWO*DF
!
                  EIP=EIP+E
!
!     Compute derivatives wrt catesian coordinates.
!
! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
!  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
                  FG=FR(1)*GR(1)+FR(2)*GR(2)+FR(3)*GR(3)
                  HG=HR(1)*GR(1)+HR(2)*GR(2)+HR(3)*GR(3)
                  RA2R=DF*RA2R
                  RB2R=DF*RB2R
                  FGA=FG*RA2R*RGR
                  HGB=HG*RB2R*RGR
                  GAA=RA2R*RG
                  GBB=RB2R*RG
! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
                  DFR(1)=-GAA*AR(1)*Prefct
                  DFR(2)=-GAA*AR(2)*Prefct
                  DFR(3)=-GAA*AR(3)*Prefct
                  DGR(1)= FGA*AR(1)*Prefct - HGB*BR(1)*Prefct
                  DGR(2)= FGA*AR(2)*Prefct - HGB*BR(2)*Prefct
                  DGR(3)= FGA*AR(3)*Prefct - HGB*BR(3)*Prefct
                  DHR(1)= GBB*BR(1)*Prefct
                  DHR(2)= GBB*BR(2)*Prefct
                  DHR(3)= GBB*BR(3)*Prefct
! Distribute over Ri.
                  DX(I)=DX(I)+DFR(1)
                  DY(I)=DY(I)+DFR(2)
                  DZ(I)=DZ(I)+DFR(3)
                  DX(J)=DX(J)-DFR(1)+DGR(1)
                  DY(J)=DY(J)-DFR(2)+DGR(2)
                  DZ(J)=DZ(J)-DFR(3)+DGR(3)
                  DX(K)=DX(K)-DHR(1)-DGR(1)
                  DY(K)=DY(K)-DHR(2)-DGR(2)
                  DZ(K)=DZ(K)-DHR(3)-DGR(3)
                  DX(L)=DX(L)+DHR(1)
                  DY(L)=DY(L)+DHR(2)
                  DZ(L)=DZ(L)+DHR(3)
!
               End if
#if KEY_OPLS==0
            End if
#endif 
         End if
      END DO

      EIP = EIP

      RETURN
      END


      SUBROUTINE PRQFEP_SQ(NUMSTP)
!
!      PRINTS OUT THE QM-MM ELECTROSTATIC FREE ENERGY PERTURBATION
!      RESULTS.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
!
  use squantm

      implicit none

      INTEGER NUMSTP
!
      IF(QMFEP .OR. QMSOLV .and. (PRNLEV.GE.2)) THEN
         WRITE(OUTU,'(1X,A,I8,A)') &
         'QM-MM Electrostatic Free Energy Perturbation for', &
         NUMSTP,' Steps '
         WRITE(OUTU,'(A)') &
       '    L(ref)  ->  L(pert)     Delta G(ref->pert)'
         WRITE(OUTU,15) lambda_qm(1),lambda_qm(2),EQPRA(QPT1)
         WRITE(OUTU,15) lambda_qm(1),lambda_qm(3),EQPRA(QPT2)
      ENDIF
   15 FORMAT(4X,F5.3,7X,F5.3,10X,F7.3,10X,F7.3)

      RETURN
      END

      SUBROUTINE PRPKA_FEP(NUMSTP)
!-----------------------------------------------------------------------
!     Print out the QM-PKA FREE ENERGY PERTURBATION RESULTS.
!
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
!
  use sizes
  use squantm

      implicit none
      INTEGER NUMSTP
!
      IF(QMFEP .and. QpKa_on .and. (PRNLEV.GE.2)) THEN
         WRITE(OUTU,'(1X,A,I8,A)') &
         'QM/MM pKa Free Energy Perturbation for', NUMSTP,' Steps'
         WRITE(OUTU,'(A)') &
         '    L(ref)  ->  L(pert)     Delta G(ref->pert)'
         WRITE(OUTU,15) lambda_qm(1),lambda_qm(2),EQPRA(QPT1)
         WRITE(OUTU,15) lambda_qm(1),lambda_qm(3),EQPRA(QPT2)
 15   FORMAT(4X,F5.3,7X,F5.3,10X,F7.3,10X,F7.3)
      END IF

#else /*  (mainsquatn)*/
!
      SUBROUTINE ChkInternal_pKa_BLANK
      !
#endif /*  (mainsquatn)*/
      RETURN
      END 

