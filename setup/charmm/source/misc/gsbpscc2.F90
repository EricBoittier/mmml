SUBROUTINE NULL_SCCRECT
  RETURN
END SUBROUTINE NULL_SCCRECT
#if KEY_PBEQ==1
#if KEY_GSBP==1
#if KEY_SCCDFTB==1
!
!      QC: Added for GSBP-SCC for RECT case - useful for membrane simu-
!      lations

SUBROUTINE SCCRECT0(X,Y,Z,NTPOL,MAXNPOL,XNPOL,YNPOL,ZNPOL, &
     RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
     MIJ,COEF,COEFX,MQ,COEF_B, &
     LSTPX,LSTPY,LSTPZ,ALPX,ALPY,ALPZ,BNORM,LSTPOL, &
     NCLX,NCLY,NCLZ,DCEL, &
     PHI1,TRANX,TRANY,TRANZ, &
     XBCEN,YBCEN,ZBCEN,FACTOR, QBSPL)
  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use consta
  use sccdftb
  use sccgsbp
  use pbeq,only:m3,dm3,lpol2,dlpol2

  implicit none

  INTEGER LSTPOL(*),NTPOL,MAXNPOL
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real)  COEF(*),COEFX(*),MIJ(*),BNORM(*),MQ(*)
  real(chm_real)  RRXCEN,RRYCEN,RRZCEN
  real(chm_real)  XSCALE,YSCALE,ZSCALE 
  INTEGER LSTPX(*),LSTPY(*),LSTPZ(*)
  INTEGER XNPOL,YNPOL,ZNPOL
  real(chm_real)  ALPX(*),ALPY(*),ALPZ(*) 
  !
  integer NCLX,NCLY,NCLZ
  real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
  real(chm_real4)  phi1(*)
  real(chm_real)  FACTOR
  real(chm_real)  COEF_B(NTPOL,*)
  LOGICAL QBSPL
  ! local
  INTEGER I,J,II,JJ,IJ,L,M,LL,MM,LMAXSCC,JQ
  real(chm_real)  NORM
  real(chm_real)  XDIFF,YDIFF,ZDIFF
  INTEGER XPOL,YPOL,ZPOL 
  !
  integer ncyz,il,ix,iy,iz,n1,in1,n2,in2,n3,in3
  real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
  real(chm_real)  enet1, enelp1, enet2, enelp2    
  real(chm_real)  aisign,bisign,cisign,prefac

  ! B-spline
  integer jx1,jx2,jy1,jy2,jz1,jz2,k,ipx,ipy,ipz,nfil
  real(chm_real)  xc,yc,zc !,M3,DM3 
  !      

  !      Evaluate arrays required in SCF iterations.
  !      The GAMA and OMEGA arrays. These will be save in the /sccgs/
  !      common block - which is O.K. for now due to the typically 
  !      small QM size. Improve later with heap.

  !      -------- Stage I, build Gamma(Ra,Rb) --------
  !      where Gamma(Ra,Rb) = Sum(m,n) {b_m(Ra)*M_mn*b_n(Rb)}
  !      Gamma contributes to both SCF (Hamiltonian) and Energy, as well
  !      as to the derivatives of the QM atoms 

  !     I.1.-->>>Calculate M_mn*b_n(Rb) (STORED as COEF_B in /sccgsbp/)

  DO JJ=1,NSCCTC
     DO II=1,NTPOL
        COEF_B(II,JJ)=ZERO
     ENDDO
  ENDDO

  DO 99 I=1,NSCCTC
     !        assume no replica for now
     J =IQMLST(I,1)
     XDIFF=X(J)-RRXCEN
     YDIFF=Y(J)-RRYCEN
     ZDIFF=Z(J)-RRZCEN

     CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
     CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
     CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)

     !        Multiply M_mn*b_n(Rb), and sum over n
     DO JJ=1,NTPOL
        DO II=1,NTPOL
           IJ=(JJ-1)*NTPOL + II
           IF (JJ.GT.II) IJ=(II-1)*NTPOL + JJ
           XPOL=LSTPX(II)
           YPOL=LSTPY(II)
           ZPOL=LSTPZ(II)
           NORM=BNORM(II)
           COEF_B(JJ,I)=COEF_B(JJ,I)+MIJ(IJ)*NORM* &
                ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)

        ENDDO ! sum over n
     ENDDO ! loop over m
99 ENDDO ! loop over Rb

  !     I.2.-->>>Calculate b_m(Ra)*COEF_B(m,Rb) sum over m
  !     save the lower triangle 

  IJ = 0 
  DO I=1,NSCCTC
     DO J=1,I
        IJ=IJ+1
        GAMAGS(IJ)=ZERO

        JQ =IQMLST(I,1)
        XDIFF=X(JQ)-RRXCEN
        YDIFF=Y(JQ)-RRYCEN
        ZDIFF=Z(JQ)-RRZCEN
        CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
        CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
        CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)

        !        Multiply b_m(Ra)*COEF_B(m,Rb), and sum over m
        DO II=1,NTPOL
           XPOL=LSTPX(II)
           YPOL=LSTPY(II)
           ZPOL=LSTPZ(II)
           NORM=BNORM(II)
           GAMAGS(IJ) = GAMAGS(IJ)  + NORM* &
                ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)*COEF_B(II,J)
        ENDDO ! sum over m

     ENDDO ! loop over Rb
  ENDDO ! loop over Ra

  !      ------- Stage II, build Omega(Ra) --------
  !      where Omega(Ra)=Phi^(o)(Ra) + Sum(m,n){b_m(Ra)*M_mn*Q^MM_n}
  !      Omega contributes to both SCF (Hamiltonian) and Energy, as well
  !      as to the derivatives of the QM atoms

  !      II.1.-->>> Compute MQ_m
  DO I=1,MAXNPOL
     II=LSTPOL(I) 
     MQ(II)=ZERO     
     DO J=1,MAXNPOL
        JJ=LSTPOL(J) 
        IJ=(II-1)*NTPOL+JJ
        IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
        MQ(II)=MQ(II)+MIJ(IJ)*(COEF(JJ)-COEFX(JJ))
     ENDDO
  ENDDO

  !      II.2.-->>> Compute b_m(Ra)*MQ_m sum over m
  DO I=1,NSCCTC 
     OMEGAGS(I)=ZERO

     JQ =IQMLST(I,1)
     XDIFF=X(JQ)-RRXCEN
     YDIFF=Y(JQ)-RRYCEN
     ZDIFF=Z(JQ)-RRZCEN
     CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
     CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
     CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)

     !        Multiply M_mn*b_n(Rb), and sum over n
     DO II=1,NTPOL
        XPOL=LSTPX(II)
        YPOL=LSTPY(II)
        ZPOL=LSTPZ(II)
        NORM=BNORM(II)
        OMEGAGS(I) = OMEGAGS(I)   + NORM* &
             ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)*MQ(II)
     ENDDO ! sum over m
  ENDDO ! loop over QM atoms


  !      ..........................................................
  !      ............ Outer potential contribution to shift........
  !      ..........................................................

  !      II.3.-->>> Add phi^(o)(Ra) (pretty much the same way as in 
  !      rforce, though use point charge as probe.

  ncyz=ncly*nclz
  IF(QBSPL) THEN
     DO I=1,NSCCTC
        JQ=IQMLST(I,1)
        xi=x(jq)+tranx-xbcen
        yi=y(jq)+trany-ybcen
        zi=z(jq)+tranz-zbcen

        nfil=1
        ix=int(xi/dcel)+1
        iy=int(yi/dcel)+1
        iz=int(zi/dcel)+1
        jx1=ix-nfil
        if(jx1.lt.1)jx1=1
        jx2=ix+nfil+1
        if(jx2.gt.nclx)jx2=nclx
        jy1=iy-nfil
        if(jy1.lt.1)jy1=1
        jy2=iy+nfil+1
        if(jy2.gt.ncly)jy2=ncly
        jz1=iz-nfil
        if(jz1.lt.1)jz1=1
        jz2=iz+nfil+1
        if(jz2.gt.nclz)jz2=nclz
        DO K=jx1,jx2
           IPX=(K-1)*NCyz
           XC=(K-1)*DCEL
           !     
           DO L=jy1,jy2
              IPY=(L-1)*NCLz
              YC=(L-1)*DCEL
              !     
              DO M=jz1,jz2
                 IPZ=M+IPY+IPX
                 ZC=(M-1)*DCEL

                 ai=1.5-(xi-xc)/dcel
                 bi=1.5-(yi-yc)/dcel
                 ci=1.5-(zi-zc)/dcel
                 fi=M3(ai)*M3(bi)*M3(ci)
                 !     Electrostatic "shift" 
                 !     NOTE: no CCELEC to be consistent with SCC
                 OMEGAGS(I)=OMEGAGS(I) +  &
                      FACTOR*fi*phi1(ipz)

              ENDDO         ! M
           ENDDO            ! L
        ENDDO               ! K
     ENDDO
  ELSE
     DO I=1,NSCCTC
        !      ...... assign the probe charge to a lattice and then sum over
        !      ...... the phi at those lattice points
        JQ=IQMLST(I,1)
        xi=x(jq)+tranx-xbcen
        yi=y(jq)+trany-ybcen
        zi=z(jq)+tranz-zbcen
        ix=int(xi/dcel)+1
        iy=int(yi/dcel)+1
        iz=int(zi/dcel)+1
        !     
        !     Atom charge distribution by 8 adjacent grid points
        !     
        do n1=1,2
           in1=ix+n1-1
           ai=xi-(in1-1)*dcel
           aisign=sign(one,ai)
           ai=1.-abs(ai)/dcel
           in1=(in1-1)*ncyz

           do n2=1,2
              in2=iy+n2-1
              bi=yi-(in2-1)*dcel
              bisign=sign(one,bi)
              bi=1. - abs(bi)/dcel
              in2=(in2-1)*nclz

              do n3=1,2
                 in3=iz+n3-1
                 ci=zi-(in3-1)*dcel
                 cisign=sign(one,ci)
                 ci=1. - abs(ci)/dcel
                 fi=ai*bi*ci
                 in3=in1+in2+in3
                 !     
                 !     Electrostatic "Energy" 
                 !     NOTE: no CCELEC to be consistent with SCC
                 OMEGAGS(I)=OMEGAGS(I) +  &
                      FACTOR*fi*phi1(in3)
                 !     
                 !                    enet1=FACTOR*fi*chi*phi1(in3)*CCELEC
                 !                    enelp1=enelp1+enet1

              enddo !n3
           enddo ! n2
        enddo !n1

     ENDDO                  ! loop over QM atoms
  ENDIF
  RETURN
END SUBROUTINE SCCRECT0

SUBROUTINE SCCRECT4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ,NTPOL,MAXNPOL, &
     RRXCEN,RRYCEN,RRZCEN,XSCALE,YSCALE,ZSCALE, &
     CGE, & ! XIAO_QC_UW0609: change to avoid conflict with psf.fcm
     MIJ,COEF,COEFX,MQ,COEF_QM, &
     LSTPX,LSTPY,LSTPZ,BNORM,ALPX,ALPY,ALPZ, &
     ADLPX,ADLPY,ADLPZ,XNPOL,YNPOL,ZNPOL,LSTPOL, &
     NCLX,NCLY,NCLZ,DCEL, &
     PHI1,TRANX,TRANY,TRANZ, &
     XBCEN,YBCEN,ZBCEN,FACTOR, &
     RXNAFX,RXNAFY,RXNAFZ, &
     RXNBFX,RXNBFY,RXNBFZ, QBSPL)
  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use consta
  use gamess_fcm
  use sccdftb
  use sccgsbp
  use blockscc_fcm
  ! XIAO_PHK_QC_UW0609: DIV
  use psf
  use chutil,only: getres
  use pbeq,only:m3,dm3,lpol2,dlpol2

  implicit none

  INTEGER NTRB,LSTRB(*)
  INTEGER LSTPX(*),LSTPY(*),LSTPZ(*),LSTPOL(*)
  real(chm_real)  ALPX(*),ALPY(*),ALPZ(*),ADLPX(*),ADLPY(*),ADLPZ(*)
  INTEGER NTPOL,MAXNPOL
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real)  DX(*),DY(*),DZ(*)
  real(chm_real)  COEF(*),COEFX(*),MIJ(*),BNORM(*),MQ(*),COEF_QM(*)
  real(chm_real)  RRXCEN,RRYCEN,RRZCEN
  real(chm_real)  XSCALE,YSCALE,ZSCALE
  INTEGER XNPOL,YNPOL,ZNPOL

  integer NCLX,NCLY,NCLZ
  real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
  real(chm_real4)  phi1(*)
  real(chm_real)  FACTOR,ENSOLV
  real(chm_real)  CGE(*)        
  real(chm_real)  RXNAFX(*),RXNAFY(*),RXNAFZ(*)
  real(chm_real)  RXNBFX(*),RXNBFY(*),RXNBFZ(*)
  logical QBSPL

  ! local
  INTEGER I,J,II,JJ,IJ,L,M,LL,MM,LMAXSCC,JQ
  INTEGER XPOL,YPOL,ZPOL
  real(chm_real)  LPOL,DLPOL,XG,YG,ZG,NORM
  real(chm_real)  LPOLX,LPOLY,LPOLZ,CMIJ,CCC
  INTEGER XPI,YPI,ZPI
  real(chm_real)  XDIFF,YDIFF,ZDIFF
  real(chm_real)  DX1,DY1,DZ1
  real(chm_real)  TMPFX,TMPFY,TMPFZ
  !
  integer ncyz,il,ix,iy,iz,n1,in1,n2,in2,n3,in3
  real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
  real(chm_real)  enet1, enelp1, enet2, enelp2    
  real(chm_real)  aisign,bisign,cisign,prefac
  ! B-spline
  integer jx1,jx2,jy1,jy2,jz1,jz2,k,ipx,ipy,ipz,nfil
  real(chm_real)  xc,yc,zc! ,M3,DM3 
  ! XIAO_PHK_QC_UW0609: DIV
  integer K1,K2, IQ, IS
  REAL(chm_real) NE, DQ

  !      QC: Evaluate and collect QM-GSBP components for energy and
  !      forces.
  !      The mulliken charges of the QM atoms have been saved in /sccmul/
  !      QMULI2(*,1).

  !      FOR FEP - scale the forces otherwise not.
  if(qlamda.eqv..false.) scal=1.0d0

  !      -------------- First, energy contributions ---------------
  !      The GSBP contribution to QM energy has been included already
  !      in scctdftbsrc/eglcao.f, nothing needed here 

  !      -------------- Second,force  contributions to QM and MM atoms ---
  !      Part of the GSBP contribution to the QM derivatives (those 
  !      depend on overlap derivatives, i.e., change in the Mulliken 
  !      charges) has been included
  !      in scctdftbsrc/eglcao.f, and has been added to (DX,DY,DZ)
  !      in scctbini.src/scctbene
  !      >>>> What's left correspond to the contributions involveing FIXED
  !      QM mulliken charges, which can be handled straightforwardly in
  !      EXACTLY the same manner as MM contributions! <<<<
  !   

  !      ..........................................................
  !      ............ Outer potential contributions ---> only to QM
  !      ..........................................................
  ncyz=ncly*nclz 

  IF(QBSPL) THEN
     DO I=1,NSCCTC
        chi=QMULI2(I,1)
        rxnafx(i)=zero
        rxnafy(i)=zero
        rxnafz(i)=zero  
        IF(CHI.EQ.0.0) GOTO 492
        JQ=IQMLST (I,1)

        xi=x(i)+tranx-xbcen
        yi=y(i)+trany-ybcen
        zi=z(i)+tranz-zbcen
        nfil=1
        ix=int(xi/dcel)+1
        iy=int(yi/dcel)+1
        iz=int(zi/dcel)+1
        jx1=ix-nfil
        if(jx1.lt.1)jx1=1
        jx2=ix+nfil+1
        if(jx2.gt.nclx)jx2=nclx
        jy1=iy-nfil
        if(jy1.lt.1)jy1=1
        jy2=iy+nfil+1
        if(jy2.gt.ncly)jy2=ncly
        jz1=iz-nfil
        if(jz1.lt.1)jz1=1
        jz2=iz+nfil+1
        if(jz2.gt.nclz)jz2=nclz

        DO K=jx1,jx2
           IPX=(K-1)*NCyz
           XC=(K-1)*DCEL
           !
           DO L=jy1,jy2
              IPY=(L-1)*NCLz
              YC=(L-1)*DCEL
              !
              DO M=jz1,jz2
                 IPZ=M+IPY+IPX
                 ZC=(M-1)*DCEL

                 ai=1.5-(xi-xc)/dcel
                 bi=1.5-(yi-yc)/dcel
                 ci=1.5-(zi-zc)/dcel
                 fi=M3(ai)*M3(bi)*M3(ci)
                 !
                 !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                 !
                 prefac=phi1(ipz)*CCELEC*chi/dcel
                 RXNAFx(i)=RXNAFx(i) + DM3(ai)*M3(bi)*M3(ci)*prefac
                 RXNAFy(i)=RXNAFy(i) + M3(ai)*DM3(bi)*M3(ci)*prefac
                 RXNAFz(i)=RXNAFz(i) + M3(ai)*M3(bi)*DM3(ci)*prefac
                 !
                 !     Electrostatic Energy
                 !
                 enelp1=enelp1+FACTOR*fi*chi*phi1(ipz)*CCELEC

              enddo
           enddo
        enddo


492     continue
     ENDDO
  ELSE
     !
     DO I=1,NSCCTC
        !      ...... assign the probe charge to a lattice and then sum over
        !      ...... the phi at those lattice points
        chi=QMULI2(I,1)
        rxnafx(i)=zero
        rxnafy(i)=zero
        rxnafz(i)=zero
        JQ=IQMLST (I,1)
        xi=x(jq)+tranx-xbcen
        yi=y(jq)+trany-ybcen
        zi=z(jq)+tranz-zbcen
        ix=int(xi/dcel)+1
        iy=int(yi/dcel)+1
        iz=int(zi/dcel)+1
        !     
        !     Atom charge distribution by 8 adjacent grid points
        !     
        do n1=1,2
           in1=ix+n1-1
           ai=xi-(in1-1)*dcel
           aisign=sign(one,ai)
           ai=1.-abs(ai)/dcel
           in1=(in1-1)*ncyz

           do n2=1,2
              in2=iy+n2-1
              bi=yi-(in2-1)*dcel
              bisign=sign(one,bi)
              bi=1. - abs(bi)/dcel
              in2=(in2-1)*nclz

              do n3=1,2
                 in3=iz+n3-1
                 ci=zi-(in3-1)*dcel
                 cisign=sign(one,ci)
                 ci=1. - abs(ci)/dcel
                 fi=ai*bi*ci
                 in3=in1+in2+in3
                 !     
                 prefac=phi1(in3)*CCELEC*chi/dcel
                 !
                 if((ai.lt.(one-rsmall)).and.(ai.gt.rsmall)) then
                    RXNAFx(i)=RXNAFx(i)+aisign*bi*ci*prefac
                 endif
                 !
                 if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                    RXNAFy(i)=RXNAFy(i)+bisign*ai*ci*prefac
                 endif
                 !
                 if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                    RXNAFz(i)=RXNAFz(i)+cisign*ai*bi*prefac
                 endif

              enddo !n3
           enddo ! n2
        enddo !n1

     ENDDO                  ! loop over QM atoms
  ENDIF
  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
     do i=1,nscctc
        write(outu,'(3x,i5,4x,3f10.5)')  &
             i,rxnafx(i),rxnafy(i),rxnafz(i)
     enddo
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO I=1,NSCCTC
     DX(IQMLST(I,1))=DX(IQMLST(I,1)) - RXNAFx(i)*scal
     DY(IQMLST(I,1))=DY(IQMLST(I,1)) - RXNAFy(i)*scal
     DZ(IQMLST(I,1))=DZ(IQMLST(I,1)) - RXNAFz(i)*scal
  ENDDO

  !      ............ QMQ type of Rxn field terms, both QM and MM atoms
  !      ...... Build Q for QM 
  DO II=1,NTPOL
     COEF_QM(II)=ZERO
  ENDDO
  DO I=1,NSCCTC
     J =IQMLST(I,1)
     CCC=QMULI2(I,1)
     IF(CCC.EQ.ZERO) CYCLE
     XDIFF=X(J)-RRXCEN
     YDIFF=Y(J)-RRYCEN
     ZDIFF=Z(J)-RRZCEN

     CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
     CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
     CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)

     DO II=1,NTPOL
        XPOL=LSTPX(II)
        YPOL=LSTPY(II)
        ZPOL=LSTPZ(II)
        NORM=BNORM(II)
        COEF_QM(II)=COEF_QM(II)+CCC*NORM* &
             ALPX(XPOL+1)*ALPY(YPOL+1)*ALPZ(ZPOL+1)

     ENDDO
  ENDDO

  !      >>>>>> First treat QM derivatives
  !      ...... Build the M*Q by augumenting with QM contributions ......

  DO I=1,MAXNPOL
     II=LSTPOL(I)
     MQ(II)=ZERO
     DO J=1,MAXNPOL
        JJ=LSTPOL(J)
        IJ=(II-1)*NTPOL+JJ
        IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
        MQ(II)=MQ(II)+MIJ(IJ)*(COEF(JJ) - COEFX(JJ) + COEF_QM(JJ))
     ENDDO
  ENDDO

  DO LL=1,NSCCTC
     !        MM=IQMLST(LL,1)
     RXNBFX(LL)=ZERO
     RXNBFY(LL)=ZERO
     RXNBFZ(LL)=ZERO
  ENDDO
  !      ...... Now proceed for QM atoms 
  ! reaction field force calculations     
  DO LL=1,NSCCTC
     MM=IQMLST(LL,1)
     XDIFF=X(MM)-RRXCEN
     YDIFF=Y(MM)-RRYCEN
     ZDIFF=Z(MM)-RRZCEN
     TMPFX=ZERO
     TMPFY=ZERO
     TMPFZ=ZERO
     CCC=QMULI2(LL,1)*CCELEC
     IF(CCC.EQ.ZERO) CYCLE
     CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
     CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
     CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)
     CALL DLPOL2(XNPOL,XDIFF,XSCALE,ALPX,ADLPX)
     CALL DLPOL2(YNPOL,YDIFF,YSCALE,ALPY,ADLPY)
     CALL DLPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ,ADLPZ)

     DO I=1,MAXNPOL
        II=LSTPOL(I)
        XPI=LSTPX(II)
        YPI=LSTPY(II)
        ZPI=LSTPZ(II)
        NORM=BNORM(II)
        LPOLX=ALPX(XPI+1)
        LPOLY=ALPY(YPI+1)
        LPOLZ=ALPZ(ZPI+1)
        DX1=NORM*ADLPX(XPI+1)*LPOLY*LPOLZ
        DY1=NORM*LPOLX*ADLPY(YPI+1)*LPOLZ
        DZ1=NORM*LPOLX*LPOLY*ADLPZ(ZPI+1)
        TMPFX=TMPFX-DX1*MQ(II)
        TMPFY=TMPFY-DY1*MQ(II)
        TMPFZ=TMPFZ-DZ1*MQ(II)
     ENDDO
     RXNBFX(LL)=TMPFX*CCC
     RXNBFY(LL)=TMPFY*CCC
     RXNBFZ(LL)=TMPFZ*CCC
  ENDDO

  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
     do i=1,nscctc
        write(outu,'(3x,i5,4x,3f10.5)')  &
             i,rxnbfx(i),rxnbfy(i),rxnbfz(i)
     enddo
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO I=1,NSCCTC
     DX(IQMLST(I,1))=DX(IQMLST(I,1)) - RXNBFx(i)*scal
     DY(IQMLST(I,1))=DY(IQMLST(I,1)) - RXNBFy(i)*scal
     DZ(IQMLST(I,1))=DZ(IQMLST(I,1)) - RXNBFz(i)*scal
  ENDDO

  !      >>>>>> Next treat MM derivatives 
  !      ...... Build the M*Q by including only QM contributions ...... 
  !
  DO I=1,MAXNPOL
     II=LSTPOL(I)
     MQ(II)=ZERO
     DO J=1,MAXNPOL
        JJ=LSTPOL(J)
        IJ=(II-1)*NTPOL+JJ
        IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
        MQ(II)=MQ(II)+MIJ(IJ)*COEF_QM(JJ)
     ENDDO
  ENDDO

  !      ...... Now proceed for MM atoms  
  ! reaction field force calculations     
  ! PS/QC: Remember to EXCLUDE "excluded atoms" from QM/MM calculations
  DO LL=1,NTRB
     MM=LSTRB(LL)
     RXNBFX(MM)=ZERO
     RXNBFY(MM)=ZERO
     RXNBFZ(MM)=ZERO
  ENDDO
  DO LL=1,NTRB
     MM=LSTRB(LL)
     XDIFF=X(MM)-RRXCEN
     YDIFF=Y(MM)-RRYCEN
     ZDIFF=Z(MM)-RRZCEN
     TMPFX=ZERO
     TMPFY=ZERO
     TMPFZ=ZERO
     CCC=CGE(MM)*CCELEC
     !        IF(CCC.EQ.ZERO) GOTO 190
     IF((CCC.EQ.ZERO).OR.IGMSEL(MM).EQ.5) CYCLE
     ! XIAO_PHK_QC_UW0609: DIV
     IF (IGMSEL(MM).EQ.5) CCC=ZERO
     IF (QSCCDIV) THEN
        IF(IGMSEL(MM).EQ.0) THEN
           ! loop over all atoms in group
           ! and find a div atom
           K1=GETRES(MM,IGPBS,NGRP)
           IS=IGPBS(K1)+1
           IQ=IGPBS(K1+1)
           DQ=0
           NE=0
           DO K2=IS,IQ
              IF (IGMSEL(K2).EQ.5) THEN
                 DQ=DQ+CGE(K2)
                 NE=NE+1
              ENDIF
           ENDDO
           IF (NE.gt.0) THEN
              CCC=CCC+(DQ/(IQ-IS+1-NE))*CCELEC
           ENDIF
        ENDIF
     ENDIF
     ! XIAO_PHK_QC_UW0609: END DIV
     CALL LPOL2(XNPOL,XDIFF,XSCALE,ALPX)
     CALL LPOL2(YNPOL,YDIFF,YSCALE,ALPY)
     CALL LPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ)
     CALL DLPOL2(XNPOL,XDIFF,XSCALE,ALPX,ADLPX)
     CALL DLPOL2(YNPOL,YDIFF,YSCALE,ALPY,ADLPY)
     CALL DLPOL2(ZNPOL,ZDIFF,ZSCALE,ALPZ,ADLPZ)

     DO I=1,MAXNPOL
        II=LSTPOL(I)
        XPI=LSTPX(II)
        YPI=LSTPY(II)
        ZPI=LSTPZ(II)
        NORM=BNORM(II)
        LPOLX=ALPX(XPI+1)
        LPOLY=ALPY(YPI+1)
        LPOLZ=ALPZ(ZPI+1)
        DX1=NORM*ADLPX(XPI+1)*LPOLY*LPOLZ
        DY1=NORM*LPOLX*ADLPY(YPI+1)*LPOLZ
        DZ1=NORM*LPOLX*LPOLY*ADLPZ(ZPI+1)
        TMPFX=TMPFX-DX1*MQ(II)
        TMPFY=TMPFY-DY1*MQ(II)
        TMPFZ=TMPFZ-DZ1*MQ(II)
     ENDDO
     RXNBFX(MM)=TMPFX*CCC
     RXNBFY(MM)=TMPFY*CCC
     RXNBFZ(MM)=TMPFZ*CCC
  ENDDO

  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
     do il=1,ntrb
        i=lstrb(il)
        write(outu,'(3x,i5,4x,3f10.5)')  &
             i,rxnbfx(i),rxnbfy(i),rxnbfz(i)
     enddo
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO IL=1,NTRB
     I=LSTRB(IL)
     DX(I)=DX(I) - RXNBFx(i)*scal
     DY(I)=DY(I) - RXNBFy(i)*scal
     DZ(I)=DZ(I) - RXNBFz(i)*scal
  ENDDO
  return
end SUBROUTINE SCCRECT4
#endif /*  SCCDFTB*/
#endif /*  GSBP*/
#endif /*  PBEQ*/

