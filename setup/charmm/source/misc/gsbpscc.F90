SUBROUTINE NULL_SCCGSBP
  RETURN
END SUBROUTINE NULL_SCCGSBP

#if KEY_PBEQ==1
#if KEY_GSBP==1
#if KEY_SCCDFTB==1
!
!      QC: Subroutines needed for computing SCC-DFTB in the presence
!      of the gsbp contribution.
!      The "difficulty" is that the GSBP contribution has to be included
!      in the SCF procedure. With SCC-DFTB, this is very straightforward
!      since the QM/GSBP interactions can be described with the Mulliken
!      charges on the QM atoms, as in SCC-DFTB/MM interactions.
!      One comment is that since GSBP is based on Poission-Boltzman,
!      which counts ALL electrostatic interactions, one ideally should
!      use NO cut-off for the electrostatic interactions for the direct
!      (Coloumbic) QM-MM (or even MM-MM) interactions. In practice,
!      however, this depends on the problem. Need to test carefully.
!      Nov. 20. 2003
!      NOT YET: (1). Basis sorting for QM-GSBP
!               (2). B-spline for Poisson-Boltzmann phi

SUBROUTINE SCCGSBP0(X,Y,Z,NTPOL,MAXNPOL,SRDIST, &
     RRXCEN,RRYCEN,RRZCEN, &
     MIJ,COEF,COEFX,MQ,COEF_B, &
     LSTPL,LSTPM,BNORM, &
     AR,AC,AS,AP,ADP,NMPOL,LSTPOL, &
     NCLX,NCLY,NCLZ,DCEL, &
     PHI1,TRANX,TRANY,TRANZ, &
     XBCEN,YBCEN,ZBCEN,FACTOR,QBSPL)
  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use consta
  use sccdftb
  use sccgsbp
  use pbeq,only:m3,dm3,rpowerl2,cosmphi2,sinmphi2,alpol2,dalpol2

  implicit none

  INTEGER LSTPL(*),LSTPM(*),LSTPOL(*),NTPOL,MAXNPOL,NMPOL
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real)  COEF(*),COEFX(*),MIJ(*),BNORM(*),MQ(*)
  real(chm_real)  RRXCEN,RRYCEN,RRZCEN
  real(chm_real)  AR(0:NMPOL-1),AC(0:NMPOL-1),AS(0:NMPOL-1)
  real(chm_real)  AP(0:NMPOL-1,0:NMPOL-1),ADP(0:NMPOL-1,0:NMPOL-1)
  integer NCLX,NCLY,NCLZ
  real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
  real(chm_real4)  phi1(*)
  real(chm_real)  FACTOR
  !     SCC-RELATED @@ CHECK NTPOL vs. MAXNPOL
  real(chm_real)  COEF_B(NTPOL,*)
  real(chm_real)  SRDIST
  real(chm_real)  SRDIST2
  ! local
  INTEGER I,J,II,JJ,IJ,L,M,LL,MM,LMAXSCC,JQ
  LOGICAL QBSPL
  INTEGER KK,K,KQ
  real(chm_real)  NORM
  real(chm_real)  SP,CP,ST,CT,R,R2,XDIFF,YDIFF,ZDIFF
  !
  integer ncyz,il,ix,iy,iz,n1,in1,n2,in2,n3,in3
  real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
  real(chm_real)  enet1, enelp1, enet2, enelp2    
  real(chm_real)  aisign,bisign,cisign,prefac

  ! B-spline (UW_05)
  integer jx1,jx2,jy1,jy2,jz1,jz2,ipx,ipy,ipz,nfil
  real(chm_real)  xc,yc,zc!,M3,DM3 

  !      Evaluate arrays required in SCF iterations.
  !      The GAMA and OMEGA arrays. These will be save in the /sccgs/
  !      common block - which is O.K. for now due to the typically 
  !      small QM size. Improve later with heap.

  !      *** The current code apparently assumes a spherical basis
  !      which are reflected in computing Gamma and Omega. For other
  !      geometries, however, things can be done in a highly parallel
  !      fashion. ***


  !      -------- Stage I, build Gamma(Ra,Rb) --------
  !      where Gamma(Ra,Rb) = Sum(m,n) {b_m(Ra)*M_mn*b_n(Rb)}
  !      Gamma contributes to both SCF (Hamiltonian) and Energy, as well
  !      as to the derivatives of the QM atoms 

  LMAXSCC=NMPOL-1
  SRDIST2=(SRDIST+RSMALL)*(SRDIST+RSMALL)

  !     write(*,*) "Calculate M_mn*b_n(Rb)",NSCCRP,NSCCTC,NTPOL
  !     I.1.-->>>Calculate M_mn*b_n(Rb) (STORED as COEF_B in /sccgsbp/)
  !     UW_0605:Haibo Yu Replica of SCC
  DO KK=1,NSCCRP
     DO JJ=1,NSCCTC
        DO II=1,NTPOL
           COEF_B(II,JJ+(KK-1)*NSCCTC)=ZERO
        ENDDO
     ENDDO
  ENDDO
  !     write(*,*) "Array COEF_B cleared "

  !     UW_0605:Haibo Yu Replica of SCC
  DO K=1,NSCCRP
     DO I=1,NSCCTC
        !        assume no replica for now
        J =IQMLST(I,K)
        XDIFF=X(J)-RRXCEN
        YDIFF=Y(J)-RRYCEN
        ZDIFF=Z(J)-RRZCEN
        R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
        IF(R2.GT.SRDIST2) CYCLE
        R=SQRT(R2)
        CT=ZDIFF/R
        ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
        CP=XDIFF/R/ST
        SP=YDIFF/R/ST
        IF(R2.LT.RSMALL) THEN                               ! in the origin
           CT=ZERO
           CP=ZERO
           SP=ZERO
        ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
             YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
           CT=ONE
           IF(ZDIFF.LT.ZERO) CT=-ONE
           CP=ZERO
           SP=ZERO
        ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
           CT=ZERO
           CP=XDIFF/R
           SP=YDIFF/R
        ENDIF

        !        pull out basis vectors at this position, Rb
        !        write(*,*) "Pulling out basis...",LMAXSCC
        CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
        CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
        CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
        CALL ALPOL2(LMAXSCC,CT,AP)       !  fill AP  (P(lm) ) array

        !        write(*,*) "Multiply and sum ......"
        !        Multiply M_mn*b_n(Rb), and sum over n
        DO JJ=1,NTPOL
           DO II=1,NTPOL
              IJ=(JJ-1)*NTPOL + II
              IF (JJ.GT.II) IJ=(II-1)*NTPOL + JJ
              L=LSTPL(II)
              M=LSTPM(II)
              NORM=BNORM(II)
              IF(L.GE.0.AND.M.EQ.0) THEN
                 !                write(*,*) "Index1",II,JJ,L,M
                 !                write(*,*) "Index1.1",MIJ(IJ)
                 !                write(*,*) "Index1.2",AR(L)
                 !                write(*,*) "Index1.3",AP(L,M)
                 COEF_B(JJ,(K-1)*NSCCTC+I)=COEF_B(JJ,(K-1)*NSCCTC+I) &
                      +MIJ(IJ)*NORM*AR(L)*AP(L,M)
              ELSEIF(L.GT.0.AND.M.GT.0) THEN
                 !                write(*,*) "Index2",II,JJ,L,M
                 COEF_B(JJ,(K-1)*NSCCTC+I)=COEF_B(JJ,(K-1)*NSCCTC+I) &
                      +MIJ(IJ)*NORM*AR(L)*AC(M)*AP(L,M)
              ELSEIF(L.GT.0.AND.M.LT.0) THEN
                 !                write(*,*) "Index3",II,JJ,L,M
                 M=-M
                 COEF_B(JJ,(K-1)*NSCCTC+I)=COEF_B(JJ,(K-1)*NSCCTC+I) &
                      +MIJ(IJ)*NORM*AR(L)*AS(M)*AP(L,M)
              ENDIF
           ENDDO ! sum over n
        ENDDO ! loop over m
     ENDDO ! loop over Rb
  ENDDO ! loop over replica

  !     write(*,*) "Calculate b_m(Ra)*COEF_B(m,Rb)"
  !     I.2.-->>>Calculate b_m(Ra)*COEF_B(m,Rb) sum over m
  !     save the lower triangle 

  IJ = 0 
  DO K=1,NSCCRP
     DO I=1,NSCCTC
        DO J=1,I
           IJ=IJ+1
           GAMAGS(IJ)=ZERO

           JQ =IQMLST(I,K)
           XDIFF=X(JQ)-RRXCEN
           YDIFF=Y(JQ)-RRYCEN
           ZDIFF=Z(JQ)-RRZCEN
           R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
           IF(R2.GT.SRDIST2) CYCLE
           R=SQRT(R2)
           CT=ZDIFF/R
           ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
           CP=XDIFF/R/ST
           SP=YDIFF/R/ST
           IF(R2.LT.RSMALL) THEN                               ! in the origin
              CT=ZERO
              CP=ZERO
              SP=ZERO
           ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
                YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
              CT=ONE
              IF(ZDIFF.LT.ZERO) CT=-ONE
              CP=ZERO
              SP=ZERO
           ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
              CT=ZERO
              CP=XDIFF/R
              SP=YDIFF/R
           ENDIF

           !        pull out basis vectors at this position, Rb
           CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
           CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
           CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
           CALL ALPOL2(LMAXSCC,CT,AP)       !  fill AP  (P(lm) ) array

           !        Multiply b_m(Ra)*COEF_B(m,Rb), and sum over m
           DO II=1,NTPOL
              L=LSTPL(II)
              M=LSTPM(II)
              NORM=BNORM(II)
              IF(L.GE.0.AND.M.EQ.0) THEN
                 GAMAGS(IJ) = GAMAGS(IJ)   &
                      +NORM*AR(L)*AP(L,M)*COEF_B(II,(K-1)*NSCCTC+J)
              ELSEIF(L.GT.0.AND.M.GT.0) THEN
                 GAMAGS(IJ) = GAMAGS(IJ)   &
                      +NORM*AR(L)*AC(M)*AP(L,M)*COEF_B(II,(K-1)*NSCCTC+J)
              ELSEIF(L.GT.0.AND.M.LT.0) THEN
                 M=-M
                 GAMAGS(IJ) = GAMAGS(IJ)   &
                      +NORM*AR(L)*AS(M)*AP(L,M)*COEF_B(II,(K-1)*NSCCTC+J)
              ENDIF
           ENDDO ! sum over m

        ENDDO ! loop over Rb
     ENDDO ! loop over Ra
  ENDDO ! loop over replica

  !      write(*,*) "Stage II, build Omega(Ra)"
  !      ------- Stage II, build Omega(Ra) --------
  !      where Omega(Ra)=Phi^(o)(Ra) + Sum(m,n){b_m(Ra)*M_mn*Q^MM_n}
  !      Omega contributes to both SCF (Hamiltonian) and Energy, as well
  !      as to the derivatives of the QM atoms

  !      write(*,*) "II.1.-->>> Compute MQ_m" 

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

  !      write(*,*) "II.2.-->>> Compute b_m(Ra)*MQ_m"
  !      II.2.-->>> Compute b_m(Ra)*MQ_m sum over m
  DO K=1,NSCCRP
     DO I=1,NSCCTC 
        OMEGAGS((K-1)*NSCCTC+I)=ZERO

        JQ =IQMLST(I,K)
        XDIFF=X(JQ)-RRXCEN
        YDIFF=Y(JQ)-RRYCEN
        ZDIFF=Z(JQ)-RRZCEN
        R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
        IF(R2.GT.SRDIST2) CYCLE
        R=SQRT(R2)
        CT=ZDIFF/R
        ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
        CP=XDIFF/R/ST
        SP=YDIFF/R/ST
        IF(R2.LT.RSMALL) THEN                               ! in the origin
           CT=ZERO
           CP=ZERO
           SP=ZERO
        ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
             YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
           CT=ONE
           IF(ZDIFF.LT.ZERO) CT=-ONE
           CP=ZERO
           SP=ZERO
        ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
           CT=ZERO
           CP=XDIFF/R
           SP=YDIFF/R
        ENDIF

        !        pull out basis vectors at this position, Rb
        CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
        CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
        CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
        CALL ALPOL2(LMAXSCC,CT,AP)       !  fill AP  (P(lm) ) array

        !        Multiply M_mn*b_n(Rb), and sum over n
        DO II=1,NTPOL
           L=LSTPL(II)
           M=LSTPM(II)
           NORM=BNORM(II)
           IF(L.GE.0.AND.M.EQ.0) THEN
              OMEGAGS((K-1)*NSCCTC+I) = OMEGAGS((K-1)*NSCCTC+I)    &
                   +NORM*AR(L)*AP(L,M)*MQ(II)
           ELSEIF(L.GT.0.AND.M.GT.0) THEN
              OMEGAGS((K-1)*NSCCTC+I) = OMEGAGS((K-1)*NSCCTC+I)    &
                   +NORM*AR(L)*AC(M)*AP(L,M)*MQ(II)
           ELSEIF(L.GT.0.AND.M.LT.0) THEN
              M=-M
              OMEGAGS((K-1)*NSCCTC+I) = OMEGAGS((K-1)*NSCCTC+I)    &
                   +NORM*AR(L)*AS(M)*AP(L,M)*MQ(II)
           ENDIF
        ENDDO ! sum over m
     ENDDO ! loop over QM atoms
  ENDDO ! loop over replica
  !      write(*,*) "II.3.-->>> Add phi^(o)(Ra)"
  !      II.3.-->>> Add phi^(o)(Ra) (pretty much the same way as in 
  !      rforce, though use point charge as probe.

  ncyz=ncly*nclz
  IF(QBSPL) THEN
     DO KQ=1,NSCCRP
        DO I=1,NSCCTC
           JQ=IQMLST(I,KQ)
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
                    OMEGAGS((KQ-1)*NSCCTC+I) = &
                         OMEGAGS((KQ-1)*NSCCTC+I) +  &
                         FACTOR*fi*phi1(ipz)

                 ENDDO         ! M
              ENDDO            ! L
           ENDDO               ! K
        ENDDO                  ! I
     ENDDO                  ! KQ
  ELSE

     DO K=1,NSCCRP
        DO I=1,NSCCTC
           !      ...... assign the probe charge to a lattice and then sum over
           !      ...... the phi at those lattice points
           JQ=IQMLST(I,K)
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
                    OMEGAGS((K-1)*NSCCTC+I)=OMEGAGS((K-1)*NSCCTC+I) +  &
                         FACTOR*fi*phi1(in3)
                    !     
                    !                    enet1=FACTOR*fi*chi*phi1(in3)*CCELEC
                    !                    enelp1=enelp1+enet1

                 enddo !n3
              enddo ! n2
           enddo !n1

        ENDDO ! loop over QM atoms
     ENDDO ! loop over replica
  ENDIF

  !      Since we use GAMAGS, OMEGAGS later in SCCGSBP4 for grad
  !      calculations, the unit is in kcal/mol, which is converted
  !      to atomic units in sccdftbsrc/shift.f

  !      Certain elements also need to be evaluated for QM force 
  !      calculations, it basically contains the derivatives of the 
  !      basis vectors. The point is that nothing needs to be iterated
  !      and therefore we can perhaps add certain components in SCCGSBP4,
  !      while the convenient parts (which does NOT include basis deri.)
  !      can be done in sccdftb source. 
  !      The MM force calculations are more straightforward
  !      and can be organized in SCCGSBP4. 

  RETURN
END SUBROUTINE SCCGSBP0

SUBROUTINE SCCGSBP4(NTRB,LSTRB,X,Y,Z,DX,DY,DZ,NTPOL,MAXNPOL, &
     SRDIST, &
     RRXCEN,RRYCEN,RRZCEN, &
     CGE, & ! XIAO_PHK_QC_UW0609: change to avoid conflict with psf.fcm
     MIJ,COEF,COEFX,MQ,COEF_QM, &
     LSTPL,LSTPM,BNORM, &
     AR,AC,AS,AP,ADP,NMPOL,LSTPOL, &
     NCLX,NCLY,NCLZ,DCEL, &
     PHI1,TRANX,TRANY,TRANZ, &
     XBCEN,YBCEN,ZBCEN,FACTOR, &
     RXNAFX,RXNAFY,RXNAFZ, &
     RXNBFX,RXNBFY,RXNBFZ,QBSPL)
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
  use pbeq,only:m3,dm3,rpowerl2,cosmphi2,sinmphi2,alpol2,dalpol2

  implicit none

  INTEGER NTRB,LSTRB(*),LSTPL(*),LSTPM(*),LSTPOL(*)
  INTEGER NTPOL,MAXNPOL,NMPOL
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real)  DX(*),DY(*),DZ(*)
  real(chm_real)  COEF(*),COEFX(*),MIJ(*),BNORM(*),MQ(*),COEF_QM(*)
  real(chm_real)  RRXCEN,RRYCEN,RRZCEN
  real(chm_real)  AR(0:NMPOL-1),AC(0:NMPOL-1),AS(0:NMPOL-1)
  real(chm_real)  AP(0:NMPOL-1,0:NMPOL-1),ADP(0:NMPOL-1,0:NMPOL-1)
  integer NCLX,NCLY,NCLZ
  real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
  real(chm_real4)  phi1(*)
  real(chm_real)  FACTOR,ENSOLV
  real(chm_real)  CGE(*)        
  real(chm_real)  RXNAFX(*),RXNAFY(*),RXNAFZ(*)
  real(chm_real)  RXNBFX(*),RXNBFY(*),RXNBFZ(*)

  !     @@ Make sure that RXN?FX are not used anymore
  real(chm_real)  SRDIST
  real(chm_real)  SRDIST2

  LOGICAL QBSPL

  ! local
  INTEGER I,J,II,JJ,IJ,L,M,LL,MM,LMAXSCC,JQ
  INTEGER KK,K,KQ
  real(chm_real)  NORM
  real(chm_real)  CCC,CMIJ,RPL,CMP,SMP,APL
  real(chm_real)  SP,CP,ST,CT,R,R2,XDIFF,YDIFF,ZDIFF
  real(chm_real)  DX1,DY1,DZ1,DR,DT,DP
  real(chm_real)  TMPFX,TMPFY,TMPFZ
  !
  integer ncyz,il,ix,iy,iz,n1,in1,n2,in2,n3,in3
  real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
  real(chm_real)  enet1, enelp1, enet2, enelp2    
  real(chm_real)  aisign,bisign,cisign,prefac

  ! B-spline
  integer jx1,jx2,jy1,jy2,jz1,jz2,ipx,ipy,ipz,nfil
  real(chm_real)  xc,yc,zc!,M3,DM3 
  ! XIAO_PHK_QC_UW0609: DIV
  integer K1,K2, IQ, IS
  REAL(chm_real) NE, DQ


  !      QC: Evaluate and collect QM-GSBP components for energy and
  !      forces.
  !      The mulliken charges of the QM atoms have been saved in /sccmul/
  !      QMULI2(*,1).

  !      FOR FEP - scale the forces otherwise not.
  if(qlamda.eqv..false.) scal=1.0d0

  !      QC:UW_041705 Debug
  !      write(*,*) "Arrived in SCCGSBP4?",NSCCTC
  IF (NSCCTC.EQ.0) RETURN
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

  ncyz=ncly*nclz 
  !      ............ Outer potential contributions ---> only to QM

  IF(QBSPL) THEN
     DO KQ=1,NSCCRP
        DO I=1,NSCCTC
           chi=QMULI2(I,KQ)
           rxnafx((kq-1)*nscctc+i)=zero
           rxnafy((kq-1)*nscctc+i)=zero
           rxnafz((kq-1)*nscctc+i)=zero  
           IF(CHI.EQ.0.0) GOTO 492
           JQ=IQMLST (I,KQ)

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
                    !
                    !     Reaction Field Force  !!! Be careful of the sign (-1/dcel)
                    !
                    prefac=phi1(ipz)*CCELEC*chi/dcel
                    RXNAFx((kq-1)*nscctc+i)=RXNAFx((kq-1)*nscctc+i)  &
                         + DM3(ai)*M3(bi)*M3(ci)*prefac
                    RXNAFy((kq-1)*nscctc+i)=RXNAFy((kq-1)*nscctc+i)  &
                         + M3(ai)*DM3(bi)*M3(ci)*prefac
                    RXNAFz((kq-1)*nscctc+i)=RXNAFz((kq-1)*nscctc+i)  &
                         + M3(ai)*M3(bi)*DM3(ci)*prefac
                    !
                    !     Electrostatic Energy
                    !
                    !                    enelp1=enelp1+FACTOR*fi*chi*phi1(ipz)*CCELEC

                 enddo
              enddo
           enddo


492        continue
        ENDDO ! I
     ENDDO ! KQ
  ELSE

     DO K=1,NSCCRP
        DO I=1,NSCCTC
           !      ...... assign the probe charge to a lattice and then sum over
           !      ...... the phi at those lattice points
           chi=QMULI2(I,K)
           rxnafx((k-1)*nscctc+i)=zero
           rxnafy((k-1)*nscctc+i)=zero
           rxnafz((k-1)*nscctc+i)=zero
           JQ=IQMLST (I,K)
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
                       RXNAFx((k-1)*nscctc+i)=RXNAFx((k-1)*nscctc+i) &
                            +aisign*bi*ci*prefac
                    endif
                    !
                    if((bi.lt.(one-rsmall)).and.(bi.gt.rsmall)) then
                       RXNAFy((k-1)*nscctc+i)=RXNAFy((k-1)*nscctc+i) &
                            +bisign*ai*ci*prefac
                    endif
                    !
                    if((ci.lt.(one-rsmall)).and.(ci.gt.rsmall)) then
                       RXNAFz((k-1)*nscctc+i)=RXNAFz((k-1)*nscctc+i) &
                            +cisign*ai*bi*prefac
                    endif

                 enddo !n3
              enddo ! n2
           enddo !n1

        ENDDO ! loop over QM atoms
     ENDDO ! loop over replica
  ENDIF

  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
     do k=1,nsccrp
        do i=1,nscctc
           write(outu,'(3x,i5,4x,3f10.5)')  &
                i,rxnafx((k-1)*nscctc+i),rxnafy((k-1)*nscctc+i), &
                rxnafz((k-1)*nscctc+i)
        enddo
     enddo ! loop over replica
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO K=1,NSCCRP
     DO I=1,NSCCTC
        DX(IQMLST(I,K))=DX(IQMLST(I,K)) - RXNAFx((k-1)*nscctc+i) &
             *scal/DBLE(NSCCRP)
        DY(IQMLST(I,K))=DY(IQMLST(I,K)) - RXNAFy((k-1)*nscctc+i) &
             *scal/DBLE(NSCCRP)
        DZ(IQMLST(I,K))=DZ(IQMLST(I,K)) - RXNAFz((k-1)*nscctc+i) &
             *scal/DBLE(NSCCRP)
     ENDDO
  ENDDO ! loop over replica
  !

  !      ............ QMQ type of Rxn field terms, both QM and MM atoms
  !      ...... Build Q for QM 
  LMAXSCC=NMPOL-1
  SRDIST2=(SRDIST+RSMALL)*(SRDIST+RSMALL)
  DO K=1,NSCCRP
     DO II=1,NTPOL
        COEF_QM((K-1)*NTPOL+II)=ZERO
     ENDDO
  ENDDO
  DO K=1,NSCCRP
     DO I=1,NSCCTC
        J =IQMLST(I,K)
        CCC=QMULI2(I,K)
        IF(CCC.EQ.ZERO) CYCLE
        XDIFF=X(J)-RRXCEN
        YDIFF=Y(J)-RRYCEN
        ZDIFF=Z(J)-RRZCEN
        R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
        IF(R2.GT.SRDIST2) CYCLE
        R=SQRT(R2)
        CT=ZDIFF/R
        ST=SQRT(ONE-(ZDIFF*ZDIFF)/R2)
        CP=XDIFF/R/ST
        SP=YDIFF/R/ST
        IF(R2.LT.RSMALL) THEN                               ! in the origin
           CT=ZERO
           CP=ZERO
           SP=ZERO
        ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
             YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
           CT=ONE
           IF(ZDIFF.LT.ZERO) CT=-ONE
           CP=ZERO
           SP=ZERO
        ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
           CT=ZERO
           CP=XDIFF/R
           SP=YDIFF/R
        ENDIF

        CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
        CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
        CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
        CALL ALPOL2(LMAXSCC,CT,AP)       !  fill AP  (P(lm) ) array

        DO II=1,NTPOL
           L=LSTPL(II)
           M=LSTPM(II)
           NORM=BNORM(II)
           IF(L.GE.0.AND.M.EQ.0) THEN
              COEF_QM((K-1)*NTPOL+II)=COEF_QM((K-1)*NTPOL+II) &
                   +CCC*NORM*AR(L)*AP(L,M)
           ELSEIF(L.GT.0.AND.M.GT.0) THEN
              COEF_QM((K-1)*NTPOL+II)=COEF_QM((K-1)*NTPOL+II) &
                   +CCC*NORM*AR(L)*AC(M)*AP(L,M)
           ELSEIF(L.GT.0.AND.M.LT.0) THEN
              M=-M
              COEF_QM((K-1)*NTPOL+II)=COEF_QM((K-1)*NTPOL+II) &
                   +CCC*NORM*AR(L)*AS(M)*AP(L,M)
           ENDIF
        ENDDO
     ENDDO
  ENDDO ! loop over replica

  !      >>>>>> First treat QM derivatives
  !      ...... Build the M*Q by augumenting with QM contributions ......

  DO K=1,NSCCRP
     DO I=1,MAXNPOL
        II=LSTPOL(I)
        MQ(II)=ZERO
        DO J=1,MAXNPOL
           JJ=LSTPOL(J)
           IJ=(II-1)*NTPOL+JJ
           IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
           MQ(II)=MQ(II)+MIJ(IJ)*(COEF(JJ) - COEFX(JJ)  &
                + COEF_QM(JJ+(K-1)*MAXNPOL))
        ENDDO
     ENDDO
  ENDDO ! loop over replica

  DO KK=1,NSCCRP
     DO LL=1,NSCCTC
        !        MM=IQMLST(LL,1)
        RXNBFX((KK-1)*NSCCTC+LL)=ZERO
        RXNBFY((KK-1)*NSCCTC+LL)=ZERO
        RXNBFZ((KK-1)*NSCCTC+LL)=ZERO
     ENDDO
  ENDDO ! loop over replica
  !      ...... Now proceed for QM atoms 
  ! reaction field force calculations     
  DO KK=1,NSCCRP
     DO LL=1,NSCCTC
        MM=IQMLST(LL,KK)
        XDIFF=X(MM)-RRXCEN
        YDIFF=Y(MM)-RRYCEN
        ZDIFF=Z(MM)-RRZCEN
        TMPFX=ZERO
        TMPFY=ZERO
        TMPFZ=ZERO
        CCC=QMULI2(LL,KK)*CCELEC
        IF(CCC.EQ.ZERO) CYCLE
        R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
        R=SQRT(R2)
        CT=ZDIFF/R
        ST=SQRT(ONE-(ZDIFF*ZDIFF)/(R2))
        CP=XDIFF/R/ST
        SP=YDIFF/R/ST
        IF(R2.LT.RSMALL) THEN                               ! in the origin
           CT=ZERO
           ST=ZERO
           CP=ZERO
           SP=ZERO
        ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
             YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
           CT=ONE
           IF(ZDIFF.LT.ZERO) CT=-ONE
           ST=ZERO
           CP=ZERO
           SP=ZERO
        ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
           CT=ZERO
           ST=ONE
           CP=XDIFF/R
           SP=YDIFF/R
        ENDIF

        CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
        CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
        CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
        CALL ALPOL2(LMAXSCC,CT,AP)            !  fill AP  (P(lm) ) array
        CALL DALPOL2(LMAXSCC,CT,AP,ADP)       !  fill ADP (DP(lm)) array

        DO I=1,MAXNPOL
           II=LSTPOL(I)
           L=LSTPL(II)
           M=LSTPM(II)
           NORM=BNORM(II)
           IF(M.EQ.0) THEN
              IF(L.EQ.0) THEN
                 DR=ZERO
                 DT=ZERO
                 DP=ZERO
              ELSE
                 RPL=ONE
                 IF(L.NE.0) RPL=AR(L-1)
                 DR= L*RPL*AP(L,M)
                 DT=-RPL*ADP(L,M)*ST
                 DP=ZERO
              ENDIF
           ELSEIF(M.GT.0) THEN
              RPL=ONE
              IF(L.NE.0) RPL=AR(L-1)
              CMP=AC(M)
              APL=AP(L,M)
              DR= L*RPL*CMP*APL
              DT=-RPL*CMP*ADP(L,M)*ST
              DP=-RPL*M*AS(M)*APL/ST
              IF(ST.EQ.ZERO) DP=ZERO
           ELSEIF(M.LT.0) THEN
              M=-M
              RPL=ONE
              IF(L.NE.0) RPL=AR(L-1)
              SMP=AS(M)
              APL=AP(L,M)
              DR= L*RPL*SMP*APL
              DT=-RPL*SMP*ADP(L,M)*ST
              DP= RPL*M*AC(M)*APL/ST
              IF(ST.EQ.ZERO) DP=ZERO
           ENDIF
           DX1=NORM*(DR*ST*CP+DT*CT*CP-DP*SP)
           DY1=NORM*(DR*ST*SP+DT*CT*SP+DP*CP)
           DZ1=NORM*(DR*CT   -DT*ST         )
           TMPFX=TMPFX-DX1*MQ(II)
           TMPFY=TMPFY-DY1*MQ(II)
           TMPFZ=TMPFZ-DZ1*MQ(II)
        ENDDO
        RXNBFX((KK-1)*NSCCTC+LL)=TMPFX*CCC
        RXNBFY((KK-1)*NSCCTC+LL)=TMPFY*CCC
        RXNBFZ((KK-1)*NSCCTC+LL)=TMPFZ*CCC
     ENDDO
  ENDDO

  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(3x,a)') '# ATOM       Fx        Fy        Fz'
     do k=1,nsccrp
        do i=1,nscctc
           write(outu,'(3x,i5,4x,3f10.5)')  &
                i,rxnbfx((k-1)*nscctc+i),rxnbfy((k-1)*nscctc+i), &
                rxnbfz((k-1)*nscctc+i)
        enddo
     enddo ! loop over replica
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO K=1,NSCCRP
     DO I=1,NSCCTC
        DX(IQMLST(I,K))=DX(IQMLST(I,K)) - RXNBFx((k-1)*nscctc+i) &
             *scal/dble(nsccrp)
        DY(IQMLST(I,K))=DY(IQMLST(I,K)) - RXNBFy((k-1)*nscctc+i) &
             *scal/dble(nsccrp)
        DZ(IQMLST(I,K))=DZ(IQMLST(I,K)) - RXNBFz((k-1)*nscctc+i) &
             *scal/dble(nsccrp)
     ENDDO
  ENDDO ! loop over replica

  !      >>>>>> Next treat MM derivatives 
  !      ...... Build the M*Q by including only QM contributions ...... 
  !
  DO K=1,NSCCRP
     DO I=1,MAXNPOL
        II=LSTPOL(I)
        MQ(II)=ZERO
        DO J=1,MAXNPOL
           JJ=LSTPOL(J)
           IJ=(II-1)*NTPOL+JJ
           IF(II.GT.JJ) IJ=(JJ-1)*NTPOL+II
           MQ(II)=MQ(II)+MIJ(IJ)*COEF_QM(JJ+(K-1)*MAXNPOL)
        ENDDO
     ENDDO
  ENDDO ! loop over replica

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
     ! XIAO_PHK_QC_UW0609 END DIV
     R2=XDIFF*XDIFF+YDIFF*YDIFF+ZDIFF*ZDIFF
     R=SQRT(R2)
     CT=ZDIFF/R
     ST=SQRT(ONE-(ZDIFF*ZDIFF)/(R2))
     CP=XDIFF/R/ST
     SP=YDIFF/R/ST
     IF(R2.LT.RSMALL) THEN                               ! in the origin
        CT=ZERO
        ST=ZERO
        CP=ZERO
        SP=ZERO
     ELSEIF(XDIFF.GT.-RSMALL.AND.XDIFF.LT.RSMALL.AND.     & ! in the Z-axis
          YDIFF.GT.-RSMALL.AND.YDIFF.LT.RSMALL) THEN
        CT=ONE
        IF(ZDIFF.LT.ZERO) CT=-ONE
        ST=ZERO
        CP=ZERO
        SP=ZERO
     ELSEIF(ZDIFF.GT.-RSMALL.AND.ZDIFF.LT.RSMALL) THEN   ! in the XY plane
        CT=ZERO
        ST=ONE
        CP=XDIFF/R
        SP=YDIFF/R
     ENDIF

     CALL RPOWERL2(LMAXSCC,R,AR)           !  fill AR  (r^l   ) array
     CALL COSMPHI2(LMAXSCC,CP,AC)          !  fill AC  (cos.. ) array
     CALL SINMPHI2(LMAXSCC,CP,SP,AS)       !  fill AS  (sin.. ) array
     CALL ALPOL2(LMAXSCC,CT,AP)            !  fill AP  (P(lm) ) array
     CALL DALPOL2(LMAXSCC,CT,AP,ADP)       !  fill ADP (DP(lm)) array

     DO I=1,MAXNPOL
        II=LSTPOL(I)
        L=LSTPL(II)
        M=LSTPM(II)
        NORM=BNORM(II)
        IF(M.EQ.0) THEN
           IF(L.EQ.0) THEN
              DR=ZERO
              DT=ZERO
              DP=ZERO
           ELSE
              RPL=ONE
              IF(L.NE.0) RPL=AR(L-1)
              DR= L*RPL*AP(L,M)
              DT=-RPL*ADP(L,M)*ST
              DP=ZERO
           ENDIF
        ELSEIF(M.GT.0) THEN
           RPL=ONE
           IF(L.NE.0) RPL=AR(L-1)
           CMP=AC(M)
           APL=AP(L,M)
           DR= L*RPL*CMP*APL
           DT=-RPL*CMP*ADP(L,M)*ST
           DP=-RPL*M*AS(M)*APL/ST
           IF(ST.EQ.ZERO) DP=ZERO
        ELSEIF(M.LT.0) THEN
           M=-M
           RPL=ONE
           IF(L.NE.0) RPL=AR(L-1)
           SMP=AS(M)
           APL=AP(L,M)
           DR= L*RPL*SMP*APL
           DT=-RPL*SMP*ADP(L,M)*ST
           DP= RPL*M*AC(M)*APL/ST
           IF(ST.EQ.ZERO) DP=ZERO
        ENDIF
        DX1=NORM*(DR*ST*CP+DT*CT*CP-DP*SP)
        DY1=NORM*(DR*ST*SP+DT*CT*SP+DP*CP)
        DZ1=NORM*(DR*CT   -DT*ST         )
        TMPFX=TMPFX-DX1*MQ(II)
        TMPFY=TMPFY-DY1*MQ(II)
        TMPFZ=TMPFZ-DZ1*MQ(II)
     ENDDO
     RXNBFX(MM)=TMPFX*CCC
     RXNBFY(MM)=TMPFY*CCC
     RXNBFZ(MM)=TMPFZ*CCC
  ENDDO

  !      UW_0605:Haibo Yu
  !      Debug printing
  IF(PRNLEV.GT.6) THEN
     WRITE(outu,'(/,3x,A)')    &
          'The Reaction Field Forces [KCAL/MOL/A] at Each Atom'
     Write(outu,'(2x,a)') '# ATOM LSTRB IGMSEL &
          Fx        Fy        Fz'
     do il=1,ntrb
        i=lstrb(il)
        write(outu,'(3x,i5,3x,i5,3x,i5,4x,3f10.5)')  &
             il,i,igmsel(i),rxnbfx(i),rxnbfy(i),rxnbfz(i)
     enddo
  ENDIF

  !      Update DX,DY,DZ - DO NOT FORGET TO SCALE THE FORCES WHEN FEP
  DO IL=1,NTRB
     I=LSTRB(IL)
     DX(I)=DX(I) - RXNBFx(i)*scal/DBLE(NSCCRP)
     DY(I)=DY(I) - RXNBFy(i)*scal/DBLE(NSCCRP)
     DZ(I)=DZ(I) - RXNBFz(i)*scal/DBLE(NSCCRP)
  ENDDO
  return
end SUBROUTINE SCCGSBP4

#endif /*  SCCDFTB*/
#endif /*  GSBP*/
#endif /*  PBEQ*/

