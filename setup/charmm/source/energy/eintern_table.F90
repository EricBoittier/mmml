module bond_tables
  use chm_kinds
  use dimens_fcm
  implicit none

  private
  logical :: lbond_tables=.false., langle_tables=.false.
  logical :: ldihedral_tables=.false.
  integer :: ntabtp,num_bnd_tbl,num_ang_tbl,num_dih_tbl
  integer,allocatable,dimension(:) :: bndtype_lst
  integer,allocatable,dimension(:,:) :: angtype_lst
  integer,allocatable,dimension(:,:) :: dihtype_lst

  type bond_table_structure
     integer :: len
     real(chm_real) :: tmin,tmax,dt,dtinv
     real(chm_real),allocatable,dimension(:) :: t,v,d 
     integer :: ida,idb,idc,idd
  end type bond_table_structure
  type(bond_table_structure),allocatable,dimension(:) :: bond_tbl,angle_tbl
  type(bond_table_structure),allocatable,dimension(:) :: dihedral_tbl
  public lbond_tables,langle_tables,ldihedral_tables,&
       ebond_table,eangle_table,read_bond_tables,read_angle_tables,&
       read_dihedral_tables,ephi_table
  
contains

  SUBROUTINE EBOND_table(EB,IB,JB,NBOND,DX,DY,DZ,X,Y,Z,qsecd)
    !   *********t a b l e   l o o k - u p   v e r s i o n ***********
    !
    !
    !     r(ntabln)=the lnfft grid
    !     vee(ntabsq,ntabln)=the potential (of mean force)
    !     ft (ntabsq,ntabln)=the derivative of the potential (of mean force)
    !                                                                      c
    !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    !

    use consta
    use number
#if KEY_DIMB==1
    use dimb  
#endif
    use psf,only:natom,iac
    use stream,only:outu
    use actclus_mod
    use blockscc_fcm
#if KEY_PARALLEL==1
    use parallel  
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec          
#endif
#if KEY_DOMDEC==1
    use domdec_bonded,only:nbondtbl,bondtbl  
#endif
    implicit none
    !
    integer,intent(in) :: NBOND
    real(chm_real) :: EB
    integer,dimension(nbond) :: IB,JB
    logical QSECD
    real(chm_real),dimension(natom) :: x,y,z,dx,dy,dz,cg

    real(chm_real) :: denom,df,rij,rij2,rrij,r0,dr,drinv,e12, dxi,dyi,dzi
    integer iaci,iacj,ibnd,icna,ipc,ipc0
    integer i,j,n,mm
    integer ioff(maxatc)
    logical lsecd
#if KEY_ACTBOND==1
    integer :: III, UPLIMIT  
#endif
#if KEY_DOMDEC==1
    integer nn,nnstart,nnend,nnadd  
#endif

    lsecd=qsecd

    do i=1,maxatc
       ioff(i)=(i*(i-1))/2
    enddo
    !

#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       nnstart = 1
       nnend = nbondtbl
       nnadd = 1
    else
       nnstart = mynodp
       nnend = nbond
       nnadd = numnod
    endif
    loop300: do nn=nnstart,nnend,nnadd
       if (q_domdec) then
          ibnd = bondtbl(nn)
       else
          ibnd = nn
       endif
#else /* (domdec)*/
#if KEY_PARALLEL==1 /*parabond*/
#if KEY_VIBPARA==1
    loop300: do ibnd=1,nbond
#else /**/

#if KEY_PARAFULL==1 /*parfbond*/
    loop300: do ibnd=mynodp,nbond,numnod
#elif KEY_PARASCAL==1 /*parfbond*/
    NOPARS=(ICONBH.GE.0)
    loop300: do ibnd=1,nbond
       IF(NOPARS) THEN
          IF(JPMAT(IPBLOCK(IB(ibnd)),IPBLOCK(JB(ibnd))).NE.MYNOD) cycle loop300
       ENDIF
#elif KEY_SPACDEC==1 /*parfbond*/
    loop300: do ibnd=1,nbond
       IF(ICPUMAP(IB(ibnd)) /= MYNOD) cycle loop300
#endif /* (parfbond)*/
#endif 

#else /* (parabond)*/
#if KEY_ACTBOND==1 /*actbond*/
    IF(QBACTON) THEN
       UPLIMIT = NACTBND
    ELSE
       UPLIMIT = NBOND
    ENDIF
    loop300: DO III = 1,UPLIMIT
       IF (QBACTON) THEN
          ibnd = actbond(iii)
       ELSE
          ibnd = iii
       ENDIF
#else /* (actbond)*/
    loop300: do ibnd=1,nbond
#endif /* (actbond)*/
#endif /* (parabond)*/
#endif /* (domdec)*/
       i=ib(ibnd)
       j=jb(ibnd)
       ipc=bndtype_lst(ibnd)
       !---- find bond length rij ------
       dxi=x(i)-x(j)
       dyi=y(i)-y(j)
       dzi=z(i)-z(j)
       rij2=dxi*dxi+dyi*dyi+dzi*dzi
       rrij=one/sqrt(rij2)
       rij = rrij*rij2
       !
       !  get the relavent lower index from the grid
       !
       dr = bond_tbl(ipc)%dt
       drinv = bond_tbl(ipc)%dtinv
       r0 = bond_tbl(ipc)%tmin
       n=int((rij-r0)*drinv) + 1
       if(n < 1 .or. n >= bond_tbl(ipc)%len) then
          write(outu,'("   Out of range atoms",2i6,f12.4)') i,j,rij
          write(outu,'("   Out of range value",f10.4," table min/max",2f10.4)') &
               rij,r0,bond_tbl(ipc)%tmax
          write(outu,'( "xyz(i) = ",3f10.4)') x(i),y(i),z(i)
          write(outu,'( "xyz(j) = ",3f10.4)') x(j),y(j),z(j)
          call wrndie(-3,'ebond_table<eintern_table>','lookup element out of range')
       endif
       
       ! linear interpolation probably will not conserve energy
       e12=zero
       !              print *,"bond",ibnd,"  atoms",i,j
       !              print '("ipc,n,iaci,iacj",4i4,2e12.4)',ipc,n,iaci,iacj,rij,r0
       e12 = bond_tbl(ipc)%v(n) + bond_tbl(ipc)%d(n)*(rij-bond_tbl(ipc)%t(n)) + & 
            ( bond_tbl(ipc)%d(n+1) - bond_tbl(ipc)%d(n) ) &           
             * (rij-bond_tbl(ipc)%t(n))**2*drinv*0.5

       eb=eb+e12

       !
       !   now for the derivatives
       !
       df = bond_tbl(ipc)%d(n) + ( bond_tbl(ipc)%d(n+1) - bond_tbl(ipc)%d(n) ) &
            * (rij-bond_tbl(ipc)%t(n))*drinv
       dx(j)=dx(j)-dxi*df/rij
       dx(i)=dx(i)+dxi*df/rij
       dy(j)=dy(j)-dyi*df/rij
       dy(i)=dy(i)+dyi*df/rij
       dz(j)=dz(j)-dzi*df/rij
       dz(i)=dz(i)+dzi*df/rij

10     continue
    enddo loop300
    !
    return
  end subroutine EBOND_table

  !=================================================================================
  !                    EANGLE_table
  !
  SUBROUTINE EANGLE_table(ET,IT,JT,KT,NTHETA,DX,DY,DZ,X,Y,Z,QSECD)

    !-----------------------------------------------------------------------
    !     Looks up ANGLE ENERGIES.
    !
    use number
    use consta,only:cosmax,pi,raddeg
    use psf,only:natom,iac
    use econtmod,only:qecont,econt
    use stream,only:idleng,outu,prnlev
    use actclus_mod
#if KEY_PARALLEL==1
    use parallel  
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec           
#endif
#if KEY_DOMDEC==1
    use domdec_bonded,only:nangletbl,angletbl 
#endif
    implicit none
#if KEY_PARALLEL==1
    LOGICAL NOPARS 
#endif

    real(chm_real) et
    integer,intent(in) :: ntheta
    integer,dimension(ntheta),intent(in) :: it,jt,kt
    real(chm_real),dimension(natom) ::  dx,dy,dz,x,y,z
    logical,intent(in) :: qsecd
    real(chm_real) dxi,dyi,dzi,dxj,dyj,dzj,ri2,rj2,ri,rj
    real(chm_real) rir,rjr,dxir,dyir,dzir,dxjr,dyjr,dzjr,cst,at,at0, &
         da,df,e, dt,dtinv
    real(chm_real) st2r,str,dtxi,dtxj,dtyi,dtyj,dtzi,dtzj
    real(chm_real) dfx,dfy,dfz,dgx,dgy,dgz,ddf,ri2rf,rj2rf,rirjf,smallv
    integer i,nwarn,ith,j,k,ic,jj,ii,kk
    logical ijtest,iktest,jktest
    integer ith0,ith1,ithd, n
    integer iaci,iacj,iack,ipc,ipc0,jindex
    integer ioff(maxatc)
#if KEY_ACTBOND==1
    integer :: III, UPLIMIT  
#endif
#if KEY_DOMDEC==1
    integer nn,nnstart,nnend,nnadd  
#endif

    do i=1,maxatc
       ioff(i)=(i*(i-1))/2
    enddo
    et=zero
    smallv=rpreci
    nwarn=0
    if(ntheta.eq.0) return
    !

#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       nnstart = 1
       nnend = nangletbl
       nnadd = 1
    else
       nnstart = mynodp
       nnend = ntheta
       nnadd = numnod
    endif
    loop20: do nn=nnstart,nnend,nnadd
       if (q_domdec) then
          ith = angletbl(nn)
       else
          ith = nn
       endif
#else /* (domdec)*/
#if KEY_PARALLEL==1 /*paraangle*/
#if KEY_VIBPARA==1
    loop20: do ith=1,ntheta
#else /**/

#if KEY_PARAFULL==1 /*parfangle*/
    loop20: do ith=mynodp,ntheta,numnod
#elif KEY_PARASCAL==1 /*parfangle*/
    NOPARS=(ICONAH.GE.0)
    loop20: do ith=1,ntheta
       IF(NOPARS) THEN
          II=IPBLOCK(IT(ITH))
          JJ=IPBLOCK(JT(ITH))
          KK=IPBLOCK(KT(ITH))
          IC=II
          IF(II.EQ.JJ) IC=KK
          IF(KK.NE.JJ .AND. KK.NE.IC) THEN
             !           the angle spans three block.
             !           make sure that we have the coordinates.
             CALL PSADDTOCL(KT(ITH),JPMAT(JJ,IC))
          ENDIF
          IF(JPMAT(JJ,IC).NE.MYNOD) cycle loop20
       ENDIF
#elif KEY_SPACDEC==1 /*parfangle*/
    loop20: do ith=1,ntheta
       IF(ICPUMAP(IT(ITH)) /= MYNOD) cycle loop20
#endif /* (parfangle)*/
#endif 
#else /* (paraangle)*/
#if KEY_ACTBOND==1 /*actbond*/
    IF(QBACTON) THEN
       UPLIMIT = NACTANG
    ELSE
       UPLIMIT = NTHETA
    ENDIF
    loop20: DO III = 1,UPLIMIT
       IF (QBACTON) THEN
          ITH = ACTANGL(III)
       ELSE
          ITH = III
       ENDIF
#else /* (actbond)*/
    loop20: do ith=1,ntheta
#endif /* (actbond)*/
#endif /* (paraangle)*/
#endif /* (domdec)*/
       !
       i=it(ith)
       j=jt(ith)
       k=kt(ith)
       !--- figure out which table to use -----
       iaci=iac(i)
       iacj=iac(j)
       iack=iac(k)
       !       jindex = angtype_lst_index(iacj)
       ipc=find_angle_table(iaci,iacj,iack)
       dxi=x(i)-x(j)
       dyi=y(i)-y(j)
       dzi=z(i)-z(j)
       dxj=x(k)-x(j)
       dyj=y(k)-y(j)
       dzj=z(k)-z(j)
       ri2=dxi*dxi+dyi*dyi+dzi*dzi
       rj2=dxj*dxj+dyj*dyj+dzj*dzj
       ri=sqrt(ri2)
       rj=sqrt(rj2)
       rir=one/ri
       rjr=one/rj
       dxir=dxi*rir
       dyir=dyi*rir
       dzir=dzi*rir
       dxjr=dxj*rjr
       dyjr=dyj*rjr
       dzjr=dzj*rjr
       cst=dxir*dxjr+dyir*dyjr+dzir*dzjr
       !
       if(abs(cst).ge.cosmax) then
          if(abs(cst).gt.one) cst=sign(one,cst)
          at=acos(cst)
          if(abs(da).gt.0.1) then
             nwarn=nwarn+1
             if((nwarn.le.5 .and. wrnlev.ge.5) .or. wrnlev.ge.6) then
                WRITE(OUTU,10) ITH,I,J,K
10              FORMAT(' WARNING FROM EANGLE. Angle',I5, &
                     '  is almost linear.', &
                     /' Derivatives may be affected for atoms:',3I5)
                WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
                WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
                WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
                WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
                WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
                WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
101             FORMAT(5X,A,5F15.5)
             ENDIF
          ENDIF
       ENDIF
       !
       AT=ACOS(CST)

       !
       !  get the relavent lower index from the grid
       !
       dt = angle_tbl(ipc)%dt
       dtinv = angle_tbl(ipc)%dtinv
       at0 = angle_tbl(ipc)%tmin
       n=int((at-at0)*dtinv) + 1
       if(n < 1 .or. n >= angle_tbl(ipc)%len) then
          write(outu,'("   Out of range atoms",2i6,f12.4)') i,j,at
          write(outu,'("   Out of range value",f10.4," table min/max",2f10.4)') &
               at,at0,angle_tbl(ipc)%tmax
          call wrndie(-3,'eangle_table<eintern_table>','lookup element out of range')
       endif

       ! linear interpolation probably will not conserve energy
       e=zero

       e = angle_tbl(ipc)%v(n) + angle_tbl(ipc)%d(n)*(at-angle_tbl(ipc)%t(n)) + &
            ( angle_tbl(ipc)%d(n+1) - angle_tbl(ipc)%d(n) ) &
            * (at-angle_tbl(ipc)%t(n))**2*dtinv*0.5

       et=et+e

       !
       !   now for the derivatives
       !
       df = angle_tbl(ipc)%d(n) + ( angle_tbl(ipc)%d(n+1) - angle_tbl(ipc)%d(n) ) &
            * (at-angle_tbl(ipc)%t(n))*dtinv

       IF(QECONT) THEN
          E=E*THIRD
          ECONT(I)=ECONT(I)+E
          ECONT(J)=ECONT(J)+E
          ECONT(K)=ECONT(K)+E
       ENDIF
       !
       IF(ABS(CST).GE.0.999) THEN
          ST2R=ONE/(ONE-CST*CST+SMALLV)
          STR=SQRT(ST2R)
          !             DF=MINTWO*CTC(IC)*(ONE+DA*DA*SIXTH)
          !             DF=TWO*CTC(IC)*(ONE+DA*DA*SIXTH)
          !          ELSE
          DF=-DF*STR
          !          ENDIF
       ELSE
          ST2R=ONE/(ONE-CST*CST)
          STR=SQRT(ST2R)
          DF=-DF*STR
       ENDIF
       !
       DTXI=RIR*(DXJR-CST*DXIR)
       DTXJ=RJR*(DXIR-CST*DXJR)
       DTYI=RIR*(DYJR-CST*DYIR)
       DTYJ=RJR*(DYIR-CST*DYJR)
       DTZI=RIR*(DZJR-CST*DZIR)
       DTZJ=RJR*(DZIR-CST*DZJR)
       !

       DFX=DF*DTXI
       DGX=DF*DTXJ
       DX(I)=DX(I)+DFX
       DX(K)=DX(K)+DGX
       DX(J)=DX(J)-DFX-DGX
       !
       DFY=DF*DTYI
       DGY=DF*DTYJ
       DY(I)=DY(I)+DFY
       DY(K)=DY(K)+DGY
       DY(J)=DY(J)-DFY-DGY
       !
       DFZ=DF*DTZI
       DGZ=DF*DTZJ
       DZ(I)=DZ(I)+DFZ
       DZ(K)=DZ(K)+DGZ
       DZ(J)=DZ(J)-DFZ-DGZ

    enddo loop20
    !
    IF(NWARN.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,30) NWARN
30  FORMAT(' TOTAL OF',I6,' WARNINGS FROM eANGLE_table')
    !
    return
  end subroutine eangle_table

  SUBROUTINE ephi_table(EP,IP,JP,KP,LP,NPHI,DX,DY,DZ,X,Y,Z,QSECD)
    !-----------------------------------------------------------------------
    !
    !     The parameters of the routine are:
    !
    !     EP          <- Dihedral Energy
    !     IP,JP,KP,LP(phi) -> atom number for the members of dihedrals
    !     NPHI        ->  Number of dihedrals
    !     DX,DY,DZ(atom) <-> Force matrices
    !     X,Y,Z(atom) -> Coordinate matrices
    !     QSECD       -> Second derivative flag.
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    use econtmod,only:qecont,econt
#if KEY_SCCDFTB==1
    use blockscc_fcm  
#endif
    use cnst_fcm
#if KEY_PARALLEL==1
    use parallel  
#endif
    use block_fcm
    use lambdam
    use pert  !Cc New PBLOCK
    use dimb
    use consta,only:pi
    use stream,only:outu,prnlev
#if KEY_GENETIC==1
    use galgor     
#endif
    use chutil,only:atomid
    use psf,only:natom,iac
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec          
#endif
#if KEY_DOMDEC==1
    use domdec_bonded,only:ndihetbl,dihetbl  
#endif
    !
    implicit none
    !
#if KEY_PARALLEL==1
    LOGICAL NOPARS  
#endif
    real(chm_real) EP
    INTEGER IP(*),JP(*),KP(*),LP(*)
    INTEGER NPHI
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    LOGICAL QSECD
    !
    real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA2R,RB2R,RG2,RG,RGR,RGR2
    real(chm_real) RABR,CP,E,DF
    real(chm_real) GAA,GBB,FG,HG,FGA,HGB,FGRG2,HGRG2,DFRG3
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    real(chm_real) DTFX,DTFY,DTFZ,DTHX,DTHY,DTHZ,DTGX,DTGY,DTGZ
    real(chm_real) GAFX,GAFY,GAFZ,GBHX,GBHY,GBHZ
    real(chm_real) FAGX,FAGY,FAGZ,HBGX,HBGY,HBGZ
    INTEGER NWARN,NWARNX,IPHI,I,J,K,L
    INTEGER II,JJ,KK,LL
    integer iaci,iacj,iack,iacl,ipc,ipc0,n,n1
    real(chm_real) abx,aby,abz,fact,dt,dtinv,phi,phi0
    !
#if KEY_BLOCK==1
    INTEGER IBL, JBL, KKK, LLL, KDOC
    real(chm_real)  COEF, DOCFI, DOCFJ, DOCFK, DOCFJ1, DOCFK1, DOCFL
#endif /*  BLOCK*/
#if KEY_DOMDEC==1
    integer nn,nnstart,nnend,nnadd  
#endif
    !
    real(chm_real), parameter :: RXMIN=0.005D0, RXMIN2=0.000025D0
    !
#if KEY_GENETIC==1
    INTEGER First
    First = 1
    If(qGA_Ener) First = Int(EP)
#endif 
    EP=ZERO


    IF(NPHI.LE.0) RETURN
    NWARN=0
    NWARNX=0
    !
#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       nnstart = 1
       nnend = ndihetbl
       nnadd = 1
    else
       nnstart = mynodp
       nnend = nphi
       nnadd = numnod
    endif
    do nn=nnstart,nnend,nnadd
       if (q_domdec) then
          iphi = dihetbl(nn)
       else
          iphi = nn
       endif
#else /* (domdec)*/
#if KEY_PARALLEL==1 /*paraphi*/
#if KEY_VIBPARA==1
    DO IPHI=1,NPHI
#else /**/
       
#if KEY_PARAFULL==1 /*parfphi*/
    DO IPHI=MYNODP,NPHI,NUMNOD
#elif KEY_PARASCAL==1 /*parfphi*/
    NOPARS=(ICONHP.GE.0)
#if KEY_GENETIC==1
    DO IPHI=First,NPHI
#else /**/
    DO IPHI=1,NPHI
#endif 
       IF(NOPARS) THEN
          II=JPBLOCK(IP(IPHI))
          JJ=JPBLOCK(JP(IPHI))
          KK=JPBLOCK(KP(IPHI))
          LL=JPBLOCK(LP(IPHI))
          IA=JJ
          IB=KK
          IF(IA.EQ.IB) IB=LL
          IF(IA.EQ.IB) IB=II
          IF(II.NE.IA .AND. II.NE.IB) THEN
             CALL PSADDTOCL(IP(IPHI),JPMAT(IA,IB))
          ENDIF
          IF(LL.NE.IA .AND. LL.NE.IB) THEN
             CALL PSADDTOCL(LP(IPHI),JPMAT(IA,IB))
          ENDIF
          IF(JPMAT(IA,IB).NE.MYNOD) GOTO 160
       ENDIF
#elif KEY_SPACDEC==1 /*parfphi*/
    DO IPHI=1,NPHI
       IF(MYNOD /= ICPUMAP(IP(IPHI))) GOTO 160
#endif /* (parfphi)*/
#endif 
#else /* (paraphi)*/
#if KEY_GENETIC==1
    DO IPHI=First,NPHI
#else /**/
    DO IPHI=1,NPHI
#endif 
#endif /* (paraphi)*/
#endif /* (domdec)*/
       !
       I=IP(IPHI)
       J=JP(IPHI)
       K=KP(IPHI)
       L=LP(IPHI)
       iaci=iac(i)
       iacj=iac(j)
       iack=iac(k)
       iacl=iac(l)
       ipc=find_dihedral_table(iaci,iacj,iack,iacl)
       !     IC=ICP(IPHI)
       !     IF(IC.EQ.0) GOTO 160
       ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
       FX=X(I)-X(J)
       FY=Y(I)-Y(J)
       FZ=Z(I)-Z(J)
       GX=X(J)-X(K)
       GY=Y(J)-Y(K)
       GZ=Z(J)-Z(K)
       HX=X(L)-X(K)
       HY=Y(L)-Y(K)
       HZ=Z(L)-Z(K)
       ! A=F^G, B=H^G
       AX=FY*GZ-FZ*GY
       AY=FZ*GX-FX*GZ
       AZ=FX*GY-FY*GX
       BX=HY*GZ-HZ*GY
       BY=HZ*GX-HX*GZ
       BZ=HX*GY-HY*GX
       ! RG=|G|, RGR=1/|G|
       RA2=AX*AX+AY*AY+AZ*AZ
       RB2=BX*BX+BY*BY+BZ*BZ
       RG2=GX*GX+GY*GY+GZ*GZ
       RG=SQRT(RG2)
       ! Warnings have been simplified.
       IF((RA2.LE.RXMIN2).OR.(RB2.LE.RXMIN2).OR.(RG.LE.RXMIN)) THEN
          NWARN=NWARN+1
          IF((NWARN.LE.5 .AND. WRNLEV.GE.5) .OR. WRNLEV.GE.6) THEN
             WRITE(OUTU,20) IPHI,I,J,K,L
20           FORMAT(' EPHI_TABLE: WARNING.  dihedral',I5,' is almost linear.'/ &
                  ' derivatives may be affected for atoms:',4I5)
          ENDIF
          GOTO 160
       ENDIF
       !
       RGR=ONE/RG
       RA2R=ONE/RA2
       RB2R=ONE/RB2
       RABR=SQRT(RA2R*RB2R)
       ! CP=cos(phi)
       CP=(AX*BX+AY*BY+AZ*BZ)*RABR
       
       ABX=AY*BZ-AZ*BY
       ABY=AZ*BX-AX*BZ
       ABZ=AX*BY-AY*BX
       fact=-(GX*ABX + GY*ABY + GZ*ABZ)
       
       if (cp > 1.0) then
          phi=0.0
       else if (cp < -1.0) then
          phi=sign(one,fact)*pi
       else
          phi=sign(one,fact)*acos(cp)
       endif
       
       !     if (phi < 0.0) then
       !        phi = phi + 2*pi
       !     endif
       
       dt = dihedral_tbl(ipc)%dt
       dtinv = dihedral_tbl(ipc)%dtinv
       phi0 = dihedral_tbl(ipc)%tmin
       n = int((phi-phi0)*dtinv) + 1
       if (n < 1 .or. n > dihedral_tbl(ipc)%len) then
          write(outu,'("   Out of range atoms",4i6,f12.4)') i,j,k,l,phi
          write(outu,'(a,2f8.4)') 'cp,fact=',cp,fact
          write(outu,'("   Out of range value",f10.4," table min/max",2f10.4)') &
               phi,phi0,dihedral_tbl(ipc)%tmax
          call wrndie(-3,'ephi_table<eintern_table>','lookup element out of range')
       endif
       
       n1 = n + 1
       if (n1 > dihedral_tbl(ipc)%len) then
          n1 = 1
       endif
       
       !     e = dihedral_tbl(ipc)%v(n) + ( dihedral_tbl(ipc)%v(n1) - dihedral_tbl(ipc)%v(n)) &
       !          * (phi-dihedral_tbl(ipc)%t(n))*dtinv
       
       e = dihedral_tbl(ipc)%v(n) + dihedral_tbl(ipc)%d(n)*(phi-dihedral_tbl(ipc)%t(n)) + &
            ( dihedral_tbl(ipc)%d(n+1) - dihedral_tbl(ipc)%d(n) ) &
            * (phi-dihedral_tbl(ipc)%t(n))**2*dtinv*0.5
       
       ep = ep + e
       
       !   now for the derivatives
       !
       df = dihedral_tbl(ipc)%d(n) + (dihedral_tbl(ipc)%d(n1) - dihedral_tbl(ipc)%d(n)) &
            * (phi-dihedral_tbl(ipc)%t(n))*dtinv
       
       !     write (*,*) 'phi,e=',phi,e
       
       ! SP=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       !     SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)
       
       !
       ! Contribution on atoms.
       IF(QECONT) THEN
          E=E*PT25
          ECONT(I)=ECONT(I)+E
          ECONT(J)=ECONT(J)+E
          ECONT(K)=ECONT(K)+E
          ECONT(L)=ECONT(L)+E
       ENDIF
       
       !
       ! Compute derivatives wrt catesian coordinates.
       ! this section is for first derivatives only.
       !
#if KEY_BLOCK==1
       IF (.NOT. NOFORC) THEN
#endif /*  BLOCK*/
          ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
          !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
          FG=FX*GX+FY*GY+FZ*GZ
          HG=HX*GX+HY*GY+HZ*GZ
          FGA=FG*RA2R*RGR
          HGB=HG*RB2R*RGR
          GAA=-RA2R*RG
          GBB=RB2R*RG
          ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
          DTFX=GAA*AX
          DTFY=GAA*AY
          DTFZ=GAA*AZ
          DTGX=FGA*AX-HGB*BX
          DTGY=FGA*AY-HGB*BY
          DTGZ=FGA*AZ-HGB*BZ
          DTHX=GBB*BX
          DTHY=GBB*BY
          DTHZ=GBB*BZ
          ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
          DFX=DF*DTFX
          DFY=DF*DTFY
          DFZ=DF*DTFZ
          DGX=DF*DTGX
          DGY=DF*DTGY
          DGZ=DF*DTGZ
          DHX=DF*DTHX
          DHY=DF*DTHY
          DHZ=DF*DTHZ
          ! Distribute over Ri.
#if KEY_BLOCK==1
#if KEY_DOCK==1
          IF(QDOCK) THEN
             DX(I)=DX(I)+DFX*DOCFI
             DY(I)=DY(I)+DFY*DOCFI
             DZ(I)=DZ(I)+DFZ*DOCFI
             DX(J)=DX(J)-DFX*DOCFJ+DGX*DOCFJ1
             DY(J)=DY(J)-DFY*DOCFJ+DGY*DOCFJ1
             DZ(J)=DZ(J)-DFZ*DOCFJ+DGZ*DOCFJ1
             DX(K)=DX(K)-DHX*DOCFK1-DGX*DOCFK
             DY(K)=DY(K)-DHY*DOCFK1-DGY*DOCFK
             DZ(K)=DZ(K)-DHZ*DOCFK1-DGZ*DOCFK
             DX(L)=DX(L)+DHX*DOCFL
             DY(L)=DY(L)+DHY*DOCFL
             DZ(L)=DZ(L)+DHZ*DOCFL
          ELSE
#endif 
#endif 
             DX(I)=DX(I)+DFX
             DY(I)=DY(I)+DFY
             DZ(I)=DZ(I)+DFZ
             DX(J)=DX(J)-DFX+DGX
             DY(J)=DY(J)-DFY+DGY
             DZ(J)=DZ(J)-DFZ+DGZ
             DX(K)=DX(K)-DHX-DGX
             DY(K)=DY(K)-DHY-DGY
             DZ(K)=DZ(K)-DHZ-DGZ
             DX(L)=DX(L)+DHX
             DY(L)=DY(L)+DHY
             DZ(L)=DZ(L)+DHZ
#if KEY_BLOCK==1
#if KEY_DOCK==1
          ENDIF
#endif 
#endif 
          !
#if KEY_IPRESS==1
          IF(QIPRSS) THEN
             PVIR(I)=PVIR(I)+DFX*FX+DFY*FY+DFZ*FZ
             PVIR(J)=PVIR(J)+DFX*FX+DFY*FY+DFZ*FZ+DGX*GX+DGY*GY+DGZ*GZ
             PVIR(K)=PVIR(K)+DGX*GX+DGY*GY+DGZ*GZ+DHX*HX+DHY*HY+DHZ*HZ
             PVIR(L)=PVIR(L)+DHX*HX+DHY*HY+DHZ*HZ
          ENDIF
#endif 
          
#if KEY_BLOCK==1
       ENDIF
#endif /*  BLOCK*/
       !
160    CONTINUE
10     CONTINUE
    ENDDO
    !
    NWARN=NWARN+NWARNX
    IF(NWARN.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,170) NWARN
170 FORMAT(' TOTAL OF',I6,' WARNINGS FROM EPHI_TABLE')
    !
    RETURN
  END SUBROUTINE ephi_table
  
  !----------------------------------------------------------------------
  !          READ Angle Tables
  !----------------------------------------------------------------------
  subroutine read_angle_tables(unit)
    use comand   
    use exfunc
    use stream
    use string
    use memory
    use rtf,only:atct,natct
    use psf
    use number,only:one
#if KEY_PARALLEL==1
    use parallel
#endif 
    integer,intent(in) :: unit
    character(len=MXCMSZ) :: comly2
    integer comle2,mxcms2,tbl_len,stat,n_ang_tbl
    integer ida,idb,idc,i,ipc0,ipc,itable
    logical eof,prnflg
    character(len=4) :: blank='    ',wrd
    real(chm_real) t,d,v
    integer nline

    langle_tables=.true.

    !------- Figure out how many tables we need -----------------
!    angtype_lst_index=0

    if (iolev > 0) then
       write(outu,'("   BTABLE> Calculating number of angle tables")')
       !----- Count how many angle tables there are for allocation purposes ------
       nline=0
       ntabtp=0
       num_ang_tbl=0
       read(unit,"(a)",iostat=stat)comlyn
       do while (stat == 0)
          nline=nline+1
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if(wrd == 'ANGL') then
             num_ang_tbl = num_ang_tbl+1
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             write(outu,'("   BTABLE>angle>cnttbl> types: ",a)')comlyn(1:comlen)
          end if
          read(unit,"(a)",iostat=stat)comlyn
       end do
       write(outu,'("   BTABLE> number of angle tables to read: ",2i6)')num_ang_tbl,nline
    endif

#if KEY_PARALLEL==1
    call psnd4(num_ang_tbl,1)
#endif 

    !-------- Allocate structure for angle tables --------------------
    allocate(angle_tbl(num_ang_tbl))
    if (iolev > 0) write(outu,'("   BTABLE> Allocating angle tables:",i4)')num_ang_tbl  

    !----- identify each table with an angle type --------
    if (iolev > 0) then
       rewind(unit)
       ntabtp=0
       n_ang_tbl=0
       read(unit,"(a)",iostat=stat)comlyn
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if(wrd == 'ANGL') then
             n_ang_tbl = n_ang_tbl+1
             if(n_ang_tbl >num_ang_tbl)then
                call wrndie(-3,"read_angle_tables<eintern_table.src>", &
                     "Not enough angle tables allocated")
                return
             endif
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             call fill_atypes
             write(outu,'("   BTABLE>angle> types: ",3i6)') ida,idb,idc
             angle_tbl(n_ang_tbl)%ida=ida
             angle_tbl(n_ang_tbl)%idb=idb
             angle_tbl(n_ang_tbl)%idc=idc
          end if
          read(unit,"(a)",iostat=stat)comlyn
       end do
       write(outu,'("   BTABLE> number of angle tables filled: ",i6)')num_ang_tbl

       !-------- Read each table ---------------------------------------
       write(outu,'("   BTABLE> Reading the angle tables")')
       rewind(unit)
       read(unit,"(a)",iostat=stat)comlyn
       
       ! itable   is the angle table number
       !          There is no sane way of calculating the table id so we will just search for
       !          a match of types each time we need a table.
       
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if (wrd.eq.'ANGL') then
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             !debug write(outu,'("   BTABLE>> types: ",a)')comlyn(1:comlen)
             call fill_atypes  ! determines ida, idb, and idc types
             print *,"Table for atom types",ida,idb,idc
             !--- figure out which table to fill -----
             itable=find_angle_table(ida,idb,idc)
             print *,"itable = ",itable
             read(unit,"(a)",iostat=stat)comlyn
             print *,"ReadAngleTables stat is:",stat
             comlen = len_trim(comlyn)
             tbl_len=nexti(comlyn,comlen)
             !dbg write(outu,'("   BTABLE>> table len ",i6)')tbl_len
             
             angle_tbl(itable)%len=tbl_len
             call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%t", &
                  tbl_len,crl=angle_tbl(itable)%t)
             call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%d", &
                  tbl_len,crl=angle_tbl(itable)%d)
             call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%v", &
                  tbl_len,crl=angle_tbl(itable)%v)
             do i=1,tbl_len
                read(unit,*)angle_tbl(itable)%t(i),angle_tbl(itable)%d(i),&
                     angle_tbl(itable)%v(i)
             enddo
             angle_tbl(itable)%tmin  = angle_tbl(itable)%t(1)
             angle_tbl(itable)%tmax  = angle_tbl(itable)%t(tbl_len)
             angle_tbl(itable)%dt    = (angle_tbl(itable)%t(tbl_len)-&
                  angle_tbl(itable)%t(1))/real(tbl_len-1)
             angle_tbl(itable)%dtinv = one/angle_tbl(itable)%dt
          endif
          read(unit,"(a)",iostat=stat)comlyn
       enddo
       rewind(unit)
       
    endif

#if KEY_PARALLEL==1
    call psnd4(n_ang_tbl,1)
    do itable=1,n_ang_tbl
       call psnd4(angle_tbl(itable)%len,1)
       if (iolev <= 0) then
          call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%t", &
               angle_tbl(itable)%len,crl=angle_tbl(itable)%t)
          call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%d", &
               angle_tbl(itable)%len,crl=angle_tbl(itable)%d)
          call chmalloc("eintern_table.src","read_angle_tables","angle_tbl(itable)%v", &
               angle_tbl(itable)%len,crl=angle_tbl(itable)%v)
       endif
       call psnd8(angle_tbl(itable)%t,angle_tbl(itable)%len)
       call psnd8(angle_tbl(itable)%d,angle_tbl(itable)%len)
       call psnd8(angle_tbl(itable)%v,angle_tbl(itable)%len)
       call psnd8(angle_tbl(itable)%tmin,1)
       call psnd8(angle_tbl(itable)%tmax,1)
       call psnd8(angle_tbl(itable)%dt,1)
       call psnd8(angle_tbl(itable)%dtinv,1)
       call psnd4(angle_tbl(itable)%ida,1)
       call psnd4(angle_tbl(itable)%idb,1)
       call psnd4(angle_tbl(itable)%idc,1)
    enddo
#endif 

    return

  contains
    subroutine fill_atypes
      character(len=6) :: name,name2,name3
      character(len=1) :: blank=' '
      integer i,l

      !--- get first type ---
      l=1
      i=1
      name(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      ida = srchws(atct,natct,name)

      !--- get second type ---
      l=1
      i=i+1
      name2(1:6)='      '

      do while(comlyn(i:i) /= blank)
         name2(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      idb = srchws(atct,natct,name2)

      !--- get third type ---
      l=1
      i=i+1
      name3(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name3(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      idc = srchws(atct,natct,name3)

    end subroutine fill_atypes

  end subroutine read_angle_tables

  !----------------------------------------------------------------------
  !          Find Angle Table
  !----------------------------------------------------------------------
  integer function find_angle_table(ida,idb,idc)
    use stream,only:outu
    character(len=6) :: name,name2,name3
    character(len=1) :: blank=' '
    integer i,l,ida,idb,idc

    do i=1,num_ang_tbl
       if(ida == angle_tbl(i)%ida ) then
          if(idb == angle_tbl(i)%idb .and. idc == angle_tbl(i)%idc ) then
             find_angle_table=i
             return
          endif
!       elseif(idc == angle_tbl(i)%ida ) then
!          if(idb == angle_tbl(i)%idb .and. ida == angle_tbl(i)%idc ) then
!             find_angle_table=i
!             return
!          endif
       endif
    enddo
    write(outu,'("Angle lookup for types:",3i6)')ida,idb,idc
    call wrndie(-3,"read_angle_tables<eintern_table.src>", &
         "Cannot find table entry in file for atoms, something wrong with code")
    return
  end function  find_angle_table


  !----------------------------------------------------------------------
  !          READ Bond Tables
  !----------------------------------------------------------------------
  subroutine read_bond_tables(unit,natc)
    use comand   
    use exfunc
    use stream
    use string
    use memory
    use rtf,only:atct,natct
    use number,only:one
    use psf,only:nbond,ib,jb,iac
#if KEY_PARALLEL==1
    use parallel
#endif 
    integer,intent(in) :: unit,natc
    character(len=MXCMSZ) :: comly2
    integer comle2,mxcms2,tbl_len,stat
    integer iaa,ibb
    integer ioff(maxatc*maxatc+1),i,ipc0,ipc,ii,jj
    logical eof,prnflg
    character(len=4) :: blank='    ',wrd
    real(chm_real) t,d,v
    integer status

    lbond_tables=.true.

    if (iolev > 0) then
       write(outu,'("   BTABLE> Calculating number of bond tables")')
       !----- Count how many bond tables there are for allocation purposes ------
       ntabtp=0
       num_bnd_tbl=0
       read(unit,"(a)",iostat=stat)comlyn
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if(wrd == 'BOND') then
             num_bnd_tbl=num_bnd_tbl+1
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             write(outu,'("   BTABLE>> types: ",a)')comlyn(1:comlen)
             ! call fill_types
          end if
          read(unit,"(a)",iostat=stat)comlyn
       end do
       write(outu,'("   BTABLE> number of bond tables to read: ",i6)')num_bnd_tbl
    endif

#if KEY_PARALLEL==1
    call psnd4(num_bnd_tbl,1)
#endif 

    !-------- Allocate structure for bond tables --------------------
    allocate(bond_tbl(num_bnd_tbl))
    if (iolev > 0) write(outu,'("   BTABLE> Allocating bond tables:",i4)')num_bnd_tbl

    !-------- Load in offsets for different types -------------------
    do i=1,maxatc
       ioff(i)=(i*(i-1))/2
    enddo

    if (iolev > 0) then
       !-------- Read each table ---------------------------------------
       write(outu,'("   BTABLE> Reading the bond tables")')
       rewind(unit)
       read(unit,"(a)",iostat=stat)comlyn
    endif

    call chmalloc("eintern_table.src","read_bond_tables","bndtype_lst",nbond,intg=bndtype_lst)
    bndtype_lst = 0
       
    ! ipc   is the bond table number
    ! ipc0  will be the bond type based on atom types
!!! bndtype_lst maps bond type to bond table number
    ipc=0
    if (iolev > 0) then
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if (wrd.eq.'BOND') then
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             call fill_types  ! determines ia and ib types
             ipc=ipc+1
             do i=1,nbond
                if (iaa == iac(ib(i)) .and. ibb == iac(jb(i))) then
                   bndtype_lst(i) = ipc
                endif
             enddo
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             tbl_len=nexti(comlyn,comlen)
             bond_tbl(ipc)%len=tbl_len
             call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(ipc)%t", &
                  tbl_len,crl=bond_tbl(ipc)%t)
             call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(ipc)%d", &
                  tbl_len,crl=bond_tbl(ipc)%d)
             call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(ipc)%v", &
                  tbl_len,crl=bond_tbl(ipc)%v)
             do i=1,tbl_len
                read(unit,*)bond_tbl(ipc)%t(i),bond_tbl(ipc)%d(i),bond_tbl(ipc)%v(i)
             enddo
             bond_tbl(ipc)%tmin  = bond_tbl(ipc)%t(1)
             bond_tbl(ipc)%tmax  = bond_tbl(ipc)%t(tbl_len)
             bond_tbl(ipc)%dt    = (bond_tbl(ipc)%t(tbl_len)-bond_tbl(ipc)%t(1))/real(tbl_len-1)
             bond_tbl(ipc)%dtinv = one/bond_tbl(ipc)%dt
          endif
          read(unit,"(a)",iostat=stat)comlyn
       enddo
    endif

#if KEY_PARALLEL==1
    call psnd4(bndtype_lst,nbond)
    call psnd4(ipc,1)
    do i=1,ipc
       call psnd4(bond_tbl(i)%len,1)
       if (iolev <= 0) then
          call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(i)%t", &
               bond_tbl(i)%len,crl=bond_tbl(i)%t)
          call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(i)%d", &
               bond_tbl(i)%len,crl=bond_tbl(i)%d)
          call chmalloc("eintern_table.src","read_bond_tables","bond_tbl(i)%v", &
               bond_tbl(i)%len,crl=bond_tbl(i)%v)
       endif
       call psnd8(bond_tbl(i)%t,bond_tbl(i)%len)
       call psnd8(bond_tbl(i)%d,bond_tbl(i)%len)
       call psnd8(bond_tbl(i)%v,bond_tbl(i)%len)
       call psnd8(bond_tbl(i)%tmin,1)
       call psnd8(bond_tbl(i)%tmax,1)
       call psnd8(bond_tbl(i)%dt,1)
       call psnd8(bond_tbl(i)%dtinv,1)
    enddo
#endif 

    if (iolev > 0) rewind(unit)

    do i=1,nbond
       if (bndtype_lst(i) == 0) then
          call wrndie(-3,'read_bond_tables<eintern_table>','some bndtype_lst elements missing')
       endif
    enddo

    return

  contains
    subroutine fill_types
      character(len=6) :: name,name2
      character(len=1) :: blank=' '
      integer i,l
      !--- get first type ---
      l=1
      i=1
      name(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      iaa = srchws(atct,natct,name)

      !--- get second type ---
      l=1
      i=i+1
      name2(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name2(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      ibb = srchws(atct,natct,name2)

      return
    end subroutine fill_types

  end subroutine read_bond_tables



!----------------------------------------------------------------------
!          READ Dihedral Tables
!----------------------------------------------------------------------
  subroutine read_dihedral_tables(unit)
    use comand
    use exfunc
    use stream
    use string
    use memory
    use rtf,only:atct,natct
    use psf
    use number,only:one
#if KEY_PARALLEL==1
    use parallel
#endif 
    integer,intent(in) :: unit
    character(len=MXCMSZ) :: comly2
    integer comle2,mxcms2,tbl_len,stat,n_dih_tbl
    integer ida,idb,idc,idd,i,ipc0,ipc,itable
    logical eof,prnflg
    character(len=4) :: blank='    ',wrd
    real(chm_real) t,d,v
    integer nline

    ldihedral_tables=.true.

    !------- Figure out how many tables we need -----------------
!    dihtype_lst_index=0

    if (iolev > 0) then
       write(outu,'("   BTABLE> Calculating number of dihedral tables")')
       !----- Count how many dihedral tables there are for allocation purposes ------
       nline=0
       ntabtp=0
       num_dih_tbl=0
       read(unit,"(a)",iostat=stat)comlyn
       do while (stat == 0)
          nline=nline+1
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if(wrd == 'DIHE') then
             num_dih_tbl = num_dih_tbl+1
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             write(outu,'("   BTABLE>dihedral>cnttbl> types: ",a)')comlyn(1:comlen)
          end if
          read(unit,"(a)",iostat=stat)comlyn
       end do
       write(outu,'("   BTABLE> number of dihedral tables to read: ",2i6)')num_dih_tbl,nline
    endif

#if KEY_PARALLEL==1
    call psnd4(num_dih_tbl,1)
#endif 

    !-------- Allocate structure for dihedral tables --------------------
    allocate(dihedral_tbl(num_dih_tbl))
    if (iolev > 0) write(outu,'("   BTABLE> Allocating dihedral tables:",i4)')num_dih_tbl

    if (iolev > 0) then
       !----- identify each table with a dihedral type --------
       rewind(unit)
       ntabtp=0
       n_dih_tbl=0
       read(unit,"(a)",iostat=stat)comlyn
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if(wrd == 'DIHE') then
             n_dih_tbl = n_dih_tbl+1
             if(n_dih_tbl >num_dih_tbl)then
                call wrndie(-3,"read_dihedral_tables<eintern_table.src>", &
                     "Not enough dihedral tables allocated")
                return
             endif
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             call fill_atypes
             write(outu,'("   BTABLE>dihedral> types: ",4i6)') ida,idb,idc,idd
             dihedral_tbl(n_dih_tbl)%ida=ida
             dihedral_tbl(n_dih_tbl)%idb=idb
             dihedral_tbl(n_dih_tbl)%idc=idc
             dihedral_tbl(n_dih_tbl)%idd=idd
          end if
          read(unit,"(a)",iostat=stat)comlyn
       end do
       write(outu,'("   BTABLE> number of dihedral tables filled: ",i6)')num_dih_tbl

       !-------- Read each table ---------------------------------------
       write(outu,'("   BTABLE> Reading the dihedral tables")')
       rewind(unit)
       read(unit,"(a)",iostat=stat)comlyn
       
       ! itable   is the dihedral table number
       !          There is no sane way of calculating the table id so we will just search for
       !          a match of types each time we need a table.
       
       do while (stat == 0)
          comlen = len_trim(comlyn)
          wrd=nexta4(comlyn,comlen)
          if (wrd.eq.'DIHE') then
             read(unit,"(a)",iostat=stat)comlyn
             comlen = len_trim(comlyn)
             call fill_atypes  ! determines ida, idb, and idc types
             print *,"Table for atom types",ida,idb,idc,idd
             !--- figure out which table to fill -----
             itable=find_dihedral_table(ida,idb,idc,idd)
             
             read(unit,"(a)",iostat=stat)comlyn
             print *,"ReadDihedralTables stat is:",stat
             comlen = len_trim(comlyn)
             tbl_len=nexti(comlyn,comlen)
             dihedral_tbl(itable)%len=tbl_len
             call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%t", &
                  tbl_len,crl=dihedral_tbl(itable)%t)
             call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%d", &
                  tbl_len,crl=dihedral_tbl(itable)%d)
             call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%v", &
                  tbl_len,crl=dihedral_tbl(itable)%v)
             do i=1,tbl_len
                read(unit,*)dihedral_tbl(itable)%t(i),dihedral_tbl(itable)%d(i),&
                     dihedral_tbl(itable)%v(i)
             enddo
             dihedral_tbl(itable)%tmin  = dihedral_tbl(itable)%t(1)
             dihedral_tbl(itable)%tmax  = dihedral_tbl(itable)%t(tbl_len)
             dihedral_tbl(itable)%dt    = (dihedral_tbl(itable)%t(tbl_len)-&
                  dihedral_tbl(itable)%t(1))/real(tbl_len-1)
             dihedral_tbl(itable)%dtinv = one/dihedral_tbl(itable)%dt
          endif
          read(unit,"(a)",iostat=stat)comlyn
       enddo
       rewind(unit)
    endif

#if KEY_PARALLEL==1
    call psnd4(n_dih_tbl,1)
    do itable=1,n_dih_tbl
       call psnd4(dihedral_tbl(itable)%len,1)
       if (iolev <= 0) then
          call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%t", &
               dihedral_tbl(itable)%len,crl=dihedral_tbl(itable)%t)
          call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%d", &
               dihedral_tbl(itable)%len,crl=dihedral_tbl(itable)%d)
          call chmalloc("eintern_table.src","read_dihedral_tables","dihedral_tbl(itable)%v", &
               dihedral_tbl(itable)%len,crl=dihedral_tbl(itable)%v)
       endif
       call psnd8(dihedral_tbl(itable)%t,dihedral_tbl(itable)%len)
       call psnd8(dihedral_tbl(itable)%d,dihedral_tbl(itable)%len)
       call psnd8(dihedral_tbl(itable)%v,dihedral_tbl(itable)%len)
       call psnd8(dihedral_tbl(itable)%tmin,1)
       call psnd8(dihedral_tbl(itable)%tmax,1)
       call psnd8(dihedral_tbl(itable)%dt,1)
       call psnd8(dihedral_tbl(itable)%dtinv,1)
       call psnd4(dihedral_tbl(itable)%ida,1)
       call psnd4(dihedral_tbl(itable)%idb,1)
       call psnd4(dihedral_tbl(itable)%idc,1)
       call psnd4(dihedral_tbl(itable)%idd,1)
    enddo
#endif 

    return

  contains
    subroutine fill_atypes
      character(len=6) :: name,name2,name3,name4
      character(len=1) :: blank=' '
      integer i,l

      !--- get first type ---
      l=1
      i=1
      name(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      ida = srchws(atct,natct,name)

      !--- get second type ---
      l=1
      i=i+1
      name2(1:6)='      '

      do while(comlyn(i:i) /= blank)
         name2(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      idb = srchws(atct,natct,name2)

      !--- get third type ---
      l=1
      i=i+1
      name3(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name3(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      idc = srchws(atct,natct,name3)

      !--- get fourth type ---
      l=1
      i=i+1
      name4(1:6)='      '
      do while(comlyn(i:i) /= blank)
         name4(l:l)=comlyn(i:i)
         l=l+1
         i=i+1
      enddo
      idd = srchws(atct,natct,name4)

    end subroutine fill_atypes

  end subroutine read_dihedral_tables

  !----------------------------------------------------------------------
  !          Find Dihedral Table
  !----------------------------------------------------------------------
  integer function find_dihedral_table(ida,idb,idc,idd)
    use stream,only:outu
    character(len=6) :: name,name2,name3,name4
    character(len=1) :: blank=' '
    integer i,l,ida,idb,idc,idd

    do i=1,num_dih_tbl
       if(ida == dihedral_tbl(i)%ida ) then
          if(idb == dihedral_tbl(i)%idb .and. idc == dihedral_tbl(i)%idc &
               .and. idd == dihedral_tbl(i)%idd) then
             find_dihedral_table=i
             return
          endif
       elseif(idd == dihedral_tbl(i)%ida ) then
          if(idc == dihedral_tbl(i)%idb .and. idb == dihedral_tbl(i)%idc &
               .and. ida == dihedral_tbl(i)%idd) then
             find_dihedral_table=i
             return
          endif
       endif
    enddo
    write(outu,'("Dihedral lookup for types:",4i6)')ida,idb,idc,idd
    call wrndie(-3,"read_dihedral_tables<eintern_table.src>", &
         "Cannot find table entry in file for atoms, something wrong with code")
    return
  end function  find_dihedral_table

end module bond_tables

