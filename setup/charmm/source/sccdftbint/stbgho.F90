#if KEY_SCCDFTB==1 /*stbgho_main*/
!-----------------------------------------------------------------------
SUBROUTINE HBDEF(X,Y,Z,BT,BTM,DBTMMM,NATVB,IATVB,JATVB,KATVB)
  !-----------------------------------------------------------------------
  !      DEFINES TRANSFORMATION MATRIX
  !
  use consta
  implicit none
  real(chm_real) X(*),Y(*),Z(*),BT(*),BTM(*),DBTMMM(3,3,*)
  INTEGER NATVB,IATVB(*),JATVB(3,*),KATVB(*)
  !
  !   local variables
  INTEGER I,II,JJ,NI
  real(chm_real) A(3),B(3),C(3),AB(3),AC(3),P(3),T(3)
  real(chm_real)  ZERO,ONE,TWO,THREE
  DATA    ZERO,ONE,TWO,THREE/0.00,1.00,2.00,3.00/
  SAVE    ZERO,ONE,TWO,THREE
  !
  NI = 1
  DO I = 1,NATVB
     !
     !      EACH QM-LINK ATOM IS CONNECTED TO 1 QM ATOM and 3 MM ATOMS
     II = IATVB(I)
     !      MM ATOMS
     JJ = JATVB(1,I)
     A(1) = X(JJ)-X(II)
     A(2) = Y(JJ)-Y(II)
     A(3) = Z(JJ)-Z(II)
     JJ = JATVB(2,I)
     B(1) = X(JJ)-X(II)
     B(2) = Y(JJ)-Y(II)
     B(3) = Z(JJ)-Z(II)
     JJ = JATVB(3,I)
     C(1) = X(JJ)-X(II)
     C(2) = Y(JJ)-Y(II)
     C(3) = Z(JJ)-Z(II)
     !      QM ATOM
     JJ = KATVB(I)
     T(1) = X(JJ)-X(II)
     T(2) = Y(JJ)-Y(II)
     T(3) = Z(JJ)-Z(II)
     P(1) = 0.0
     P(2) = 0.0
     P(3) = 0.0
     CALL HBDRIV(A,B,C,T,P,BT(NI),BTM(NI),DBTMMM(1,1,NI))

     IF(P(1).NE.0.0) CALL WRNDIE(-5,'HBDEF>', &
          'HYBRID ORBITAL ILLDEFINED.')
     NI = NI+16
  ENDDO
  !
  RETURN
END SUBROUTINE HBDEF


!-----------------------------------------------------------------
SUBROUTINE ACB(A,B,C,S)
  !-----------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  real(chm_real) A(3),B(3),C(3)
  !
  real(chm_real) X,Y,Z,S
  !
  X = A(2)*B(3)-B(2)*A(3)
  Y = B(1)*A(3)-A(1)*B(3)
  Z = A(1)*B(2)-B(1)*A(2)
  S = SQRT(X*X+Y*Y+Z*Z)
  C(1) = X
  C(2) = Y
  C(3) = Z
  RETURN
END SUBROUTINE ACB

!----------------------------------------------------------------
SUBROUTINE VBFTN(F1,F1VB,X,NDIM)
  !----------------------------------------------------------------
  use chm_kinds
  implicit none

  !
  INTEGER NDIM,L1,I,J,K,L,L2
  real(chm_real) F1(*),F1VB(*),X(NDIM,NDIM)
  !
  real(chm_real) FAC
  !
  L1 = 0
  DO I = 1,NDIM
     DO J = 1,I
        L1 = L1+1
        L2 = 0
        F1VB(L1) = 0.0
        DO K = 1,NDIM
           DO L = 1,K-1
              L2 = L2+1
              FAC = X(K,I)*X(L,J)+X(K,J)*X(L,I)
              F1VB(L1) = F1VB(L1)+F1(L2)*FAC
           ENDDO
           L2 = L2+1
           FAC = X(K,I)*X(K,J)
           F1VB(L1) = F1VB(L1)+F1(L2)*FAC
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE VBFTN

!--------------------------------------------------------------------
SUBROUTINE HBDRIV(A,B,C,D,O,BT,BTM,DBTMMM)
  !--------------------------------------------------------------------
  use chm_kinds
  implicit none

  !
  INTEGER I,J,K,N,IJ
  real(chm_real) A(3),B(3),C(3),D(3),O(3), &
       BT(4,4),BTM(4,4),DBTMMM(3,3,1)
  !
  !   Local variables.
  real(chm_real) AA(3),BB(3),CC(3),U(3),V(3),T(4,4)
  real(chm_real) X(3),Y(3),Z(3),TETH(4,4)
  real(chm_real) dthma(3,4,4),dthmb(3,4,4),dthmc(3,4,4)
  real(chm_real) dxa(3,3),dxb(3,3),dxc(3,3), &
       dya(3,3),dyb(3,3),dyc(3,3), &
       dza(3,3),dzb(3,3),dzc(3,3),daa(3,3), &
       dd1a(3),dd1b(3),dd1c(3)
  real(chm_real) DBTMM(4,4),DBTMMB(4,4),DBTMMC(4,4)
  real(chm_real) GRADA(4,4,3),GRADB(4,4,3),GRADC(4,4,3)
  real(chm_real) drxa(3),drxb(3),drxc(3),drza(3),drzb(3),drzc(3)
  real(chm_real) xx(3),ab(3),bc(3),ca(3)
  real(chm_real) dcsa(3),dcsb(3),dcsc(3),dcpa(3),dcpb(3),dcpc(3)
  real(chm_real) AF(3,3)
  INTEGER IR(3,3)
  !
  real(chm_real) DD,RA,RB,RC,CS,PFAC,RX,RY,RZ,RA2,RB2,RC2,CS2
  real(chm_real) THRFAC,RBA,RAC,RBC,D0
  !
  SAVE AF,IR
  DATA AF/0.0,1.0,-1.0,-1.0,0.0,1.0,1.0,-1.0,0.0/
  DATA IR/0,3,2,3,0,1,2,1,0/
  !
  !     data teth/0.50,0.866,0.0,0.0,0.5,-0.2887,0.8165,0.0,
  !    & 0.5,-0.2887,-0.4082,0.7071, 0.5,-0.2887,-0.4082,-0.7071/
  !
  ra2 = a(1)**2+a(2)**2+a(3)**2
  rb2 = b(1)**2+b(2)**2+b(3)**2
  rc2 = c(1)**2+c(2)**2+c(3)**2
  ra = sqrt(ra2)
  rb = sqrt(rb2)
  rc = sqrt(rc2)

  ! do some initialization, bug fix 0601PJ07
  do i=1,4
     do j=1,4
        do k=1,3
           grada(i,j,k)=0.0D0
           gradb(i,j,k)=0.0D0
           gradc(i,j,k)=0.0D0
        end do
     end do
  end do

  do i = 1,3
     aa(i) = a(i)/ra
     bb(i) = b(i)/rb
     cc(i) = c(i)/rc
     u(i) = bb(i) - aa(i)
     v(i) = cc(i) - aa(i)
  enddo

  CALL ACB(u,v,x,rx)

  d0 = (aa(1)*x(1)+aa(2)*x(2)+aa(3)*x(3))/rx
  dd  = abs(d0)
  pfac = 1.0
  if(D0.GT.0.0) pfac = -1.0
  !
  !      tetrahedarl hybrid orbitals:
  cs = sqrt(dd/(1.0+dd))

  !JIALI cs = 0.5    !perfect tetrahedral geometry.

  cs2 = sqrt((1.0-cs**2)/3.0)

  do i = 1,4
     do j = 1,4
        teth(j,i) = 0.0
     enddo
     teth(1,i) = cs2
  enddo
  teth(1,1) = cs

  teth(2,1) = sqrt(1.0d+00-teth(1,1)**2)

  teth(2,2) = -teth(1,1)*teth(1,2)/teth(2,1)
  teth(3,2) = sqrt(1.d+00-teth(1,2)**2-teth(2,2)**2)

  teth(2,3) = -teth(1,1)*teth(1,3)/teth(2,1)
  teth(3,3) = -(teth(1,2)*teth(1,3)+teth(2,2)*teth(2,3))/teth(3,2)
  teth(4,3) = sqrt(1.d+00-teth(1,3)**2-teth(2,3)**2-teth(3,3)**2)

  teth(2,4) = -teth(1,1)*teth(1,4)/teth(2,1)
  teth(3,4) = -(teth(1,2)*teth(1,4)+teth(2,2)*teth(2,4))/teth(3,2)
  teth(4,4) = -(teth(1,3)*teth(1,4)+teth(2,3)*teth(2,4)+ &
       teth(3,3)*teth(3,4))/teth(4,3)
  !
  !
  do i = 1,3
     x(i) = pfac*x(i)
  enddo
  call acb(x,aa,z,rz)
  call acb(z,x,y,ry)
  do i = 1,4
     t(1,i) = 0.0
     t(i,1) = 0.0
  enddo
  t(1,1) = 1.0
  do j = 1,3
     t(j+1,2) = x(j)/rx
     t(j+1,3) = y(j)/ry
     t(j+1,4) = z(j)/rz
  enddo

  call acb(bb,cc,bc,rbc)
  call acb(cc,aa,ca,rac)
  call acb(aa,bb,ab,rba)

  !     dai
  do i = 1,3
     !        dxj
     do j = 1,3
        !        dxj/dai
        dxa(j,i) = -aa(i)*(ab(j)+ca(j))/ra
        dxb(j,i) = -bb(i)*(bc(j)+ab(j))/rb
        dxc(j,i) = -cc(i)*(ca(j)+bc(j))/rc
        if(j.ne.i) then
           dxa(j,i) = dxa(j,i)+af(j,i)*(cc(ir(j,i))-bb(ir(j,i)))/ra
           dxb(j,i) = dxb(j,i)+af(j,i)*(aa(ir(j,i))-cc(ir(j,i)))/rb
           dxc(j,i) = dxc(j,i)+af(j,i)*(bb(ir(j,i))-aa(ir(j,i)))/rc
        endif
     enddo
  enddo
  if(pfac.eq.1.0) then
     do i = 1,3
        do j= 1,3
           dxa(j,i) = -dxa(j,i)
           dxb(j,i) = -dxb(j,i)
           dxc(j,i) = -dxc(j,i)
        enddo
     enddo
  endif
  !
  !      (Rx^2)'
  do i = 1,3
     drxa(i) = 2.0*(x(1)*dxa(1,i)+x(2)*dxa(2,i)+x(3)*dxa(3,i))
     drxb(i) = 2.0*(x(1)*dxb(1,i)+x(2)*dxb(2,i)+x(3)*dxb(3,i))
     drxc(i) = 2.0*(x(1)*dxc(1,i)+x(2)*dxc(2,i)+x(3)*dxc(3,i))
  enddo

  !     dxj/dmi
  do i = 1,3
     do j = 1,3
        grada(j+1,2,i) = dxa(j,i)/rx-0.5*x(j)*drxa(i)/(rx**3)
        gradb(j+1,2,i) = dxb(j,i)/rx-0.5*x(j)*drxb(i)/(rx**3)
        gradc(j+1,2,i) = dxc(j,i)/rx-0.5*x(j)*drxc(i)/(rx**3)
     enddo
  enddo
  !
  !      daaj/dai
  do i = 1,3
     do j = 1,3
        daa(j,i) = -aa(j)*aa(i)/ra
     enddo
     daa(i,i) = daa(i,i)+1.0/ra
  enddo

  do i = 1,3
     dza(1,i) = dxa(2,i)*aa(3)+x(2)*daa(3,i) &
          -daa(2,i)*x(3)-aa(2)*dxa(3,i)
     dza(2,i) = dxa(3,i)*aa(1)+x(3)*daa(1,i) &
          -daa(3,i)*x(1)-aa(3)*dxa(1,i)
     dza(3,i) = dxa(1,i)*aa(2)+x(1)*daa(2,i) &
          -daa(1,i)*x(2)-aa(1)*dxa(2,i)

     dzb(1,i) = dxb(2,i)*aa(3)-aa(2)*dxb(3,i)
     dzb(2,i) = dxb(3,i)*aa(1)-aa(3)*dxb(1,i)
     dzb(3,i) = dxb(1,i)*aa(2)-aa(1)*dxb(2,i)

     dzc(1,i) = dxc(2,i)*aa(3)-aa(2)*dxc(3,i)
     dzc(2,i) = dxc(3,i)*aa(1)-aa(3)*dxc(1,i)
     dzc(3,i) = dxc(1,i)*aa(2)-aa(1)*dxc(2,i)
  enddo

  !      (Rz^2)'
  do i = 1,3
     drza(i) = 2.0*(z(1)*dza(1,i)+z(2)*dza(2,i)+z(3)*dza(3,i))
     drzb(i) = 2.0*(z(1)*dzb(1,i)+z(2)*dzb(2,i)+z(3)*dzb(3,i))
     drzc(i) = 2.0*(z(1)*dzc(1,i)+z(2)*dzc(2,i)+z(3)*dzc(3,i))
  enddo
  !
  !      dzj/dmi
  !
  do i = 1,3
     do j = 1,3
        grada(j+1,4,i) = dza(j,i)/rz-0.5*z(j)*drza(i)/(rz**3)
        gradb(j+1,4,i) = dzb(j,i)/rz-0.5*z(j)*drzb(i)/(rz**3)
        gradc(j+1,4,i) = dzc(j,i)/rz-0.5*z(j)*drzc(i)/(rz**3)
     enddo
  enddo
  !
  do i = 1,3
     dya(1,i) = dza(2,i)*x(3)+z(2)*dxa(3,i)-dxa(2,i)*z(3)-x(2)*dza(3,i)
     dya(2,i) = dza(3,i)*x(1)+z(3)*dxa(1,i)-dxa(3,i)*z(1)-x(3)*dza(1,i)
     dya(3,i) = dza(1,i)*x(2)+z(1)*dxa(2,i)-dxa(1,i)*z(2)-x(1)*dza(2,i)
     dyb(1,i) = dzb(2,i)*x(3)+z(2)*dxb(3,i)-dxb(2,i)*z(3)-x(2)*dzb(3,i)
     dyb(2,i) = dzb(3,i)*x(1)+z(3)*dxb(1,i)-dxb(3,i)*z(1)-x(3)*dzb(1,i)
     dyb(3,i) = dzb(1,i)*x(2)+z(1)*dxb(2,i)-dxb(1,i)*z(2)-x(1)*dzb(2,i)
     dyc(1,i) = dzc(2,i)*x(3)+z(2)*dxc(3,i)-dxc(2,i)*z(3)-x(2)*dzc(3,i)
     dyc(2,i) = dzc(3,i)*x(1)+z(3)*dxc(1,i)-dxc(3,i)*z(1)-x(3)*dzc(1,i)
     dyc(3,i) = dzc(1,i)*x(2)+z(1)*dxc(2,i)-dxc(1,i)*z(2)-x(1)*dzc(2,i)
  enddo
  !
  !      dyj/dmi
  !
  do i = 1,3
     do j = 1,3
        grada(j+1,3,i) = (dya(j,i)-0.5*y(j)*(drza(i)/(rz**2)+ &
             drxa(i)/(rx**2)))/ry
        gradb(j+1,3,i) = (dyb(j,i)-0.5*y(j)*(drzb(i)/(rz**2)+ &
             drxb(i)/(rx**2)))/ry
        gradc(j+1,3,i) = (dyc(j,i)-0.5*y(j)*(drzc(i)/(rz**2)+ &
             drxc(i)/(rx**2)))/ry
     enddo
  enddo
  !
  !     d d/dmi
  !
  do i = 1,4
     do j = 1,4
        do k = 1,3
           dthma(k,j,i) = 0.0
           dthmb(k,j,i) = 0.0
           dthmc(k,j,i) = 0.0
        enddo
     enddo
  enddo
  do i = 1,3
     !MJF . Signs have been changed here!
     dd1a(i) = ( (dxa(1,i)*aa(1)+dxa(2,i)*aa(2)+dxa(3,i)*aa(3)- &
          daa(1,i)*x(1)-daa(2,i)*x(2)-daa(3,i)*x(3))/rx- &
          0.5*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxa(i)/(rx**3))
     dd1b(i) = ((dxb(1,i)*aa(1)+dxb(2,i)*aa(2)+dxb(3,i)*aa(3))/rx- &
          0.5*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxb(i)/(rx**3))
     dd1c(i) = ((dxc(1,i)*aa(1)+dxc(2,i)*aa(2)+dxc(3,i)*aa(3))/rx- &
          0.5*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxc(i)/(rx**3))
  enddo

  do i = 1,3
     dcsa(i) = -0.5*(1.0/teth(1,1)-teth(1,1))*(teth(2,1)**2)*dd1a(i)
     dcsb(i) = -0.5*(1.0/teth(1,1)-teth(1,1))*(teth(2,1)**2)*dd1b(i)
     dcsc(i) = -0.5*(1.0/teth(1,1)-teth(1,1))*(teth(2,1)**2)*dd1c(i)
     dcpa(i) = 0.5*dd1a(i)*teth(2,1)**3
     dcpb(i) = 0.5*dd1b(i)*teth(2,1)**3
     dcpc(i) = 0.5*dd1c(i)*teth(2,1)**3
  enddo
  !
  thrfac = 1.0/sqrt(3.0d+00)
  do i = 1,3
     dthma(i,1,1) = dcsa(i)
     dthma(i,2,1) = dcpa(i)
     dthma(i,1,2) = dcpa(i)*thrfac
     dthma(i,1,3) = dcpa(i)*thrfac
     dthma(i,1,4) = dcpa(i)*thrfac
     dthma(i,2,2) = -dcsa(i)*thrfac
     dthma(i,2,3) = -dcsa(i)*thrfac
     dthma(i,2,4) = -dcsa(i)*thrfac
     !
     dthmb(i,1,1) = dcsb(i)
     dthmb(i,2,1) = dcpb(i)
     dthmb(i,1,2) = dcpb(i)*thrfac
     dthmb(i,1,3) = dcpb(i)*thrfac
     dthmb(i,1,4) = dcpb(i)*thrfac
     dthmb(i,2,2) = -dcsb(i)*thrfac
     dthmb(i,2,3) = -dcsb(i)*thrfac
     dthmb(i,2,4) = -dcsb(i)*thrfac
     !C
     dthmc(i,1,1) = dcsc(i)
     dthmc(i,2,1) = dcpc(i)
     dthmc(i,1,2) = dcpc(i)*thrfac
     dthmc(i,1,3) = dcpc(i)*thrfac
     dthmc(i,1,4) = dcpc(i)*thrfac
     dthmc(i,2,2) = -dcsc(i)*thrfac
     dthmc(i,2,3) = -dcsc(i)*thrfac
     dthmc(i,2,4) = -dcsc(i)*thrfac
  enddo
  !
  !      COLLECT
  !
  DO J = 1,4
     DO I = 1,4
        BT(I,J) = 0.0
        DO K = 1,4
           BT(I,J) = BT(I,J)+T(I,K)*TETH(K,J)
        ENDDO
        BTM(J,I) = BT(I,J)
     ENDDO
  ENDDO
  !
  !      Derivatives
  !
  DO N = 1,3
     DO J = 1,4
        DO I = 1,4
           dbtmm(i,j) = 0.0
           dbtmmb(i,j) = 0.0
           dbtmmc(i,j) = 0.0
           DO K = 1,4
              !MJF . Lines have been uncommented!
              dbtmm(i,j) = dbtmm(i,j)+grada(i,k,n)*teth(k,j) &
                   +t(i,k)*dthma(n,k,j)
              dbtmmb(i,j) = dbtmmb(i,j)+gradb(i,k,n)*teth(k,j) &
                   +t(i,k)*dthmb(n,k,j)
              dbtmmc(i,j) = dbtmmc(i,j)+gradc(i,k,n)*teth(k,j) &
                   +t(i,k)*dthmc(n,k,j)
           ENDDO
        ENDDO
     ENDDO
     ij = 0
     do i = 1,4
        do j = 1,4
           ij = ij+1
           dbtmmm(n,1,ij) = dbtmm(i,j)
           dbtmmm(n,2,ij) = dbtmmb(i,j)
           dbtmmm(n,3,ij) = dbtmmc(i,j)
        enddo
     enddo
  ENDDO
  return
END SUBROUTINE HBDRIV

!------------------------------------------------------------------------
SUBROUTINE DQLINK(DX,DY,DZ,X,Y,Z,PHB,F)
  !------------------------------------------------------------------------
  !   Computes the derivatives of the density matrix as a result of
  !   the transformation from hybrid orbitals to atomic orbitals basis.
  !   Updates nucleus force arrays.
  !
  !   J. Gao, P. Amara, C. Alhambra, M. J. Field
  !   J. Phys. Chem. A, 102, 4714 (1998).
  !
  !   The parameter list in the interface are changed, I added PHB
  !   into the list, so that the code can be reused for the derivative
  !   correction term for the density force (dW/dq)*S, where W is the
  !   energy weighted density matrix in AO N+4 basis, and W = T'*WtHB*T,
  !   and S is the overlap matrix in AO (N+4).
  !   Recall that the transformation matrix depends on the position
  !   of the GHO boundary atom and MMs directly linked to the GHO.
  !
  !   The similarity of the this term to the (dP/dq)*F correction ensured
  !   the reuse of this segment of code for both purposes.
  !
  !   To reuse the code, the following mapping is important:
  !      PHB --> WHB (energy weighted density matrix in HB N+4)
  !        F --> S   (overlap matrix in AO N+4)
  !   
  !   Note: PHO is changed to PHB to avoid name conflict with 
  !         common block varibles, 1020PJ03
  !
  use dimens_fcm
  use sizes
  use stream
  use stbgho
  use consta
  use psf
  implicit none

  !
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*),F(*),PHB(*)
  !
  ! gradient conversion factor compatible with SCCDFTB
  ! from hartree to kcal/mol.
  !
  real(chm_real),PARAMETER :: CONFAC=TOKCAL
  !  Local variables
  INTEGER I, J, K, L, M, N, II, JJ, LL, MM, I1, L1, M1, IJ
  INTEGER KK1,KK2,KK3,II2,JJ2,N16,N16I1,N16J
  INTEGER I1DIAG(4),NAOS,LINAO
  real(chm_real) XDTA(4,4),YDTA(4,4),ZDTA(4,4),XDTB(4,4),YDTB(4,4)
  real(chm_real) XDTC(4,4),YDTC(4,4),ZDTC(4,4),ZDTB(4,4),TINV(4,4)
  real(chm_real) DMMXA,DMMYA,DMMZA,DMMXB,DMMYB,DMMZB,DMMXC,DMMYC,DMMZC
  real(chm_real) DL2XA,DL2YA,DL2ZA,DL2XB,DL2YB,DL2ZB,DL2XC,DL2YC,DL2ZC
  real(chm_real) DMMXA1,DMMYA1,DMMZA1,DMMXB1,DMMYB1,DMMZB1, &
       DMMXC1,DMMYC1,DMMZC1
  real(chm_real) PF,PF1,PF2,XFAC,DELX,DELY,DELZ,RR2,RR1,FACT1,XTMP

  !     QC: UW_050214 fix bug 
  INTEGER NORBS
  !     QC: UW_050214 fix bug 
  !
  ! get the number of orbital from common block file 'stbgho.src'
  !
  NORBS=NORBG
  !
  NAOS   = NORBS-4*NQMLNK
  IJ = NAOS*(NAOS+1)/2
  N16 = 0
  !
  !      LOOP OVER QM-LINK ATOMS
  !
  DO I = 1,NQMLNK
     L1 = 4*(I-1)
     II = IQLINK(I)
     K = NAOS+4*(I-1)+1       !N1ST(II)
     L = K+3
     LINAO = K*(K-1)/2
     !      LOCATE POSITION OF DIAGONAL ELEMENTS
     DO J = K,L
        I1DIAG(J-K+1) = J*(J+1)/2
     ENDDO
     !      ZERO TEMPORARY VARIABLES
     DMMXA = 0.0
     DMMYA = 0.0
     DMMZA = 0.0
     DMMXB = 0.0
     DMMYB = 0.0
     DMMZB = 0.0
     DMMXC = 0.0
     DMMYC = 0.0
     DMMZC = 0.0

     N16J = N16
     DO J = 1,4
        DO K = 1,4
           N16J = N16J+1
           TINV(K,J) = BTM(N16J)
           XDTA(K,J) = DBTMMM(1,1,N16J)
           YDTA(K,J) = DBTMMM(2,1,N16J)
           ZDTA(K,J) = DBTMMM(3,1,N16J)
           XDTB(K,J) = DBTMMM(1,2,N16J)
           YDTB(K,J) = DBTMMM(2,2,N16J)
           ZDTB(K,J) = DBTMMM(3,2,N16J)
           XDTC(K,J) = DBTMMM(1,3,N16J)
           YDTC(K,J) = DBTMMM(2,3,N16J)
           ZDTC(K,J) = DBTMMM(3,3,N16J)
        ENDDO
     ENDDO

     !      AO-HB BLOCKS
     DO J = 1,4
        DO K = 1,NAOS
           IJ = IJ+1
           PF = 2.0*PHB(LINAO+K)*F(IJ)
           DMMXA = DMMXA + XDTA(1,J)*PF
           DMMXB = DMMXB + XDTB(1,J)*PF
           DMMXC = DMMXC + XDTC(1,J)*PF

           DMMYA = DMMYA + YDTA(1,J)*PF
           DMMYB = DMMYB + YDTB(1,J)*PF
           DMMYC = DMMYC + YDTC(1,J)*PF

           DMMZA = DMMZA + ZDTA(1,J)*PF
           DMMZB = DMMZB + ZDTB(1,J)*PF
           DMMZC = DMMZC + ZDTC(1,J)*PF
        ENDDO
        !
        !      HB-other HB blocks
        !
        M1 = LINAO+NAOS
        DO I1 = 1,I-1
           N16I1 = 16*(I1-1)
           DL2XA = 0.0
           DL2YA = 0.0
           DL2ZA = 0.0
           DL2XB = 0.0
           DL2YB = 0.0
           DL2ZB = 0.0
           DL2XC = 0.0
           DL2YC = 0.0
           DL2ZC = 0.0

           KK3 = M1+4*(I1-1)+1
           DO L1 = 1,4
              IJ = IJ+1
              PF = 2.0*PHB(KK3)*F(IJ)
              PF1 = BT (N16+J)*PF
              PF2 = BT (N16I1+L1)*PF
              DMMXA = DMMXA+PF2*XDTA(1,J)
              DMMXB = DMMXB+PF2*XDTB(1,J)
              DMMXC = DMMXC+PF2*XDTC(1,J)
              DMMYA = DMMYA+PF2*YDTA(1,J)
              DMMYB = DMMYB+PF2*YDTB(1,J)
              DMMYC = DMMYC+PF2*YDTC(1,J)
              DMMZA = DMMZA+PF2*ZDTA(1,J)
              DMMZB = DMMZB+PF2*ZDTB(1,J)
              DMMZC = DMMZC+PF2*ZDTC(1,J)

              KK1 = N16I1+4*(L1-1)+1
              DL2XA = DL2XA+PF1*DBTMMM(1,1,KK1)
              DL2YA = DL2YA+PF1*DBTMMM(2,1,KK1)
              DL2ZA = DL2ZA+PF1*DBTMMM(3,1,KK1)
              DL2XB = DL2XB+PF1*DBTMMM(1,2,KK1)
              DL2YB = DL2YB+PF1*DBTMMM(2,2,KK1)
              DL2ZB = DL2ZB+PF1*DBTMMM(3,2,KK1)
              DL2XC = DL2XC+PF1*DBTMMM(1,3,KK1)
              DL2YC = DL2YC+PF1*DBTMMM(2,3,KK1)
              DL2ZC = DL2ZC+PF1*DBTMMM(3,3,KK1)

           ENDDO

           II2 = IQLINK(I1)
           JJ = JQLINK(1,I1)
           LL = JQLINK(2,I1)
           MM = JQLINK(3,I1)
           DX(JJ) = DX(JJ)-DL2XA*CONFAC
           DY(JJ) = DY(JJ)-DL2YA*CONFAC
           DZ(JJ) = DZ(JJ)-DL2ZA*CONFAC
           DX(LL) = DX(LL)-DL2XB*CONFAC
           DY(LL) = DY(LL)-DL2YB*CONFAC
           DZ(LL) = DZ(LL)-DL2ZB*CONFAC
           DX(MM) = DX(MM)-DL2XC*CONFAC
           DY(MM) = DY(MM)-DL2YC*CONFAC
           DZ(MM) = DZ(MM)-DL2ZC*CONFAC
           DX(II2) = DX(II2)      +(DL2XA+DL2XB+DL2XC)*CONFAC
           DY(II2) = DY(II2)      +(DL2YA+DL2YB+DL2YC)*CONFAC
           DZ(II2) = DZ(II2)      +(DL2ZA+DL2ZB+DL2ZC)*CONFAC

        ENDDO
        !
        !      HB-HB BLOCKS
        !
        DO K = 1,J
           IJ = IJ+1
           XFAC = 2.0
           IF(K.EQ.J) XFAC = 1.0
           DO L = 1,4
              PF = XFAC*PHB(I1DIAG(L))*F(IJ)
              DMMXA = DMMXA+PF*(XDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTA(L,K))
              DMMXB = DMMXB+PF*(XDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTB(L,K))
              DMMXC = DMMXC+PF*(XDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTC(L,K))

              DMMYA = DMMYA+PF*(YDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTA(L,K))
              DMMYB = DMMYB+PF*(YDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTB(L,K))
              DMMYC = DMMYC+PF*(YDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTC(L,K))

              DMMZA = DMMZA+PF*(ZDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTA(L,K))
              DMMZB = DMMZB+PF*(ZDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTB(L,K))
              DMMZC = DMMZC+PF*(ZDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTC(L,K))
           ENDDO
        ENDDO
     ENDDO
     DMMXA1 = CONFAC*DMMXA
     DMMYA1 = CONFAC*DMMYA
     DMMZA1 = CONFAC*DMMZA
     DMMXB1 = CONFAC*DMMXB
     DMMYB1 = CONFAC*DMMYB
     DMMZB1 = CONFAC*DMMZB
     DMMXC1 = CONFAC*DMMXC
     DMMYC1 = CONFAC*DMMYC
     DMMZC1 = CONFAC*DMMZC
     !
     JJ = JQLINK(1,I)
     LL = JQLINK(2,I)
     MM = JQLINK(3,I)
     DX(JJ) = DX(JJ)-DMMXA1
     DY(JJ) = DY(JJ)-DMMYA1
     DZ(JJ) = DZ(JJ)-DMMZA1
     DX(LL) = DX(LL)-DMMXB1
     DY(LL) = DY(LL)-DMMYB1
     DZ(LL) = DZ(LL)-DMMZB1
     DX(MM) = DX(MM)-DMMXC1
     DY(MM) = DY(MM)-DMMYC1
     DZ(MM) = DZ(MM)-DMMZC1
     DX(II) = DX(II)      +DMMXA1+DMMXB1+DMMXC1
     DY(II) = DY(II)      +DMMYA1+DMMYB1+DMMYC1
     DZ(II) = DZ(II)      +DMMZA1+DMMZB1+DMMZC1
     N16 = N16+16
  ENDDO
  RETURN
END SUBROUTINE DQLINK

!---------------------------------------------------------------------
SUBROUTINE FTOFHB(F,FHB,BT,NATQM,NQMLNK,NORBS,N1ST,NLAST, &
     FAUXHB)
  !---------------------------------------------------------------------
  !
  !   TRANSFORMS A FULL FOCK MATRIX IN AO BASIS INTO ACTIVE HO BASIS
  !   ON INPUT:
  !      F      FOCK MATRIX IN AO, LOWER TRIANGLE
  !      BT     TRANSFORMATION MATRIX FOR EACH BOUNDARY ATOM, 4x4,NQMLNK
  !      N1ST,NMIDLE,NLAST - STANDARD MOPAC ARRAY !!!! NMIDLE removed 
  !      NATQM  NUMBER OF QM ATOMS
  !      NQMLNK NUMBER OF GHO BOUNDARY ATOMS
  !
  !   ON OUTPUT:
  !      FHB    FOCK MATRIX IN HO. INCLUDES ONLY ACTIVE ORBITALS
  !      FAUXHB diagnal elements of FOCK matrix for auxiliary orbital
  !             in hbo (N+4) basis
  !
  use chm_kinds
  implicit none

  INTEGER NATQM,NQMLNK,NORBS,N1ST(*),NLAST(*)
  real(chm_real)  F(*),FHB(*),BT(16,*)
  !
  INTEGER I,J,K,L,II,JJ,IJ,IA,IB,JA,JB,I1,IAMONE,INDF,IAL
  INTEGER NACTATM,NORBAO,LIN1
  real(chm_real)  FTMP(10),FTMP2(10)
  LOGICAL FIRST
  DATA FIRST/.TRUE./
  SAVE FIRST,NORBAO,LIN1,NACTATM
  !
  ! auxiliary orbitals on each gho boundary
  !
  INTEGER IND
  real(chm_real) FAUXHB(3,*)

  IF(FIRST) THEN
     NORBAO = NORBS-4*NQMLNK
     LIN1 = NORBAO*(NORBAO+1)/2
     NACTATM = NATQM-NQMLNK
     FIRST = .FALSE.
  ENDIF
  !
  !      CORE PART NOT AFFECTED
  !
  DO I = 1,LIN1
     FHB(I) = F(I)
  ENDDO
  !   Loop over QM-boundary atoms for orbitals to be transformed.
  DO I = 1,NQMLNK
     !   F(mu,l), AO-HO block
     !   Only one active HO per QM-boundary atom.
     I1 = NORBAO+I
     I1 = I1*(I1-1)/2
     II = NACTATM+I
     IA = N1ST(II)
     IB = NLAST(II)
     IAMONE = IA-1
     IJ = I1
     DO J = 1,NORBAO
        IJ = IJ+1
        FHB(IJ) = 0.0
        DO K = IA,IB
           FHB(IJ) = FHB(IJ)+BT(K-IA+1,I)*F(J+K*(K-1)/2)
        ENDDO
     ENDDO
     !   F(l,l'), HO-other HO block
     DO J = 1,I-1
        JJ = NACTATM+J
        JA = N1ST(JJ)
        IJ = IJ+1
        FHB(IJ) = 0.0
        DO L = 1,4
           FTMP(L) = 0.0
           IAL = IA+L-1
           INDF = JA-1+IAL*(IAL-1)/2
           DO K = 1,4
              FTMP(L) = FTMP(L)+F(INDF+K)*BT(K,J)
           ENDDO
           FHB(IJ) = FHB(IJ)+FTMP(L)*BT(L,I)
        ENDDO
     ENDDO
     !   F(l,l), HO-HO corner block
     L = 0
     DO J = IA,IB
        JA = J*(J-1)/2
        DO K = IA,J
           L = L+1
           FTMP(L) = F(K+JA)
           FTMP2(L) = 0.0
        ENDDO
     ENDDO
     CALL VBFTN(FTMP,FTMP2,BT(1,I),4)
     FHB(IJ+1) = FTMP2(1)
     !
     ! save the diagnal FOCK elements for auxililary orbital, these will
     ! be used to calculate the auxilieary orbital energy when the energy
     ! weighted density matrix is constructed.
     !
     DO IND = 1, 3
        FAUXHB(IND, I) = FTMP2((IND+1)*(IND+2)/2)
     ENDDO

  ENDDO
  RETURN
END SUBROUTINE FTOFHB

!
!--------------------------------------------------------------------
SUBROUTINE CTRASF(CHB,C,NORBS)
  !--------------------------------------------------------------------
  use stbgho
  implicit none

  real(chm_real) CHB(*),C(*)
  INTEGER I,J,I1,J1,IJ,NORBHB,NAOS
  !     QC: UW_050214: fix bug
  INTEGER NORBS
  !     QC: UW_050214: fix bug
  !
  NORBHB = NORBS - 3*NQMLNK
  NAOS = NORBHB-NQMLNK

  DO I = 1,NORBHB
     I1 = NORBS*(I-1)
     J1 = NORBHB*(I-1)
     DO J = 1,NAOS
        I1 = I1+1
        J1 = J1+1
        C(I1) = CHB(J1)
     ENDDO
     DO J = NAOS+1,NORBHB
        J1 = J1+1
        I1 = I1+1
        IJ = 16*(J-NAOS)-15
        C(I1) = CHB(J1)*BT(IJ)
        I1 = I1+1
        IJ = IJ+1
        C(I1) = CHB(J1)*BT(IJ)
        I1 = I1+1
        IJ = IJ+1
        C(I1) = CHB(J1)*BT(IJ)
        I1 = I1+1
        IJ = IJ+1
        C(I1) = CHB(J1)*BT(IJ)
     ENDDO
  ENDDO
  !
  !  append and tranform auxiliary hybrid orbitals ... PJ 12/2002
  !
  DO I = 1, NQMLNK
     DO J = 1, 3
        ! bug fixed, 04/2003
        !            I1 = NORBS*(NORBHB+I*J-1)
        I1 = NORBS*(NORBHB+(I-1)*3+J-1)
        IJ = 16*(I-1)+4*J
        DO J1 = 1, NORBS
           I1 = I1 + 1
           IF (J1 .GT. NAOS+(I-1)*4 .AND. &
                J1 .LE. NAOS+I*4) THEN
              IJ = IJ + 1
              C(I1) = BT(IJ)
           ELSE
              C(I1) = 0.0
           END IF
        END DO
     END DO
  END DO

  RETURN
END SUBROUTINE CTRASF


!-----------------------------------------------------------------------
SUBROUTINE RPMMM(DX,DY,DZ,X,Y,Z,ECLASS)
  !-----------------------------------------------------------------------
  ! compute the classical nuclear repulsion between MMs linked to
  ! GHO atom
  use dimens_fcm
  use sizes
  use stbgho
  use stream
  use consta
  use psf
  !
  implicit none

  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*),ECLASS
  !
  !  Local variables
  !
  INTEGER I, J, K, L, M, N, II, JJ, LL, MM, I1, L1, M1, IJ
  INTEGER KK1,KK2,KK3,II2,JJ2,N16,N16I1,N16J
  INTEGER I1DIAG(4),NAOS,LINAO
  real(chm_real) XDTA(4,4),YDTA(4,4),ZDTA(4,4),XDTB(4,4),YDTB(4,4)
  real(chm_real) XDTC(4,4),YDTC(4,4),ZDTC(4,4),ZDTB(4,4),TINV(4,4)
  real(chm_real) DMMXA,DMMYA,DMMZA,DMMXB,DMMYB,DMMZB,DMMXC,DMMYC,DMMZC
  real(chm_real) DL2XA,DL2YA,DL2ZA,DL2XB,DL2YB,DL2ZB,DL2XC,DL2YC,DL2ZC
  real(chm_real) DMMXA1,DMMYA1,DMMZA1,DMMXB1,DMMYB1,DMMZB1, &
       DMMXC1,DMMYC1,DMMZC1
  real(chm_real) PF,PF1,PF2,XFAC,DELX,DELY,DELZ,RR2,RR1,FACT1,XTMP
  !
  !     QC: UW_050214: Fix bug
  INTEGER NORBS
  !     QC: UW_050214: Fix bug
  NORBS=NORBG
  NAOS   = NORBS-4*NQMLNK
  IJ = NAOS*(NAOS+1)/2
  N16 = 0
  !
  ! include nuclear interactions between mm-boundary atoms
  ! loop over QM-LINK atoms
  !
  DO I = 1,NQMLNK
     DO M = 1,2
        JJ = JQLINK(M,I)
        M1 = M+1
        DO N = M1,3
           JJ2 = JQLINK(N,I)
           DELX = X(JJ2)-X(JJ)
           DELY = Y(JJ2)-Y(JJ)
           DELZ = Z(JJ2)-Z(JJ)
           RR2 = DELX*DELX+DELY*DELY+DELZ*DELZ
           RR1 = SQRT(RR2)
           fact1 = ccelec*cg(jj)*cg(jj2)/rr1
           ECLASS = ECLASS+FACT1
           FACT1 = FACT1/RR2
           XTMP = DELX*FACT1
           DX(JJ) = DX(JJ)+XTMP
           DX(JJ2) = DX(JJ2)-XTMP
           XTMP = DELY*FACT1
           DY(JJ) = DY(JJ)+XTMP
           DY(JJ2) = DY(JJ2)-XTMP
           XTMP = DELZ*FACT1
           DZ(JJ) = DZ(JJ)+XTMP
           DZ(JJ2) = DZ(JJ2)-XTMP
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE RPMMM

!-----------------------------------------------------------------
SUBROUTINE CEXPD(CHB,NORBS,NQMLNK)
  !-----------------------------------------------------------------
  ! expand MO in the HB basis to the reduced dimension, rearrange
  ! the order of basis from active-grouped to atom-based
  use chm_kinds
  implicit none

  !
  INTEGER NORBS, NQMLNK
  real(chm_real) CHB(*)
  !
  INTEGER I,J,K,KK,NORBHB
  real(chm_real) C(NORBS*NORBS)
  !
  NORBHB = NORBS-3*NQMLNK
  K = 0
  KK = 0
  DO I = 1, NORBS
     DO J = 1, NORBS
        K = K + 1
        IF (I.LE.NORBHB .AND. J.LE.NORBHB) THEN
           KK = KK + 1
           C(K) = CHB(KK)
        ELSE IF (I.EQ.J) THEN
           C(K) = 1.0
        ELSE
           C(K) = 0.0
        ENDIF
     ENDDO
  ENDDO
  !
  IF (NQMLNK .GT. 1) THEN
     CALL REBAS(C, NORBS, NQMLNK)
  ENDIF
  !
  K = 0
  DO I = 1, NORBS
     DO J = 1, NORBS
        K = K+1
        CHB(K) = C(K)
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE CEXPD

!----------------------------------------------------------------------
SUBROUTINE REBAS(VHB, NORBS, NQMLNK)
  !----------------------------------------------------------------------
  ! reorder the HBO basis to ensure the basis for the same boundary
  ! atom come together, which then become consistent to any basis
  ! transformation
  use chm_kinds
  implicit none

  !
  INTEGER NORBS, NQMLNK
  real(chm_real) VHB(*)
  !
  real(chm_real) V1(NORBS,NORBS), V2(NORBS, NORBS)
  INTEGER I,J,K, NORBHB, NAOS
  !
  NORBHB=NORBS-3*NQMLNK
  NAOS=NORBHB-NQMLNK
  K = 1
  DO J = 1, NORBS
     DO I = 1, NORBS
        V1(I,J) = VHB(K)
        K = K + 1
     ENDDO
  ENDDO
  DO J = 1, NORBS
     DO I = 1, NORBS-4*NQMLNK
        V2(I,J) = V1(I,J)
     ENDDO
     DO K = 1, NQMLNK
        V2(NAOS+4*(K-1)+1,J)=V1(NAOS+K,J)
        V2(NAOS+4*(K-1)+2,J)=V1(NORBHB+3*(K-1)+1,J)
        V2(NAOS+4*(K-1)+3,J)=V1(NORBHB+3*(K-1)+2,J)
        V2(NAOS+4*(K-1)+4,J)=V1(NORBHB+3*(K-1)+3,J)
     ENDDO
  ENDDO
  K = 1
  DO J = 1, NORBS
     DO I = 1, NORBS
        VHB(K) = V2(I,J)
        K = K + 1
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE REBAS

!-----------------------------------------------------------------
SUBROUTINE PWHO(P,W,V,OCC,E,NORBS)
  !-----------------------------------------------------------------
  !
  ! Compute the total density matrix (P) and energy weighted density 
  ! matrix (W) in HB basis for GHO wavefunction, note auxiliary 
  ! orbitals are fractionally occupied.
  ! Note: W computed here is NEGATIVE energy weigthed
  !       density matrix.
  !
  use chm_kinds
  implicit none

  !
  INTEGER NORBS
  real(chm_real) P(*),W(*),V(NORBS,NORBS),OCC(*),E(*)
  !
  INTEGER IJ,I,J,K
  real(chm_real) DUMP,DUMW
  !
  IJ = 0
  DO I = 1,NORBS
     DO J = 1,I
        IJ = IJ + 1
        DUMP = 0.0
        DUMW = 0.0
        DO K = 1, NORBS
           DUMP = DUMP + OCC(K)*V(I,K)*V(J,K)
           DUMW = DUMW - OCC(K)*E(K)*V(I,K)*V(J,K)
        enddo
        P(IJ) = DUMP
        W(IJ) = DUMW
     enddo
  enddo
  RETURN
END SUBROUTINE PWHO

!----------------------------------------------------------------------
SUBROUTINE GHOHYB(ISLCT, LSLCT)
  !----------------------------------------------------------------------
  use dimens_fcm
  use coord
  use psf
  use stream
  !
  use stbgho
  !
  implicit none

  INTEGER ISLCT(*), LSLCT(*)

  INTEGER I, J, II, JJ, IBND, ITST
  real(chm_real)  ZERO,ONE,TWO,THREE
  DATA ZERO,ONE,TWO,THREE/0.0,1.00,2.00,3.00/
  SAVE ZERO,ONE,TWO,THREE

  !
  ! put NATQM to common block to used by GHO routin, later
  !
  NATQM = 0
  NQMLNK = 0
  DO I = 1, NATOM

     !
     ! a QM atom
     !
     IF ((ISLCT(I).EQ.1) .AND. (LSLCT(I) .NE. 1)) THEN
        NATQM = NATQM + 1
        IGHOSL(NATQM) = 0
     END IF
  END DO

  ! we need to loop all atom twice to put pure-QM atoms first and
  ! group GHO atom at the end of QM selection, 0601PJ07

  DO I = 1, NATOM
     !
     ! a GHO boundary atom
     !
     IF ((ISLCT(I).EQ.1) .AND. (LSLCT(I).EQ. 1)) THEN
        NATQM = NATQM + 1
        NQMLNK = NQMLNK + 1
        IQLINK(NQMLNK) = I
        IGHOSL(NATQM) = 1
     END IF
  END DO

  !
  ! check number of GHO atoms, 0601PJ07
  !
  IF (NQMLNK .GT. MAXQMLINK) CALL WRNDIE(-5,'GHOHYB>', &
       'Too many GHO boundary atoms')

  ! check the connectivity of GHO boundary atom to MM and QM fragment
  !
  DO I = 1, NQMLNK
     IBND = 0
     ITST = 0
     DO J = 1,NBOND
        II = IB(J)
        JJ = JB(J)
        IF (II .EQ. IQLINK(I)) THEN
           IF (ISLCT(JJ) .GT. 0) THEN
              ITST = ITST + 1
              IF(ITST .GT. 1) CALL WRNDIE(-5,'GHOHYB>', &
                   'Too many QM atoms connected to the GHO boundary atom')
              KQLINK(I) = JJ
           ELSE
              IBND = IBND + 1
              IF (IBND .GT. 3 ) CALL WRNDIE(-5, 'GHOHYB>', &
                   'Too many MM bonds connecting the GHO boundary atom')
              JQLINK(IBND,I) = JJ
           END IF
        ELSE IF (JJ .EQ. IQLINK(I)) THEN
           IF (ISLCT(II) .GT. 0) THEN
              ITST = ITST + 1
              IF(ITST .GT. 1) CALL WRNDIE(-5,'GHOHYB>', &
                   'Too many QM atoms connected to the GHO boundary atom')
              KQLINK(I) = II
           ELSE
              IBND = IBND + 1
              IF (IBND .GT. 3) CALL WRNDIE(-5, 'GHOHYB>', &
                   'Too many MM bonds connecting the GHO boundary atom')
              JQLINK(IBND,I) = II
           END IF
        END IF
     END DO
  END DO

  !
  ! set label for a GHO and QM atom directly linked to GHO boundary in
  ! the QM domain, this will be used in the 1-e integral scaling
  ! to find out the scaling orbital pairs in QM domain.
  !        IGLNK - index of GHO boudnary in the QM atom list
  !        KGLNK - index of QM connected to GHO in the QM list
  !
  J = 0
  DO I = 1, NATQM
     IF (IGHOSL(I) .EQ. 1) THEN
        J = J + 1
        IGLNK (J) = I
        ! if regroup GHO atom at the end of QM, this relative position does not work,
        ! we have to explicitly obtain the QM index of the frontier atom, which is
        ! important for the empirical repulsive correction term between Q-B, 0601PJ07
        !             KGLNK (J) = KQLINK(J) - IQLINK(J) + I
        !
        II = 0
        DO JJ = 1, NATOM
           IF (ISLCT(JJ) .EQ. 1 .AND. LSLCT(JJ) .NE. 1) THEN
              II = II + 1
              IF (KQLINK(J) .EQ. JJ) KGLNK(J) = II
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  !
  ! define hybrid orbital transformation matrix
  !
  CALL HBDEF(X,Y,Z,BT,BTM,DBTMMM,NQMLNK,IQLINK,JQLINK,KQLINK)

  !
  ! determine core potentials, record QM-link atom charges for
  ! auxiliary density, and then zero MM charge on QM-link atom.
  !
  DO I = 1,NQMLNK
     II = NATQM-NQMLNK+I
     QMATMQ(I) = CG(IQLINK(I))
     CG(IQLINK(I)) = 0.0D+00
  ENDDO

  RETURN
END SUBROUTINE GHOHYB

!----------------------------------------------------------
FUNCTION QBREP(R,C1,C2,C4)
  !----------------------------------------------------------
  ! compute an empirical repulsion between Q-B pair for GHO,
  ! the cubic polynomial is obtained from the difference
  ! between QM and QM/MM energies along A-B bond in ethane
  ! PES scan
  use chm_kinds
  implicit none

  real(chm_real) QBREP
  real(chm_real) R,C1,C2,C4, CONF
  QBREP = (C1*R + C2*R**2.0D0 + C4*R**4.0D0)
  RETURN
END FUNCTION QBREP

!----------------------------------------------------------
FUNCTION QBGRD(R,C1,C2,C4)
  !----------------------------------------------------------
  ! compute gradient of an empirical repulsion between
  ! Q-B pair (w/ to r) for GHO, the cubic polynomial is
  ! obtained  from the difference between QM and QM/MM
  ! energies along A-B bond in ethane PES scan
  !
  use chm_kinds
  implicit none

  real(chm_real) QBGRD
  real(chm_real) R,C1,C2,C4
  QBGRD = C1 + 2.0D0*C2*R + 4.0D0*C4*R**3.0D0
  RETURN
END FUNCTION QBGRD

!----------------------------------------------------------
SUBROUTINE GTCOEF(IZP,J,K,C1,C2,C4)
  !----------------------------------------------------------
  ! obtain the empirical correction term coefficients for
  ! various QM/MM bond types.
  use consta
  !
  implicit none

  INTEGER IZP(*),J,K
  real(chm_real) C1,C2,C4
  real(chm_real) CONE, CTWO, CFOUR
  COMMON/QECOR/ CONE(5), CTWO(5), CFOUR(5)
  !
  INTEGER ITYP,LJ,LK
  real(chm_real) QJ,QK
  !
  INTEGER,PARAMETER :: MAXTYP= 6
  real(chm_real) QZERO, UHUBB
  COMMON /MCHARGE/ QZERO(MAXTYP), UHUBB(MAXTYP)
  INTEGER LMAX(MAXTYP)
  COMMON /LMAX/ LMAX
  !
  QJ = QZERO(IZP(J))
  QK = QZERO(IZP(K))
  LJ = LMAX(IZP(J))
  LK = LMAX(IZP(K))
  !
  ! C-C bond
  !
  IF ((QJ.EQ.4.0.AND.QK.EQ.4.0).AND. &
       (LJ.EQ.2.AND.LK.EQ.2)) THEN
     ITYP = 1
     !
     ! C-O bond
     !
  ELSE IF ((QJ.EQ.6.0.AND.QK.EQ.4.0.AND. &
       LJ.EQ.2.AND.LK.EQ.2) .OR. &
       (QJ.EQ.4.0.AND.QK.EQ.6.0.AND. &
       LJ.EQ.2.AND.LK.EQ.2)) THEN
     ITYP = 2
     !
     ! C-S bond
     !
  ELSE IF ((QJ.EQ.6.0.AND.QK.EQ.4.0.AND. &
       LJ.EQ.3.AND.LK.EQ.2) .OR. &
       (QJ.EQ.4.0.AND.QK.EQ.6.0.AND. &
       LJ.EQ.2.AND.LK.EQ.3)) THEN
     ITYP = 3
     !
     ! other wise, use zero coefficient
     !
  ELSE
     ITYP = 5
  END IF

  C1 = CONE(ITYP)*BOHRR/TOKCAL
  C2 = CTWO(ITYP)*BOHRR**2.0/TOKCAL
  C4 = CFOUR(ITYP)*BOHRR**4.0/TOKCAL

  RETURN
END SUBROUTINE GTCOEF

!----------------------------------------------------------
SUBROUTINE GTDIM(NN, NGLT, NGSQ)
  !----------------------------------------------------------
  ! obtain the dimensionality of the lower-triangle elements
  ! for total density matrix (or Fock matrix) in GHO in order
  ! to allocate arrays for them dynamically 
  !
  implicit none

  INTEGER NN,NEL,NBEWEG,J,IZPJ,NDIM,NGLT,NGSQ   
  integer,PARAMETER :: NNDIM=650,MAXTYP= 6
  INTEGER IZP(NNDIM),IND(NN+1),LMAX(MAXTYP)
  COMMON /IZPC/ IZP,NEL,NBEWEG
  COMMON /LMAX/ LMAX
  !
  IND(1) = 0
  DO J = 1,NN
     IZPJ = IZP(J)
     IND(J+1) = IND(J)+LMAX(IZPJ)**2
  END DO
  NDIM = IND(NN+1)
  NGLT = NDIM*(NDIM+1)/2
  NGSQ = NDIM*NDIM
  RETURN
END SUBROUTINE GTDIM

!--mfc-- !----------------------------------------------------------
!--mfc--       BLOCK DATA GECOR
!--mfc-- !----------------------------------------------------------
!--mfc-- ! empirical repulsion term between A-B
!--mfc-- !
!--mfc--       real(chm_real) CONE, CTWO, CFOUR
!--mfc--       COMMON/QECOR/ CONE(5), CTWO(5), CFOUR(5)
!--mfc-- !     ----------------------------------------------------------
!--mfc-- !                     C-C        C-O        C-S
!--mfc-- !     ----------------------------------------------------------
!--mfc--       DATA CONE  /  6.422D0,  16.767D0,   9.262D0, 0.0D0, 0.0D0/
!--mfc--       DATA CTWO  /-55.715D0, -45.902D0, -55.496D0, 0.0D0, 0.0D0/
!--mfc--       DATA CFOUR /  7.040D0,   1.721D0,   5.006D0, 0.0D0, 0.0D0/
!--mfc-- !     ----------------------------------------------------------
!--mfc-- 
#endif /* (stbgho_main)*/

subroutine stbgho_null
  return
end subroutine stbgho_null

