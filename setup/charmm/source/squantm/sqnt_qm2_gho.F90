#if KEY_SQUANTM==1 /*mainsquatn*/
      SUBROUTINE GHOHYB(Natom,ISL,LSL,NBOND,IB,JB,Cggho,X,Y,Z,CG,QFail, &
                        Qdual_check,qfirst)
!
! Prepare for GHO setup, initial setup
!
  use qmmm_module, only : Get_Type_qm2_ghos, &
                              qm2_ghos,qm2_ghos_p,qm2_ghos_r

  use chm_kinds
  use stream

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: Natom, ISL(*), LSL(*)
      integer, intent(in)    :: NBOND, IB(*), JB(*)
      real(chm_real),intent(in)    :: X(*),Y(*),Z(*)
      real(chm_real),intent(inout) :: CG(*)
      real(chm_real),intent(inout) :: Cggho        ! Charge on GHO atoms to be 
                                             ! used in QM/MM-Ewald sum
      logical, intent(inout) :: QFail
      logical, intent(in)    :: Qdual_check(2),qfirst

! Local variables
      integer                :: ier=0
      integer                :: i, j, ii, jj, ibnd, itst
      integer                :: ngho,ngho2

! for dual quantum region
! copy from ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            if(qfirst) then
               call Get_Type_qm2_ghos(qm2_ghos_r,qm2_ghos) ! put from previous call.
               call Get_CG_GHO(CG,Qdual_check)
            end if
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_p)
         else                                              ! for 1st qm region
! this not setup yet!
!           if(qfirst) then 
!              call Get_Type_qm2_ghos(qm2_ghos_p,qm2_ghos) ! put from previous call.
!              call Get_CG_GHO(CG,Qdual_check)
!           end if
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_r)
         end if
      end if

! Put qm2_ghos%natqm to common block to used by GHO routine later
      qm2_ghos%nqmlnk = 0
      qm2_ghos%natqm  = 0

! loop over Natom to check
      Do i=1,natom
         If ( (ISL(i).eq.1) .and. (LSL(i).ne.1) ) then     ! for pure QM atom
            qm2_ghos%natqm  = qm2_ghos%natqm + 1
         Else if( (ISL(i).eq.1) .and. (LSL(i).eq.1) ) then ! for GHO atom 
            qm2_ghos%nqmlnk = qm2_ghos%nqmlnk + 1
            qm2_ghos%natqm  = qm2_ghos%natqm + 1
         End if
      End do

      If(qm2_ghos%nqmlnk.GE.1) then
!!!! this line moved into QMMM_INITIALIZE
!!!! qm2_ghos%QGHO=.TRUE.

! Deallocate necessary memory after check
         If(associated(qm2_ghos%IQLINK)) Deallocate(qm2_ghos%IQLINK)
         If(associated(qm2_ghos%JQLINK)) Deallocate(qm2_ghos%JQLINK)
         If(associated(qm2_ghos%KQLINK)) Deallocate(qm2_ghos%KQLINK)
         If(associated(qm2_ghos%QMATMQ)) Deallocate(qm2_ghos%QMATMQ)
         If(associated(qm2_ghos%BT))     Deallocate(qm2_ghos%BT)
         If(associated(qm2_ghos%BTM))    Deallocate(qm2_ghos%BTM)
         If(associated(qm2_ghos%DBTMMM)) Deallocate(qm2_ghos%DBTMMM)
 
! Allocate memory
         ngho  = qm2_ghos%nqmlnk
         ngho2 = qm2_ghos%nqmlnk * qm2_ghos%mqm16
         Allocate(qm2_ghos%IQLINK(ngho), stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','IQLINK')

         Allocate(qm2_ghos%JQLINK(3,ngho),stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','JQLINK')

         Allocate(qm2_ghos%KQLINK(ngho), stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','KQLINK')

         Allocate(qm2_ghos%QMATMQ(ngho), stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','QMATMQ')

         Allocate(qm2_ghos%BT(ngho2),    stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','BT   ')

         Allocate(qm2_ghos%BTM(ngho2),   stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','BTM  ')

         Allocate(qm2_ghos%DBTMMM(3,3,ngho2),stat=ier)
         if(ier.ne.0) call Aass(1,'GHOHYB','DBTMMM')

! loop over again and fill qm2_ghos%IQLINK array
         ii=0
         Do i = 1, natom
            If( (ISL(i).eq.1) .and. (LSL(i).eq.1) ) then
               ii=ii+1
               qm2_ghos%IQLINK(ii) = i 
            End if
         End do

! check the connectivity of GHO boundary atoms to MM and QM fragment
         Do i=1,qm2_ghos%nqmlnk
            ibnd = 0
            itst = 0
            do j = 1, NBOND
               ii=IB(J)
               jj=JB(J)
               If(II .eq. qm2_ghos%IQLINK(i)) then              ! II is gho atom
                  If(ISL(jj) .GT. 0) then                       ! jj is qm atom
                     itst = itst + 1
                     If(itst.gt.1) then
                        If(Prnlev.ge.2) Write(6,*)   &
          'GHOHYB> Too many QM atoms connected to the GHO boundary atom'
                        QFail=.FALSE.
                        return
                     End if
                     qm2_ghos%KQLINK(i) = jj
                  Else                                          ! jj is mm atom
                     ibnd = ibnd + 1
                     If(ibnd.gt.3) then
                        If(Prnlev.ge.2) Write(6,*)    &
            'GHOHYB> Too many MM bonds connecting the GHO boundary atom'
                        QFail=.FALSE.
                        return
                     End if
                     qm2_ghos%JQLINK(ibnd,i) = jj
                  End if
               Else if(JJ .eq. qm2_ghos%IQLINK(i)) then         ! JJ is gho atom
                  If(ISL(ii) .GT. 0) then                       ! ii is qm atom
                     itst = itst + 1
                     If(itst.gt.1) then
                        If(Prnlev.ge.2) Write(6,*)    &
          'GHOHYB> Too many QM atoms connected to the GHO boundary atom'
                        QFail=.FALSE.
                        return
                     End if
                     qm2_ghos%KQLINK(i) = ii
                  Else                                          ! ii is mm atom
                     ibnd = ibnd + 1
                     If(ibnd.gt.3) then
                        If(Prnlev.ge.2) Write(6,*)    &
            'GHOHYB> Too many MM bonds connecting the GHO boundary atom'
                        QFail=.FALSE.
                        return
                     End if
                     qm2_ghos%JQLINK(ibnd,i) = ii
                  End if
               End if
            end do
         End do

! determine core potentials, record QM-link atom charges for
! auxiliary density, and then zero MM charge on QM-link atom
         Cggho = zero
         Do i=1,qm2_ghos%nqmlnk
            qm2_ghos%QMATMQ(i)     = CG(qm2_ghos%IQLINK(i))
            CG(qm2_ghos%IQLINK(i)) = zero
            Cggho                  = Cggho + qm2_ghos%QMATMQ(i)
         End do

! Define hybrid orbital transformation matrix
         Call HBDEF(X,Y,Z,qm2_ghos%BT,qm2_ghos%BTM,qm2_ghos%DBTMMM,   &
                    qm2_ghos%nqmlnk,   &
                    qm2_ghos%IQLINK,qm2_ghos%JQLINK,qm2_ghos%KQLINK,   &
                    QFail)

      Else
!!!! This line moved into QMMM_INITIALIZE
!!!! qm2_ghos%QGHO=.FALSE.
         If(Prnlev.ge.2) Write(6,*)    &
         'GHOHYB> No GHO atoms are selected. Maybe coding error.'
      End if

! for dual quantum region
! copy back into ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qm2_ghos(qm2_ghos_p,qm2_ghos)
         else                                              ! for 1st qm region
            call Get_Type_qm2_ghos(qm2_ghos_r,qm2_ghos)
         end if
      end if

      return
      END SUBROUTINE GHOHYB


      SUBROUTINE gho_allocate(NORBS)
!
! Allocate necessary memory for GHO atom
!

  use qmmm_module, only : qm2_ghos

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer, intent(in)    :: NORBS

! Local variables
      integer                :: ier=0
      integer                :: mems

! Set number of AOs for GHO gradient calculations
      qm2_ghos%norbsgho = NORBS

!**Note***********************************************************
! For the moment no UHF, since AMBER new mopac code doesn't 
! support UHF yet.
      qm2_ghos%UHFGHO   = .FALSE.
!*****************************************************************

! Deallocate necessary memory if associated
! When you change here, then also change routine "qm_print_dyn_mem"
      If(associated(qm2_ghos%PHO )) Deallocate(qm2_ghos%PHO )

! Allocate memory
      mems = NORBS*(NORBS+1) / 2
      Allocate(qm2_ghos%PHO(mems) , stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','PHO')
      
      If(qm2_ghos%UHFGHO) then
         If(associated(qm2_ghos%PBHO)) Deallocate(qm2_ghos%PBHO)
         Allocate(qm2_ghos%PBHO(mems), stat=ier)
         if(ier.ne.0) call Aass(1,'gho_allocate','PBHO')
      End if

!***Note
!!! commenting out this and will be cleaned up later
!!! We don't need these since fock_matrix is intact in 
!!! qm2_scf routine
!!!
!!! If(associated(qm2_ghos%FAOA)) Deallocate(qm2_ghos%FAOA)
!!! If(associated(qm2_ghos%FAOB)) Deallocate(qm2_ghos%FAOB)
!!! Allocate(qm2_ghos%FAOA(mems), stat=ier)
!!! if(ier.ne.0) call Aass(1,'gho_allocate','FAOA')
!!! Allocate(qm2_ghos%FAOB(mems), stat=ier)
!!! if(ier.ne.0) call Aass(1,'gho_allocate','FAOB')

! Allocate more local variables used in qm2_scf
! Put here, since when GHO method is used, qm2_scf routine 
!           will use this variable
!           everytime energy has been called.
      If(associated(qm2_ghos%CAHB )) Deallocate(qm2_ghos%CAHB )
      If(associated(qm2_ghos%DAHB )) Deallocate(qm2_ghos%DAHB )
      If(associated(qm2_ghos%FAHB )) Deallocate(qm2_ghos%FAHB )
      If(associated(qm2_ghos%PAHB )) Deallocate(qm2_ghos%PAHB )

      Allocate(qm2_ghos%CAHB(NORBS*NORBS), stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','CAHB')
      Allocate(qm2_ghos%DAHB(mems), stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','DAHB')
      Allocate(qm2_ghos%FAHB(mems), stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','FAHB')
      Allocate(qm2_ghos%PAHB(mems), stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','PAHB')


!!! temporay commenting out, till GHO-DIIS to be implemented
      If(associated(qm2_ghos%PAOLD)) Deallocate(qm2_ghos%PAOLD)
!!!   If(associated(qm2_ghos%PBOLD)) Deallocate(qm2_ghos%PBOLD)
      Allocate(qm2_ghos%PAOLD(mems), stat=ier)
      if(ier.ne.0) call Aass(1,'gho_allocate','PAOLD')
!!! Allocate(qm2_ghos%PBOLD(mems), stat=ier)
!!! if(ier.ne.0) call Aass(1,'gho_allocate','PBOLD')

      If(qm2_ghos%UHFGHO) then
         If(associated(qm2_ghos%CBHB )) Deallocate(qm2_ghos%CBHB )
         If(associated(qm2_ghos%DBHB )) Deallocate(qm2_ghos%DBHB )
         If(associated(qm2_ghos%FBHB )) Deallocate(qm2_ghos%FBHB )
         If(associated(qm2_ghos%PBHB )) Deallocate(qm2_ghos%PBHB )

         Allocate(qm2_ghos%CBHB(NORBS*NORBS), stat=ier)
         if(ier.ne.0) call Aass(1,'gho_allocate','CBHB')
         Allocate(qm2_ghos%DBHB(mems), stat=ier)
         if(ier.ne.0) call Aass(1,'gho_allocate','DBHB')
         Allocate(qm2_ghos%FBHB(mems), stat=ier)
         if(ier.ne.0) call Aass(1,'gho_allocate','FBHB')
         Allocate(qm2_ghos%PBHB(mems), stat=ier)
         if(ier.ne.0) call Aass(1,'gho_allocate','PBHB')
      End if

      return
      END SUBROUTINE gho_allocate


      SUBROUTINE HBDEF(X,Y,Z,BT,BTM,DBTMMM,NATVB,IATVB,JATVB,KATVB, &
                      QFail)
! 
! Define transformation matrix
!

      
  use chm_kinds

  use qm2_double
  use qm2_constants

  use stream

      implicit none

! Passed in
      real(chm_real), intent(in)   :: X(*),Y(*),Z(*)
      real(chm_real), intent(out)  :: BT(*),BTM(*),DBTMMM(3,3,*)
      integer,intent(in)     :: IATVB(*),JATVB(3,*),KATVB(*)
      integer,intent(in)     :: NATVB
      logical,intent(inout)  :: QFail

! Local variables
      integer                :: I,II,JJ,J1,J2,J3,NI
      real(chm_real)               :: A(3),B(3),C(3),AB(3),AC(3),P(3),T(3)


      NI = 0
      Do I = 1,NATVB                  ! Each GHO atom is connected to 1 QM atom 
         II = IATVB(I)                ! and 3 MM atoms

         J1 = JATVB(1,I)              ! MM atoms
         A(1) = X(J1)-X(II)
         A(2) = Y(J1)-Y(II)
         A(3) = Z(J1)-Z(II)
         J2 = JATVB(2,I)
         B(1) = X(J2)-X(II)
         B(2) = Y(J2)-Y(II)
         B(3) = Z(J2)-Z(II)
         J3 = JATVB(3,I)
         C(1) = X(J3)-X(II)
         C(2) = Y(J3)-Y(II)
         C(3) = Z(J3)-Z(II)

         JJ = KATVB(I)                ! QM atom
         T(1) = X(JJ)-X(II)
         T(2) = Y(JJ)-Y(II)
         T(3) = Z(JJ)-Z(II)

         P(1) = zero
         P(2) = zero
         P(3) = zero
         CALL HBDRIV(A,B,C,T,P,BT(NI+1:NI+16),BTM(NI+1:NI+16),  &
                     DBTMMM(1:3,1:3,NI+1:NI+16))

         IF(P(1).NE.zero) THEN
            If(Prnlev.ge.2) Write(6,*)  &
                            'HBDEF> HYBRID ORBITAL ILLDEFINED.'
            QFail=.FALSE.
            Return
         END IF

         NI = NI+16

      End do

      return
      END SUBROUTINE HBDEF


      SUBROUTINE HBDRIV(A,B,C,D,O,BT,BTM,DBTMMM)
!
!
!

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      real(chm_real), intent(in)   :: A(3),B(3),C(3),D(3)
      real(chm_real), intent(inout):: O(3)
      real(chm_real), intent(out)  :: BT(4,4),BTM(4,4),DBTMMM(3,3,16)

! Local variables
      integer                :: I,J,K,N,IJ
      real(chm_real)               :: AA(3),BB(3),CC(3),U(3),V(3),T(4,4)
      real(chm_real)               :: X(3),Y(3),Z(3),TETH(4,4)
      real(chm_real)               :: xx(3),ab(3),bc(3),ca(3)
      real(chm_real)               :: dthma(3,4,4),dthmb(3,4,4),dthmc(3,4,4)
      real(chm_real)               :: DBTMM(4,4),DBTMMB(4,4),DBTMMC(4,4)
      real(chm_real)               :: dxa(3,3),dxb(3,3),dxc(3,3), &
                                dya(3,3),dyb(3,3),dyc(3,3),  &
                                dza(3,3),dzb(3,3),dzc(3,3)
      real(chm_real)               :: daa(3,3),dd1a(3),dd1b(3),dd1c(3)
      real(chm_real)               :: drxa(3),drxb(3),drxc(3),   &
                                drza(3),drzb(3),drzc(3)
      real(chm_real)               :: dcsa(3),dcsb(3),dcsc(3),   &
                                dcpa(3),dcpb(3),dcpc(3)
      real(chm_real)               :: GRADA(4,4,3),GRADB(4,4,3),GRADC(4,4,3)
      real(chm_real)               :: DD,RA,RB,RC,CS,PFAC,RX,RY,RZ,   &
                                RA2,RB2,RC2,CS2
      real(chm_real)               :: THRFAC,RBA,RAC,RBC,D0
      real(chm_real)               :: rtemp(3)

!!! This cause compile error using xlf90
!!!   integer, SAVE          :: IR(3,3)=(/ 0,3,2,3,0,1,2,1,0 /)
!!!   real(chm_real),  SAVE        :: AF(3,3)=(/ 0.0D0, 1.0D0,-1.0D0,  
!!!  *                                    -1.0D0, 0.0D0, 1.0D0,  
!!!  *                                     1.0D0,-1.0D0, 0.0D0 /) 
!!!
      integer, SAVE          :: IR(3,3)
      real(chm_real),  SAVE        :: AF(3,3)
      DATA IR / 0,3,2,3,0,1,2,1,0 /
      DATA AF / 0.0D0, 1.0D0,-1.0D0,   &
               -1.0D0, 0.0D0, 1.0D0,    &
                1.0D0,-1.0D0, 0.0D0 /

!
! data teth/0.50,0.866,0.0,0.0,0.5,-0.2887,0.8165,0.0, 
!& 0.5,-0.2887,-0.4082,0.7071, 0.5,-0.2887,-0.4082,-0.7071/
!

      ra2 = a(1)**2+a(2)**2+a(3)**2
      rb2 = b(1)**2+b(2)**2+b(3)**2
      rc2 = c(1)**2+c(2)**2+c(3)**2
      ra = sqrt(ra2)
      rb = sqrt(rb2)
      rc = sqrt(rc2)

! do some initialization
      grada=zero
      gradb=zero
      gradc=zero

      rtemp(1)=one/ra
      rtemp(2)=one/rb
      rtemp(3)=one/rc

      aa(1:3) = a(1:3)*rtemp(1)
      bb(1:3) = b(1:3)*rtemp(2)
      cc(1:3) = c(1:3)*rtemp(3)
      u(1:3)  = bb(1:3)-aa(1:3)
      v(1:3)  = cc(1:3)-aa(1:3)

      Call ACB(u,v,x,rx)
 
      d0 = (aa(1)*x(1)+aa(2)*x(2)+aa(3)*x(3))/rx
      dd  = abs(d0)
      pfac = one
      if(d0 .GT. zero) pfac = -one

! tetrahedarl hybrid orbitals:
      cs = sqrt(dd / (one+dd) )
      cs2 = sqrt( (one - cs**2) / three )

      do i = 1,4
         teth(1:4,i)=zero
!!!           do j = 1,4
!!!              teth(j,i) = zero
!!!           end do
         teth(1,i) = cs2
      end do

      teth(1,1) =  cs

      teth(2,1) =  sqrt(one-teth(1,1)**2)

      teth(2,2) = -teth(1,1)*teth(1,2)/teth(2,1)
      teth(3,2) =  sqrt(one-teth(1,2)**2-teth(2,2)**2)

      teth(2,3) = -teth(1,1)*teth(1,3)/teth(2,1)
      teth(3,3) =-(teth(1,2)*teth(1,3)+teth(2,2)*teth(2,3))/teth(3,2)
      teth(4,3) =  sqrt(one-teth(1,3)**2-teth(2,3)**2-teth(3,3)**2)

      teth(2,4) = -teth(1,1)*teth(1,4)/teth(2,1)
      teth(3,4) =-(teth(1,2)*teth(1,4)+teth(2,2)*teth(2,4))/teth(3,2)
      teth(4,4) =-(teth(1,3)*teth(1,4)+teth(2,3)*teth(2,4) +   &
                   teth(3,3)*teth(3,4)) / teth(4,3)

!
      x(1:3) = pfac*x(1:3)
      Call ACB(x,aa,z,rz)
      Call ACB(z,x,y,ry)

      do i=1,4
         t(1,i) = zero
         t(i,1) = zero
      end do
      t(1,1) = one

      rtemp(1)=one/rx
      rtemp(2)=one/ry
      rtemp(3)=one/rz

      t(2:4,2) = x(1:3)*rtemp(1)
      t(2:4,3) = y(1:3)*rtemp(2)
      t(2:4,4) = z(1:3)*rtemp(3)

      Call ACB(bb,cc,bc,rbc)
      Call ACB(cc,aa,ca,rac)
      Call ACB(aa,bb,ab,rba)

! dai
      rtemp(1)=one/ra
      rtemp(2)=one/rb
      rtemp(3)=one/rc
      do i = 1,3
! dxj
         do j = 1,3
! dxj/dai
            dxa(j,i) = -aa(i)*(ab(j)+ca(j))*rtemp(1)
            dxb(j,i) = -bb(i)*(bc(j)+ab(j))*rtemp(2)
            dxc(j,i) = -cc(i)*(ca(j)+bc(j))*rtemp(3)
            if(j.ne.i) then
               dxa(j,i) = dxa(j,i)+af(j,i)*   &
                          ( cc(ir(j,i))-bb(ir(j,i)) )*rtemp(1)
               dxb(j,i) = dxb(j,i)+af(j,i)*   &
                          ( aa(ir(j,i))-cc(ir(j,i)) )*rtemp(2)
               dxc(j,i) = dxc(j,i)+af(j,i)*   &
                          ( bb(ir(j,i))-aa(ir(j,i)) )*rtemp(3)
            end if
         end do
      end do
      if(pfac.eq.one) then
         dxa=-dxa
         dxb=-dxb
         dxc=-dxc
      end if


! (Rx^2)'
      drxa(1:3) = two*(x(1)*dxa(1,1:3)+x(2)*dxa(2,1:3)+x(3)*dxa(3,1:3))
      drxb(1:3) = two*(x(1)*dxb(1,1:3)+x(2)*dxb(2,1:3)+x(3)*dxb(3,1:3))
      drxc(1:3) = two*(x(1)*dxc(1,1:3)+x(2)*dxc(2,1:3)+x(3)*dxc(3,1:3))

! dxj/dmi
      rtemp(1)=one/rx
      rtemp(2)=one/(rx**3)
      do i = 1,3
         grada(2:4,2,i) = dxa(1:3,i)*rtemp(1)   &
                        - half*x(1:3)*drxa(i)*rtemp(2)
         gradb(2:4,2,i) = dxb(1:3,i)*rtemp(1)   &
                        - half*x(1:3)*drxb(i)*rtemp(2)
         gradc(2:4,2,i) = dxc(1:3,i)*rtemp(1)   &
                        - half*x(1:3)*drxc(i)*rtemp(2)
      end do
              
! daaj/dai
      rtemp(1)=one/ra
      do i = 1,3
         daa(1:3,i) = -aa(1:3)*aa(i)*rtemp(1)
         daa(i,i)   =  daa(i,i)+rtemp(1)
      end do

      dza(1,1:3) = dxa(2,1:3)*aa(3)+x(2)*daa(3,1:3)   &
                 - daa(2,1:3)*x(3)-aa(2)*dxa(3,1:3)
      dza(2,1:3) = dxa(3,1:3)*aa(1)+x(3)*daa(1,1:3)   &
                 - daa(3,1:3)*x(1)-aa(3)*dxa(1,1:3)
      dza(3,1:3) = dxa(1,1:3)*aa(2)+x(1)*daa(2,1:3)   &
                 - daa(1,1:3)*x(2)-aa(1)*dxa(2,1:3)
 
      dzb(1,1:3) = dxb(2,1:3)*aa(3)-aa(2)*dxb(3,1:3)
      dzb(2,1:3) = dxb(3,1:3)*aa(1)-aa(3)*dxb(1,1:3)
      dzb(3,1:3) = dxb(1,1:3)*aa(2)-aa(1)*dxb(2,1:3)

      dzc(1,1:3) = dxc(2,1:3)*aa(3)-aa(2)*dxc(3,1:3)
      dzc(2,1:3) = dxc(3,1:3)*aa(1)-aa(3)*dxc(1,1:3)
      dzc(3,1:3) = dxc(1,1:3)*aa(2)-aa(1)*dxc(2,1:3)


! (Rz^2)'
      drza(1:3) = two*(z(1)*dza(1,1:3)+z(2)*dza(2,1:3)+z(3)*dza(3,1:3))
      drzb(1:3) = two*(z(1)*dzb(1,1:3)+z(2)*dzb(2,1:3)+z(3)*dzb(3,1:3))
      drzc(1:3) = two*(z(1)*dzc(1,1:3)+z(2)*dzc(2,1:3)+z(3)*dzc(3,1:3))
     
! dzj/dmi
      rtemp(1)=one/rz
      rtemp(2)=one/(rz**3)
      do i = 1,3
         grada(2:4,4,i) = dza(1:3,i)*rtemp(1)   &
                        - half*z(1:3)*drza(i)*rtemp(2)
         gradb(2:4,4,i) = dzb(1:3,i)*rtemp(1)   &
                        - half*z(1:3)*drzb(i)*rtemp(2)
         gradc(2:4,4,i) = dzc(1:3,i)*rtemp(1)   &
                        - half*z(1:3)*drzc(i)*rtemp(2)
      end do
 
      dya(1,1:3)= dza(2,1:3)*x(3)+z(2)*dxa(3,1:3)   &
                - dxa(2,1:3)*z(3)-x(2)*dza(3,1:3)
      dya(2,1:3)= dza(3,1:3)*x(1)+z(3)*dxa(1,1:3)   &
                - dxa(3,1:3)*z(1)-x(3)*dza(1,1:3)
      dya(3,1:3)= dza(1,1:3)*x(2)+z(1)*dxa(2,1:3)   &
                - dxa(1,1:3)*z(2)-x(1)*dza(2,1:3)

      dyb(1,1:3)= dzb(2,1:3)*x(3)+z(2)*dxb(3,1:3)   &
                - dxb(2,1:3)*z(3)-x(2)*dzb(3,1:3)
      dyb(2,1:3)= dzb(3,1:3)*x(1)+z(3)*dxb(1,1:3)   &
                - dxb(3,1:3)*z(1)-x(3)*dzb(1,1:3)
      dyb(3,1:3)= dzb(1,1:3)*x(2)+z(1)*dxb(2,1:3)   &
                - dxb(1,1:3)*z(2)-x(1)*dzb(2,1:3)

      dyc(1,1:3)= dzc(2,1:3)*x(3)+z(2)*dxc(3,1:3)   &
                - dxc(2,1:3)*z(3)-x(2)*dzc(3,1:3)
      dyc(2,1:3)= dzc(3,1:3)*x(1)+z(3)*dxc(1,1:3)   &
                - dxc(3,1:3)*z(1)-x(3)*dzc(1,1:3)
      dyc(3,1:3)= dzc(1,1:3)*x(2)+z(1)*dxc(2,1:3)   &
                - dxc(1,1:3)*z(2)-x(1)*dzc(2,1:3)

! dyj/dmi
      rtemp(1)=one/ry
      rtemp(2)=one/(rz**2)
      rtemp(3)=one/(rx**2)
      do i = 1,3
         grada(2:4,3,i) = ( dya(1:3,i) -   &
                      half*y(1:3)*(drza(i)*rtemp(2)+drxa(i)*rtemp(3))   &
                          )*rtemp(1)
         gradb(2:4,3,i) = ( dyb(1:3,i) -   &
                      half*y(1:3)*(drzb(i)*rtemp(2)+drxb(i)*rtemp(3))   &
                          )*rtemp(1)
         gradc(2:4,3,i) = ( dyc(1:3,i) -   &
                      half*y(1:3)*(drzc(i)*rtemp(2)+drxc(i)*rtemp(3))   &
                          )*rtemp(1)
      end do

! d d/dmi
! initialization
      dthma = zero
      dthmb = zero
      dthmc = zero

! MJF . Signs have been changed here!
      rtemp(1)=one/rx
      rtemp(2)=one/(rx**3)
      dd1a(1:3) = ( dxa(1,1:3)*aa(1)+dxa(2,1:3)*aa(2)+dxa(3,1:3)*aa(3)   &
                   -daa(1,1:3)*x(1) -daa(2,1:3)*x(2) -daa(3,1:3)*x(3)    &
                  )*rtemp(1)    &
                  - half*( x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3) )*   &
                    drxa(1:3)*rtemp(2)
      dd1b(1:3) = ( dxb(1,1:3)*aa(1)+dxb(2,1:3)*aa(2)+dxb(3,1:3)*aa(3)   &
                  )*rtemp(1)    &
                  - half*( x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3) )*   &
                    drxb(1:3)*rtemp(2)
      dd1c(1:3) = ( dxc(1,1:3)*aa(1)+dxc(2,1:3)*aa(2)+dxc(3,1:3)*aa(3)   &
                  )*rtemp(1)    &
                  - half*( x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3) )*   &
                    drxc(1:3)*rtemp(2)

      rtemp(1) =(one/teth(1,1)-teth(1,1))*(teth(2,1)**2)
      dcsa(1:3)=-half*rtemp(1)*dd1a(1:3)
      dcsb(1:3)=-half*rtemp(1)*dd1b(1:3)
      dcsc(1:3)=-half*rtemp(1)*dd1c(1:3)
      dcpa(1:3)= half*dd1a(1:3)*teth(2,1)**3
      dcpb(1:3)= half*dd1b(1:3)*teth(2,1)**3
      dcpc(1:3)= half*dd1c(1:3)*teth(2,1)**3

!
      thrfac = one/sqrt(three)

      dthma(1:3,1,1) = dcsa(1:3)
      dthma(1:3,2,1) = dcpa(1:3)
      dthma(1:3,1,2) = dcpa(1:3)*thrfac
      dthma(1:3,1,3) = dcpa(1:3)*thrfac
      dthma(1:3,1,4) = dcpa(1:3)*thrfac
      dthma(1:3,2,2) =-dcsa(1:3)*thrfac
      dthma(1:3,2,3) =-dcsa(1:3)*thrfac
      dthma(1:3,2,4) =-dcsa(1:3)*thrfac

      dthmb(1:3,1,1) = dcsb(1:3)
      dthmb(1:3,2,1) = dcpb(1:3)
      dthmb(1:3,1,2) = dcpb(1:3)*thrfac
      dthmb(1:3,1,3) = dcpb(1:3)*thrfac
      dthmb(1:3,1,4) = dcpb(1:3)*thrfac
      dthmb(1:3,2,2) =-dcsb(1:3)*thrfac
      dthmb(1:3,2,3) =-dcsb(1:3)*thrfac
      dthmb(1:3,2,4) =-dcsb(1:3)*thrfac

      dthmc(1:3,1,1) = dcsc(1:3)
      dthmc(1:3,2,1) = dcpc(1:3)
      dthmc(1:3,1,2) = dcpc(1:3)*thrfac
      dthmc(1:3,1,3) = dcpc(1:3)*thrfac
      dthmc(1:3,1,4) = dcpc(1:3)*thrfac
      dthmc(1:3,2,2) =-dcsc(1:3)*thrfac
      dthmc(1:3,2,3) =-dcsc(1:3)*thrfac
      dthmc(1:3,2,4) =-dcsc(1:3)*thrfac

! Collect
      do j=1,4
         do i=1,4
            bt(i,j) = zero
            do k=1,4
               bt(i,j) = bt(i,j)+t(i,k)*teth(k,j)
            end do
            btm(j,i)=bt(i,j)
         end do
      end do

! Derivatives
      do n=1,3
         do j=1,4
            do i=1,4
               dbtmm(i,j) = zero
               dbtmmb(i,j)= zero
               dbtmmc(i,j)= zero
! MJF . Lines have been uncommented!
               do k = 1,4
                  dbtmm(i,j)  = dbtmm(i,j)  + grada(i,k,n)*teth(k,j)   &
                                            + t(i,k)*dthma(n,k,j)
                  dbtmmb(i,j) = dbtmmb(i,j) + gradb(i,k,n)*teth(k,j)   &
                                            + t(i,k)*dthmb(n,k,j)
                  dbtmmc(i,j) = dbtmmc(i,j) + gradc(i,k,n)*teth(k,j)   &
                                            + t(i,k)*dthmc(n,k,j)
               end do
            end do
         end do
 
!
         ij=0
         do i = 1,4
            do j = 1,4
               ij=ij+1
               dbtmmm(n,1,ij) = dbtmm(i,j)
               dbtmmm(n,2,ij) = dbtmmb(i,j)
               dbtmmm(n,3,ij) = dbtmmc(i,j)
            end do
         end do
      end do

      return
      END SUBROUTINE HBDRIV


      SUBROUTINE ACB(A,B,C,S)
!
!

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      real(chm_real), intent(in)  :: A(3),B(3)
      real(chm_real), intent(out) :: C(3),S

! Local variables

      C(1) = A(2)*B(3)-B(2)*A(3)
      C(2) = B(1)*A(3)-A(1)*B(3)
      C(3) = A(1)*B(2)-B(1)*A(2)
      S    = SQRT(C(1)**2+C(2)**2+C(3)**2)

      return
      END SUBROUTINE ACB


      SUBROUTINE FTOFHB(F,Fhb,Bt,Natqm,Nqmlnk,Norbs,Orb_loc)
!
! Tansform a full Fock matrix in AO basis into active HO basis
! 
! On input
!    F       : Fock matrix in AO, lower triangle
!    Bt      : Transformation matrix for each GHO boundary atom, 
!              (4x4,Nqmlnk)
!    Natqm   : Number of QM atoms
!    Nqmlnk  : Number of GHO boundary atoms
!    Norbs   : Number of AOs
!    Orb_loc : Location of each orbital, such that 
!              Orb_loc(1,i): start of ith orbital
!              Orb_loc(2,i): end of ith orbital
!
! On output
!    Fhb     : Fock matrix in HO, include only active orbitals


  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: Natqm,Nqmlnk,Norbs,Orb_loc(2,*)
      real(chm_real),intent(in)    :: F(*),Bt(16,*)
      real(chm_real),intent(inout) :: Fhb(*)

! Local variables
      integer                :: i,j,k,ii,jj,ij,ia,ib,ja,jb,I1
      integer                :: L,IAMONE,INDF,IAL
      real(chm_real)               :: FTMP(10),FTMP2(10)

      logical                :: FIRST=.TRUE.
      integer                :: NACTATM,NORBAO,LIN1
!     logical, SAVE          :: FIRST=.TRUE.
!     integer, SAVE          :: NACTATM,NORBAO,LIN1

! compute the following every step..(instead of saving them.)      
!     If(FIRST) then
         NORBAO = Norbs - 4*Nqmlnk
         LIN1   = NORBAO*(NORBAO+1)/2
         NACTATM= Natqm - Nqmlnk
!        FIRST  =.FALSE.
!     End if

! Core part not affected
      Fhb(1:LIN1) = F(1:LIN1)

! Loop over GHO boundary atoms for orbitals to be transformed
      Do i=1,Nqmlnk
         i1     = NORBAO+i
         i1     = i1*(i1-1)/2
         ii     = NACTATM+i
         ia     = Orb_loc(1,ii)
         ib     = Orb_loc(2,ii)
         IAMONE = ia-1
         ij     = i1

         do j=1,NORBAO                           ! F(mu,l), AO-HO block
            ij = ij + 1                          ! Only one active HO 
                                                 ! per QM-boundary atom
            Fhb(ij) = zero
            do k=ia,ib
               Fhb(ij) = Fhb(ij)+Bt(k-ia+1,i)*F(j+k*(k-1)/2)
            end do
         end do

         do j=1,i-1                         ! F(l,l'), HO-other HO block
            jj      = NACTATM+j
            ja      = Orb_loc(1,jj)
            ij      = ij +1
            Fhb(ij) = zero
            do L=1,4
               FTMP(L)=zero
               IAL    =ia+L-1
               INDF   =ja-1+IAL*(IAL-1)/2
               do k=1,4
                  FTMP(L) = FTMP(L)+Bt(k,j)*F(INDF+K)
               end do

               Fhb(ij) = Fhb(ij) + Bt(L,i)*FTMP(L)
            end do
         end do

         L = 0                              ! F(l,l), HO-HO corner block
         do j=ia,ib
            ja = j*(j-1)/2
            do k = ia, j
               L = L+1
               FTMP(L)  = F(k+ja)
               FTMP2(L) = zero
            end do
         end do

         Call VBFTN(FTMP,FTMP2,Bt(1:16,I),4)
         Fhb(ij+1) = FTMP2(1)
      End do

      return
      END SUBROUTINE FTOFHB


      SUBROUTINE VBFTN(F1,F1vb,X,Ndim)
!
!
!

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: Ndim
      real(chm_real),intent(in)    :: F1(*), X(NDIM,NDIM)
      real(chm_real),intent(inout) :: F1vb(*)

! Local variables
      integer                :: i,j,k,L,L1,L2
      real(chm_real)               :: fac

      L1=0
      Do i=1,Ndim
         do j=1,i
            L1      = L1+1
            L2      = 0
            F1vb(L1)= zero
            do k=1,Ndim
               do L=1,k-1
                  L2      = L2+1
                  fac     = x(k,i)*x(L,j) + x(k,j)*x(L,i)
                  F1vb(L1)= F1vb(L1) + fac*F1(L2)
               end do
               L2      = L2+1
               fac     = x(k,i)*X(k,j)
               F1vb(L1)= F1vb(L1) + fac*F1(L2)
            end do
         end do
      End do

      return
      END SUBROUTINE VBFTN


      SUBROUTINE gho_densit_expansion(nqmlnk,mqm16,norbs,QMATMQ,Bt,Btm, &
                                   CAHB,C,PAHB,P,PAOLD)
!
! Density expansion from GHO boundary atoms
!

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)      :: nqmlnk, norbs, mqm16
      real(chm_real),  intent(in)    :: QMATMQ(nqmlnk), Bt(mqm16*nqmlnk),   &
                                  Btm(mqm16*nqmlnk)
      real(chm_real),  intent(in)    :: CAHB(norbs*norbs)
      real(chm_real),  intent(inout) :: C(norbs*norbs)
      real(chm_real),  intent(inout) :: PAHB(norbs*(norbs+1)/2),   &
                                  P(norbs*(norbs+1)/2), &
                                  PAOLD(norbs*(norbs+1)/2)

! Local variables
      integer   :: i, j, ii, jj, i1, j1, ij, k, L, kk1, kk2, kk3
      integer   :: norbhb, iorb1b, naos, iqatm, Linao
      integer   :: norbs2, norbhb2
      real(chm_real)  :: xbt1


! local variables 
      norbhb = norbs - 3*nqmlnk
      naos   = norbhb - nqmlnk
      Linao  =(naos * (naos + 1)) / 2
      norbs2 = norbs*(norbs+1)/2

! for DIIS
      norbhb2= (norbhb*(norbhb+1))/2
      paold  = zero
      paold(1:norbhb2) = PAHB(1:norbhb2)

!! It's been moved int qm2_scf routine
!! Transform orbitals to AO basis, used by Mulliken analysis
!! Call CTRASF(norbs,nqmlnk,mqm16,Bt,CAHB,C)
     
!
!*** GHO expansion I: RHF total or UHF alpha density**************
!  
! Relocate the positions of active hybrid orbitals if there are  
! more than one GHO boundary atom.
      Do i=norbhb,naos+2,-1
         iqatm = i-naos
         ii    = i*(i-1)/2
         iorb1b= naos + 4*(iqatm-1)
         jj    = iorb1b*(iorb1b+1)/2

! or maybe
! PAHB(jj+1:jj+naos) = PAHB(ii+1,ii+naos)
! PAHB(ii+1:ii+naos) = zero
         do j=1,naos
            ii = ii+1
            jj = jj+1
            PAHB(jj) = PAHB(ii)
            PAHB(ii) = zero
         end do

! HB-HB blocks
         do j=1,iqatm
            ii=ii+1
            jj=jj+1
            PAHB(jj) = PAHB(ii)
            PAHB(ii) = zero
            if(j.ne.iqatm) then
!   PAHB(jj+1:jj+3) = zero
               PAHB(jj+1) = zero
               PAHB(jj+2) = zero
               PAHB(jj+3) = zero
               jj = jj+3
            end if
         end do

! The rest three auxiliary orbitals
         do j=2,4
            PAHB(jj+1:jj+iorb1b+j)= zero
            jj                    = jj+iorb1b+j
            PAHB(jj)              = one-third*QMATMQ(iqatm)
         end do
      End do

! Auxiliary density for the first GHO boundary atom
      k = naos + 2
      L = naos + 4
      Do i=k,L
         jj = i*(i-1)/2
         PAHB(jj+1:jj+i) = zero
         PAHB(jj+i)      = one-third*QMATMQ(1)
      End do

! AO blocks, Not affected by orbital transformation
      P(1:Linao) = PAHB(1:Linao)

! Loop ver GHO boundary atoms
      ij = Linao
      Do k=1,nqmlnk
         j1     = mqm16*(k-1)
         i1     = naos+4*(k-1)
         iorb1b = i1*(i1+1)/2
         
! Boundary atom must be non-hydrogen atoms
! Thus, for each boundary atom, there are 4 AOs
         L = 0                               
         do i=1,4
 
! Since only one hybrid-orbital density on the
! boundary atom is non-zero, sum over orbitals
! is not needed
            xbt1 = Bt(j1+i)              ! Btm(j1+4*(i-1)+1)
!
! P(ij+1:ij+naos) = PAHB(iorb1b+1:naos)*xbt1
! ij   = ij+naos
            do j=1,naos
               ij    = ij+1
               P(ij) = PAHB(iorb1b+j)*xbt1
            end do

! Boundary atom-other boundary atom block
            do L=1,k-1
               kk1 = mqm16*(L-1)
               kk3 = iorb1b+naos+4*(L-1)+1
! P(ij+1:ij+4)=PAHB(kk3)*xbt1*Bt(kk1+1:kk1+4)   
! Btm(kk1+4*(j-1)+1)
! ij=ij+4
               do j=1,4
                  ij   = ij+1
                  P(ij)= PAHB(kk3)*xbt1*Bt(kk1+j)
               end do
            end do

! Boundary atom self block
            kk1 = 4*(i-1)+j1
            do j=1,i
               ij    = ij+1
               kk2   = 4*(j-1) + j1
               P(ij) = zero
               do L=1,4
                  KK3   =(i1+L)*(i1+L+1)/2
                  P(ij) = P(ij)+PAHB(kk3)*Btm(kk1+L)*Btm(KK2+L)
               end do
            end do
         end do
      End do

      return
      END SUBROUTINE gho_densit_expansion


      SUBROUTINE  CTRASF(norbs,nqmlnk,mqm16,Bt,CHB,C)
!
!
!

  use chm_kinds
  use qm2_double
  use qm2_constants

      implicit none

! Passed in
      integer, intent(in)    :: norbs,nqmlnk,mqm16
      real(chm_real),  intent(in)    :: Bt(mqm16*nqmlnk), CHB(norbs*norbs)
      real(chm_real),  intent(inout) :: C(norbs*norbs)

! Local variables
      integer                :: i,j,i1,j1,ij,mqm15
      integer                :: norbhb, naos


      norbhb = norbs - 3*nqmlnk
      naos   = norbhb - nqmlnk
      mqm15  = mqm16-1

      Do i=1,norbhb
         i1 = norbs*(i-1)
         j1 = norbhb*(i-1)

! Upto normal naos
         C(i1+1:i1+naos) = CHB(i1+1:i1+naos)

         i1 = i1+naos
         j1 = j1+naos

         do j=naos+1,norbhb
            ij = mqm16*(j-naos) - mqm15
            C(i1+1:i1+4)=CHB(j1+1:j1+4)*Bt(ij)

            i1 = i1+4
            j1 = j1+4
         end do
      End do

! Append and transform auxiliary hybrid orbitals
      Do i=1,nqmlnk
         do j=1,3
            ij = norbs*(norbhb+i*j-1)
            ij = mqm16*(i-1) + 4*j
            do j1 = 1,norbs
               i1 = i1+1
               if( j1.gt.(naos+(i-1)*4) .and. j1.le.(naos+i*4) ) then
                  ij    = ij+1
                  C(i1) = Bt(ij)
               else
                  C(i1) = zero
               end if
            end do
         end do
      End do

      return
      END SUBROUTINE CTRASF


      SUBROUTINE DQLINK4(Dx,Dy,Dz,X,Y,Z,CG,Fock,UHF,Eclass,grad_fct)
!
! Computes the derivatives of the density matrix as a result of 
! the transformation from hybrid orbitals to atomic orbitals
! basis. Updates nucleus force arrays, and computes a classical
! energy contribution from MM atoms directly connected to the
! QM boundary atom.
!
! References:
!   J. Gao, P. Amara, C. Alhambra, M. J. Field J. Phys. Chem. A, 
!   102, 4714 (1998).
!

  use qmmm_module, only : qm2_struct, qm2_ghos

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

      implicit none

! Passed in
      real(chm_real),intent(inout) :: Dx(*), Dy(*), Dz(*)
      real(chm_real),intent(in)    :: X(*), Y(*), Z(*), CG(*), Fock(*)
      real(chm_real),intent(inout) :: Eclass
      real(chm_real),intent(in)    :: grad_fct
      logical, intent(in)    :: UHF

! Local variables
      integer                :: i,j,k,L,m,n,ii,jj,LL,mm,i1,L1,m1,ij
      integer                :: kk1,kk2,kk3,ii2,jj2,n16,n16i1,n16j
      integer                :: norbs,naos,Linao
      integer                :: i1diag(4)

      real(chm_real)               :: xdta(4,4),ydta(4,4),zdta(4,4)
      real(chm_real)               :: xdtb(4,4),ydtb(4,4),zdtb(4,4)
      real(chm_real)               :: xdtc(4,4),ydtc(4,4),zdtc(4,4)
      real(chm_real)               :: tinv(4,4)
      real(chm_real)               :: dmmxyz(3,3)
      real(chm_real)               :: dL2xyz(3,3)
      real(chm_real)               :: dmmxyz1(3,3)
      real(chm_real)               :: pf, pf1, pf2, xfac,rr1,rr2,fact1,Confac
      real(chm_real)               :: dLtmp(3), delxyz(3)

!! real(chm_real)                 :: dmmxa, dmmya, dmmza        (1,1:3)
!! real(chm_real)                 :: dmmxb, dmmyb, dmmzb        (2,1:3)
!! real(chm_real)                 :: dmmxc, dmmyc, dmmzc        (3,1:3)

!! real(chm_real)                 :: dL2xa, dL2ya, dL2za        (1:3,1)
!! real(chm_real)                 :: dL2xb, dL2yb, dL2zb        (1:3,2)
!! real(chm_real)                 :: dL2xc, dL2yc, dL2zc        (1:3,3)

!! real(chm_real)                 :: dmmxa1, dmmya1, dmmza1     (1,1:3)
!! real(chm_real)                 :: dmmxb1, dmmyb1, dmmzb1     (2,1:3)
!! real(chm_real)                 :: dmmxc1, dmmyc1, dmmzc1     (3,1:3)

! get the number of orbitals
      norbs = qm2_ghos%norbsgho
      naos  = norbs - 4*qm2_ghos%nqmlnk
      ij    = naos*(naos+1)/2
      n16   = 0

      Confac = EV_TO_KCAL*grad_fct

!*** Main Loop ***
! Loop over GHO boundary atoms
      Do i=1,qm2_ghos%nqmlnk
         L1    = 4*(i-1)
         ii    = qm2_ghos%IQLINK(i)
         k     = naos + 4*(i-1) + 1                   ! Orb_loc(1,?)
         L     = k + 3                                ! Orb_loc(2,?)
         Linao = k*(k-1)/2

! locate position of diagonal elements
         do j=k,L
            i1diag(j-k+1) = j*(j+1)/2
         end do

! initialize temporary variables
         dmmxyz = zero
         n16j   = n16
!do j=1,4
!   tinv(1:4,j) = qm2_ghos%Btm(n16j+1:n16j+4)
!   xdta(1:4,j) = qm2_ghos%Dbtmmm(1,1,n16j+1:n16j+4)
!   ydta(1:4,j) = qm2_ghos%Dbtmmm(2,1,n16j+1:n16j+4)
!   zdta(1:4,j) = qm2_ghos%Dbtmmm(3,1,n16j+1:n16j+4)
!   xdtb(1:4,j) = qm2_ghos%Dbtmmm(1,2,n16j+1:n16j+4)
!   ydtb(1:4,j) = qm2_ghos%Dbtmmm(2,2,n16j+1:n16j+4)
!   zdtb(1:4,j) = qm2_ghos%Dbtmmm(3,2,n16j+1:n16j+4)
!   xdtc(1:4,j) = qm2_ghos%Dbtmmm(1,3,n16j+1:n16j+4)
!   ydtc(1:4,j) = qm2_ghos%Dbtmmm(2,3,n16j+1:n16j+4)
!   zdtc(1:4,j) = qm2_ghos%Dbtmmm(3,3,n16j+1:n16j+4)
!
!   n16j = n16j + 4
!end do
         do j=1,4
            do k=1,4
               n16j        = n16j+1
               tinv(k,j) = qm2_ghos%Btm(n16j)
               xdta(k,j) = qm2_ghos%Dbtmmm(1,1,n16j)
               ydta(k,j) = qm2_ghos%Dbtmmm(2,1,n16j)
               zdta(k,j) = qm2_ghos%Dbtmmm(3,1,n16j)
               xdtb(k,j) = qm2_ghos%Dbtmmm(1,2,n16j)
               ydtb(k,j) = qm2_ghos%Dbtmmm(2,2,n16j)
               zdtb(k,j) = qm2_ghos%Dbtmmm(3,2,n16j)
               xdtc(k,j) = qm2_ghos%Dbtmmm(1,3,n16j)
               ydtc(k,j) = qm2_ghos%Dbtmmm(2,3,n16j)
               zdtc(k,j) = qm2_ghos%Dbtmmm(3,3,n16j)
            end do
         end do

! AO-HB blocks
         do j=1,4
            do k=1,naos
               ij = ij + 1
               pf = two * qm2_ghos%Pho(Linao+k) * Fock(ij)

! For UHF case
               if(UHF) then
!! temporary commented out till UHF implemented
!!                pf = two * qm2_ghos%Pbho(Linao+k) * Fock(ij)
               end if

               dmmxyz(1,1) = dmmxyz(1,1) + xdta(1,j)*pf         ! x axis
               dmmxyz(2,1) = dmmxyz(2,1) + xdtb(1,j)*pf
               dmmxyz(3,1) = dmmxyz(3,1) + xdtc(1,j)*pf

               dmmxyz(1,2) = dmmxyz(1,2) + ydta(1,j)*pf         ! y axis
               dmmxyz(2,2) = dmmxyz(2,2) + ydtb(1,j)*pf
               dmmxyz(3,2) = dmmxyz(3,2) + ydtc(1,j)*pf

               dmmxyz(1,3) = dmmxyz(1,3) + zdta(1,j)*pf         ! zaxis
               dmmxyz(2,3) = dmmxyz(2,3) + zdtb(1,j)*pf
               dmmxyz(3,3) = dmmxyz(3,3) + zdtc(1,j)*pf
            end do

! HB-other HB blocks
            m1 = Linao + naos
            do i1 = 1, i-1
               n16i1  = qm2_ghos%mqm16 * (i1-1)
               dL2xyz = zero

               kk3 = m1 + 4*(i1-1) + 1
               do L1 = 1,4
                  ij = ij +1
                  pf = two * qm2_ghos%Pho(kk3) * Fock(ij)

! For UHF case
                  if(UHF) then
!! temporary commented out till UHF implemented
!!                   pf = two * qm2_ghos%Pbho(kk3) * Fock(ij)
                  end if

                  pf1 = qm2_ghos%Bt(n16+j)*pf
                  pf2 = qm2_ghos%Bt(n16i1+L1)*pf

                  dmmxyz(1,1) = dmmxyz(1,1) + xdta(1,j)*pf2     ! x axis
                  dmmxyz(2,1) = dmmxyz(2,1) + xdtb(1,j)*pf2
                  dmmxyz(3,1) = dmmxyz(3,1) + xdtc(1,j)*pf2

                  dmmxyz(1,2) = dmmxyz(1,2) + ydta(1,j)*pf2     ! y axis
                  dmmxyz(2,2) = dmmxyz(2,2) + ydtb(1,j)*pf2
                  dmmxyz(3,2) = dmmxyz(3,2) + ydtc(1,j)*pf2

                  dmmxyz(1,3) = dmmxyz(1,3) + zdta(1,j)*pf2     ! z axis
                  dmmxyz(2,3) = dmmxyz(2,3) + zdtb(1,j)*pf2
                  dmmxyz(3,3) = dmmxyz(3,3) + zdtc(1,j)*pf2

                  kk1 = n16i1 + 4*(L1-1) + 1
                  dL2xyz(1:3,1) = dL2xyz(1:3,1) +      & ! x,y,z axis for a
                                  pf1*qm2_ghos%Dbtmmm(1:3,1,kk1)  
                  dL2xyz(1:3,2) = dL2xyz(1:3,2) +      & ! x,y,z axis for b
                                  pf1*qm2_ghos%Dbtmmm(1:3,2,kk1) 
                  dL2xyz(1:3,3) = dL2xyz(1:3,3) +      & ! x,y,z axis for c
                                  pf1*qm2_ghos%Dbtmmm(1:3,3,kk1)
               end do

               ii2   = qm2_ghos%IQLINK(i1)
               jj    = qm2_ghos%JQLINK(1,i1)
               LL    = qm2_ghos%JQLINK(2,i1)
               mm    = qm2_ghos%JQLINK(3,i1)
               dLtmp = zero

               Dx(jj) = Dx(jj) - dL2xyz(1,1)*Confac
               Dy(jj) = Dy(jj) - dL2xyz(2,1)*Confac
               Dz(jj) = Dz(jj) - dL2xyz(3,1)*Confac
               dLtmp(1:3) = dL2xyz(1:3,1)

               Dx(LL) = Dx(LL) - dL2xyz(1,2)*Confac
               Dy(LL) = Dy(LL) - dL2xyz(2,2)*Confac
               Dz(LL) = Dz(LL) - dL2xyz(3,2)*Confac
               dLtmp(1:3) = dLtmp(1:3) + dL2xyz(1:3,2)

               Dx(mm) = Dx(mm) - dL2xyz(1,3)*Confac
               Dy(mm) = Dy(mm) - dL2xyz(2,3)*Confac
               Dz(mm) = Dz(mm) - dL2xyz(3,3)*Confac
               dLtmp(1:3) = dLtmp(1:3) + dL2xyz(1:3,3)

               dLtmp  = dLtmp*Confac
               Dx(ii2)= Dx(ii2)+dLtmp(1)
               Dy(ii2)= Dy(ii2)+dLtmp(2)
               Dz(ii2)= Dz(ii2)+dLtmp(3)
            end do

! HB-HB blocks
            do k=1,j
               ij   = ij + 1
               xfac = two
               if(k.eq.j) xfac = one
  
               do L =1,4
                  pf = xfac * qm2_ghos%Pho(i1diag(L)) * Fock(ij)

! For UHF case
                  if(UHF) then
!! temporary commented out till UHF implemented
!!                   pf = xfac * qm2_ghos%Pbho(i1diag(L)) * Fock(ij)
                  end if

                  dmmxyz(1,1) = dmmxyz(1,1) + pf*(xdta(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*xdta(L,k))
                  dmmxyz(2,1) = dmmxyz(2,1) + pf*(xdtb(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*xdtb(L,k))
                  dmmxyz(3,1) = dmmxyz(3,1) + pf*(xdtc(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*xdtc(L,k))

                  dmmxyz(1,2) = dmmxyz(1,2) + pf*(ydta(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*ydta(L,k))
                  dmmxyz(2,2) = dmmxyz(2,2) + pf*(ydtb(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*ydtb(L,k))
                  dmmxyz(3,2) = dmmxyz(3,2) + pf*(ydtc(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*ydtc(L,k))

                  dmmxyz(1,3) = dmmxyz(1,3) + pf*(zdta(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*zdta(L,k))
                  dmmxyz(2,3) = dmmxyz(2,3) + pf*(zdtb(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*zdtb(L,k))
                  dmmxyz(3,3) = dmmxyz(3,3) + pf*(zdtc(L,j)*tinv(L,k)   &
                                            + tinv(L,j)*zdtc(L,k))
               end do
            end do
         end do

! conversion factor
         dmmxyz1 = dmmxyz * Confac

! add gradient
         jj    = qm2_ghos%JQLINK(1,i)
         LL    = qm2_ghos%JQLINK(2,i)
         mm    = qm2_ghos%JQLINK(3,i)
         dLtmp = zero

         Dx(jj)   = Dx(jj) - dmmxyz1(1,1)
         Dx(LL)   = Dx(LL) - dmmxyz1(2,1)
         Dx(mm)   = Dx(mm) - dmmxyz1(3,1)
         dLtmp(1) = dmmxyz1(1,1)+dmmxyz1(2,1)+dmmxyz1(3,1)

         Dy(jj)   = Dy(jj) - dmmxyz1(1,2)
         Dy(LL)   = Dy(LL) - dmmxyz1(2,2)
         Dy(mm)   = Dy(mm) - dmmxyz1(3,2)
         dLtmp(2) = dmmxyz1(1,2)+dmmxyz1(2,2)+dmmxyz1(3,2)

         Dz(jj)   = Dz(jj) - dmmxyz1(1,3)
         Dz(LL)   = Dz(LL) - dmmxyz1(2,3)
         Dz(mm)   = Dz(mm) - dmmxyz1(3,3)
         dLtmp(3) = dmmxyz1(1,3)+dmmxyz1(2,3)+dmmxyz1(3,3)

         Dx(ii)   = Dx(ii) + dLtmp(1)
         Dy(ii)   = Dy(ii) + dLtmp(2)
         Dz(ii)   = Dz(ii) + dLtmp(3)

         n16 = n16 + qm2_ghos%mqm16
!
! End of electron interactions


! 
! Include Nuclear interactions between MM-boundary atoms
!
! Avoid double counting in Beta correction for UHF case
         If(.not.UHF) then
            do m = 1,2
               jj = qm2_ghos%JQLINK(m,i)
               m1 = m+1

               do n = m1,3
                  jj2 = qm2_ghos%JQLINK(n,i)
                  delxyz(1) = x(jj2) - x(jj)
                  delxyz(2) = y(jj2) - y(jj)
                  delxyz(3) = z(jj2) - z(jj)
                  rr2  = delxyz(1)**2+delxyz(2)**2+delxyz(3)**2
                  rr1  = SQRT(RR2)

                  fact1 = CCELEC*CG(jj)*CG(jj2) / rr1
                  Eclass = Eclass + fact1

                  fact1  = fact1/rr2
                  dLtmp(1:3) = delxyz(1:3)*fact1*grad_fct

                  dx(jj) = dx(jj) + dLtmp(1)
                  dx(jj2)= dx(jj2)- dLtmp(1)

                  dy(jj) = dy(jj) + dLtmp(2)
                  dy(jj2)= dy(jj2)- dLtmp(2)

                  dz(jj) = dz(jj) + dLtmp(3)
                  dz(jj2)= dz(jj2)- dLtmp(3)
               end do

            end do
         End if
      End do
!*** End Main Loop ***

      return
      END SUBROUTINE DQLINK4


      SUBROUTINE DQLINK4_Ener(X,Y,Z,CG,Fock,UHF,Eclass)
!
! Compute a classical energy contribution from MM atoms
! directly connected to the QM boundary atom.
!
! Local copy of subroutine DQLINK4.
!
! References:
!   J. Gao, P. Amara, C. Alhambra, M. J. Field J. Phys. Chem. A,
!   102, 4714 (1998).
!

  use qmmm_module, only : qm2_struct, qm2_ghos

  use chm_kinds

  use qm2_double
  use qm2_constants
  use qm2_conversions

      implicit none

! Passed in
      real(chm_real),intent(in)    :: X(*), Y(*), Z(*), CG(*), Fock(*)
      real(chm_real),intent(inout) :: Eclass
      logical, intent(in)    :: UHF

! Local variables
      integer                :: i,m,n,jj,jj2,m1
      real(chm_real)               :: delxyz(3),rr1,rr2,fact1

!*** Main Loop ***
! Loop over GHO boundary atoms
! Include Nuclear interactions between MM-boundary atoms
! Avoid double counting in Beta correction for UHF case
      If(.not. UHF) then
         Do i=1,qm2_ghos%nqmlnk
            do m = 1,2
               jj = qm2_ghos%JQLINK(m,i)
               m1 = m+1

               do n = m1,3
                  jj2 = qm2_ghos%JQLINK(n,i)
                  delxyz(1) = x(jj2) - x(jj)
                  delxyz(2) = y(jj2) - y(jj)
                  delxyz(3) = z(jj2) - z(jj)
                  rr2  = delxyz(1)**2+delxyz(2)**2+delxyz(3)**2
                  rr1  = SQRT(RR2)

                  fact1 = CCELEC*CG(jj)*CG(jj2) / rr1
                  Eclass = Eclass + fact1

                  fact1  = fact1/rr2
               end do
            end do
         End do
      End if
!*** End Main Loop ***

      return
      END SUBROUTINE DQLINK4_Ener

      SUBROUTINE Get_Charge_GHO(natqm_1,qminb1_dual,cginb,Qdual_check)
!
! copy charge on gho atoms to cginb array.
!
  use qmmm_module, only : qm2_ghos,qm2_ghos_r

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer, intent(in)    :: natqm_1,qminb1_dual(*)
      real(chm_real),intent(inout) :: cginb(*)
      logical, intent(in)    :: Qdual_check(2)

! Local variables
      integer                :: i,j,ii,jj

! for dual quantum region
      if(Qdual_check(1)) then
         Do i=1,qm2_ghos_r%nqmlnk
            ii = qm2_ghos_r%IQLINK(i)
            do j=1,natqm_1
               jj=iabs(qminb1_dual(j))
               if(ii.eq.jj) cginb(j) = qm2_ghos_r%QMATMQ(i)
            end do
         End do
      else 
         Do i=1,qm2_ghos%nqmlnk
            ii = qm2_ghos%IQLINK(i)
            do j=1,natqm_1
               jj=iabs(qminb1_dual(j))
               if(ii.eq.jj) cginb(j) = qm2_ghos%QMATMQ(i)
            end do
         End do
      end if

      return
      END SUBROUTINE Get_Charge_GHO

      SUBROUTINE Get_CG_GHO(CG,Qdual_check)
!
! copy charge on gho atoms to CG array.
!
  use qmmm_module, only : qm2_ghos,qm2_ghos_r,qm2_ghos_p

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      real(chm_real),intent(inout) :: CG(*)
      logical, intent(in)    :: Qdual_check(2)

! Local variables
      integer                :: i,j,ii,jj

! for dual quantum region
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then             ! get 1st gho charge for 
            Do i=1,qm2_ghos_r%nqmlnk         ! 2nd qm
               ii     = qm2_ghos_r%IQLINK(i)
               CG(ii) = qm2_ghos_r%QMATMQ(i)
            End do
         else
            Do i=1,qm2_ghos_p%nqmlnk         ! get 2nd gho charge for
               ii     = qm2_ghos_p%IQLINK(i) ! 1st qm
               CG(ii) = qm2_ghos_p%QMATMQ(i)
            End do
         end if
      else
         Do i=1,qm2_ghos%nqmlnk
            ii     = qm2_ghos%IQLINK(i)
            CG(ii) = qm2_ghos%QMATMQ(i)
         End do
      end if

      return
      END SUBROUTINE Get_CG_GHO

#endif /* (mainsquatn)*/

SUBROUTINE GHOHYB_BLANK
  !
  ! dummy routine for compilation
  !
  RETURN
END SUBROUTINE GHOHYB_BLANK

