!     SUBROUTINE EGLCAO
!     =================
!
!     Copyright 1997 by Peter Blaudeck, Dirk Porezag, Michael Haugk,
!                       Joachim Elsner
!
! *********************************************************************
!
!     PROGRAM CHARACTERISTICS
!     -----------------------
!
! eglcao calculates energy and gradient for dylcao as shown by Seifert. 
! The determination of the occupation numbers has been changed to be
! also valid for metallic systems.
!
! PARAMETERS:
! n3      i  3 * number of atoms
! telec   r  electronic temperature (for Fermi distribution)
! x       r  coordinates (n3)
! scftol  r  convergence criterion for scf (if negative, no scf)
! atomic  l  should there be an atomic energy calculation
! e       r  total energy
! eel     r  electronic energy
! gr      r  gradient (n3)
! miter   i  number of scf-iterations performed
! evector l  prints out the eigenvectors
! lgrad   l  should gradients be calculated?
!
! *********************************************************************
!
!      subroutine eglcao(n3,telec,x,scftol,atomic,e,eel,gr,
!    &                  miter,evector,qmat,lgrad,
       subroutine eglcao(n3,      x,       atomic,e,eel,gr,
     &                                     lgrad,
     &                  qsccewc,qsccews,qscqewc,qscqews,
     &                  erfct,del,pts,
! added for GHO ... PJ 7/2004
     &                  pho,fao,sao,who,fhb,shb,chb,cao)
!      =================
!
! maxima definitions
!
       use sccdftbsrc, miter=>mxitscf,izp=>izp2,evector=>lprvec
       use sccdftb, xe=>cptc, ze=>zptc, ext=>extflag,ne=>nptc
       use sccgsbp
       use sccpb
       use chm_kinds
! QC: control output
       use stream
#if KEY_DFTBMKL==1
       use cpe, only: cpe_scf, cpe_grad
#endif
       use dftd3, only: calc_e_dftd3,
     &                  calc_g_dftd3

       implicit REAL*8 (A-H,O-Z)
!      include 'maxima.inc'
!
       logical doscf, atomic, lgrad
       real*8 x(3,*), gr(3,*)
       real*8 eel, ecoul, e, efermi
!      QC: UW0406 add ewald related
       real*8  qsccewc(*),qsccews(*)
       real*8  qscqewc(*),qscqews(*)
!      xiao: add erfc for ewald
       real*8  erfct(*)
       real*8  del 
       integer pts  

       external skhpar,skspar

!      integer izp(NNDIM), ind(NNDIM+1),lmax(MAXTYP)
       integer             ind(NNDIM+1)
       integer indj,indk,indj1,indk1,nstart
!      integer conat(NNDIM+1),cai
!      integer numint(MAXTYP,MAXTYP)
!      real*8 coeff(6,MAXINT,MAXTYP,MAXTYP)
!      real*8 xr(2,MAXINT,MAXTYP,MAXTYP)
!      real*8 efkt(3,MAXTYP,MAXTYP),cutoff(MAXTYP,MAXTYP)

       real*8 au(LDIM,LDIM), bu(LDIM,LDIM)
       real*8 spro
       real*8 derivx(NNDIM,NNDIM) !MG_UW1210
       real*8 derivy(NNDIM,NNDIM),derivz(NNDIM,NNDIM)
       real*8 h(MDIM), ev(MDIM), occ(MDIM)
!      real*8 qmulli(MDIM), qmold(NNDIM), qmat(NNDIM)
       real*8 qmulli(MDIM), qmold(NNDIM)
!      real*8 Qdiff(NNDIM),dipol(3),dipabs
       real*8              dipol(3),dipabs
       real*8 shift(NNDIM,3)
       real*8 shiftE(NNDIM)
       real*8 shiftE2(NNDIM) ! MG+Guanhua_QC_UW1206: KO
       real*8 shift3(NNDIM,3),shift3A(NNDIM,3),shift3B(NNDIM),ecoul3 !MG_UW1210
       ! MG_QC_UW1207: lcolspin
       integer mi,mj,mu,nu,ier,ier2,lldim,indl(NNDIM+1),indli,li 
       real*8 evdown(MDIM), spinshift(NNDIM,3)
       real*8  occdown(MDIM),qmulliup(MDIM),qmullidown(MDIM),phelp
       real*8  qmatdown(NNDIM),qmatup(NNDIM),qlold(3*NNDIM)
       real*8  qlupold(3*NNDIM),qldownold(3*NNDIM),efermiup,efermidown
       real*8  dipolup(3),dipoldown(3),espinpol
       ! end MG_QC_UW1207: lcolspin 

       real(8), allocatable :: hamil(:, :), overl(:, :)
       real(8), allocatable :: a(:, :), b(:, :), adown(:, :)
       real(8), allocatable :: ahb(:, :), bhb(:, :)
       real(8), allocatable :: gammamat(:, :, :, :), gammader(:, :, :, :)
       real(8), allocatable :: hgrad(:, :)
       
c      Anders Christensen (2015) CPE calculation
       logical docpe
       real*8  C(3*NNDIM),shiftCPE(NNDIM),cpedipol(3),cpedipabs
       real*8  e_cpe,e_delta_cpe
       ! real*8  grad_cpe(3,NNDIM), CPEPar(NNDIM,4)
       real*8  grad_cpe(3,NNDIM)

!      ASC: Dispersion gradient contribution. Defined this way to 
!      fit into the wrapper for Grimmes code.
       real*8, dimension(3,nndim) :: g_dftd3
       real*8 :: g_dftd3_norm
       real*8 :: e_dftd3

! Guishan Zheng, 10/19/2014 Arrays & parameters for DIIS algorithms
       Parameter(MxCDim=10,mdiis=4)
       real*8 qdiis(nndim,MxCDim),bdiis(MxCDim,MxCDim), cdiis(MxCDim)
       real*8 qall(nndim,MxCDim)

       logical lerep

! QC_UW_1017 based on XL_QC_UW15: NBO [moved over to wrtnbo47
!      integer elebasis(MDIM)
!      character(len=10) s_nn,s_ndim,s_num, s_format
!      character(len=500) s_center, s_center_num, s_label, s_label_num
!      real*8 overlap(MDIM),density(MDIM,MDIM),dens(MDIM) 
!      real*8 h(MDIM) 
       real(8), allocatable :: ham(:, :) !Stores the Fock matrix for NBO
       real(8), allocatable :: hamdown(:, :) !Stores the Fock matrix for NBO
! QC_UW_1017 end

!      Haibo Yu_QC_UW0609 additional convergence criteria
       real*8 qmulliold(MDIM),dqmulli,maxqmuldev,qmullitol


!      --------------------------------------------------
!      Local variables
       real*8  eqhu,qdif(NNDIM),uhuq(NNDIM),uhuqd(NNDIM)
!      --------------------------------------------------
!
!      QC: UW0110: did NOT touch GHO related items 
! GHO common block and variables ... PJ 7/2004
!
! ---------------------- GHO common block ---------------------------
!
      LOGICAL QLINK
      INTEGER NATQM,NQMLNK,IQLINK,JQLINK,KQLINK,MAXQMLINK,MXQM16
      REAL*8  BT,BTM,QMATMQ,DBTMMM

      PARAMETER (MAXQMLINK=5,MXQM16=16*MAXQMLINK)
      COMMON/QLINKI/NATQM,NQMLNK,IQLINK(MAXQMLINK),JQLINK(3,MAXQMLINK),
     &              KQLINK(MAXQMLINK)
      COMMON/QLINKL/QLINK
      COMMON/QLINKF/BT(MXQM16),BTM(MXQM16),QMATMQ(MAXQMLINK),
     &              DBTMMM(3,3,MXQM16)

      INTEGER  NORBG
      COMMON/QLINKN/NORBG
!
! label of GHO atoms and QM atom directly linked to GHO atoms in
! the QM domain, similiar to IQLINK and KQLINK
!
      INTEGER MXNN
      PARAMETER (MXNN=650)

      INTEGER IGHOSL
      COMMON /GHOSEL/ IGHOSL(MXNN)

      INTEGER IGLNK,KGLNK
      COMMON/QLINKQ/ IGLNK(MAXQMLINK),KGLNK(MAXQMLINK)
!
! large 1-d arrays have been allocated dynamically
!
      real*8  pho(*),fao(*),sao(*),who(*),
     &        fhb(*),shb(*),chb(*),cao(*)
!
! arraies for working space related to GHO 
!
      integer nfirst, nlast
      common /gorbind/ nfirst(NNDIM+1), nlast(NNDIM+1)
      real*8 eauxhb(3,NNDIM)
!
! Variables for QM-boundary atoms
!
      integer k, norbhb, naos, linao
      real*8 nelhb


!---------------------------------------------------------------------
!
       real*8 factor
       factor = 0.529177249D0

       e_cpe = 0.0d0
       e_delta_cpe = 0.0d0
       docpe = lcpe

C GZ 10/19/2014
       qdiis = 0.0d0
       bdiis = 0.0d0
       cdiis = 0.0d0
       qall  = 0.0d0


       ier=0  ! MG_QC_UW1207
       ier2=0 ! MG_QC_UW1207

       allocate(hamil(mdim, mdim))
       allocate(overl(mdim, mdim))

       allocate(a(mdim, mdim))
       allocate(adown(MDIM,MDIM))
       allocate(ahb(MDIM, MDIM))
       
       allocate(b(mdim, mdim))
       allocate(bhb(MDIM, MDIM))

       allocate(ham(mdim, mdim))
       allocate(hamdown(MDIM,MDIM))

       allocate(gammamat(NNDIM,NNDIM,3,3))
       allocate(gammader(NNDIM,NNDIM,3,3))
       allocate(hgrad(3,NNDIM))

! no gradients in mode 6
!      if (atomic) lgrad= .false.
       lerep = .false.
       if (scftol .lt. 0.0) then
         doscf = .false. 
       else
         doscf = .true.
!        Haibo_QC_UW0609: SCF based on Mulliken
         if (lscfchg) qmullitol=0.05*sqrt(scftol)
       endif
       nn = n3 / 3
!      QC_UW04: specify nbeweg as well - which is ALWAYS nn for CHARMM
       nbeweg=nn
!      QC_UW04
       
       
! calculation of indices for matrices H and S 
!
       ind(1) = 0
       do j = 1,nn 
         izpj = izp(j)
! QC: debug
         ind(j+1) = ind(j)+lmax(izpj)**2
       end do
!
! actual dimension of matrix
       ndim = ind(nn+1)      
       
       if (ndim .gt. MDIM) then
         write(*,*) ' eglcao: ndim > ', MDIM
         stop
       endif

! MG_QC_UW1207: for l-dependence (here: for lcolspin)
! maximum number of l-index
       lldim = 0
       do i=1,nn
         lldim=lldim+lmax(izp(i))
       enddo
! index for l-dependent charges
       indl(1)=0
       do i=1,nn
         indl(i+1) = indl(i)+lmax(izp(i))
       enddo

!
! construct GHO related dimension and number active electrons
!                                                  ... PJ 7/2004
!
      if(qlink) then
         norbs = ndim
         norbg = norbs
         norbhb = norbs - 3*nqmlnk
         naos = norbhb-nqmlnk
         linao = (naos*(naos+1))/2
         do i = 1, nn
            nfirst(i) = ind(i)+1
            nlast(i) = ind(i+1)
         end do
         nlast(nn) = ndim
         nelhb = nel - 3*nqmlnk
      endif
       
! setup of charge-independent part of H and S
!          ^^^^^^^^^^^^^^^^^^
       do j = 1,nn 
         do k = 1,j 
         
           call slkmatrices(j,k,x,au,bu)
           
           do n = 1,ind(k+1)-ind(k) 
              do m = 1,ind(j+1)-ind(j)
               hamil(ind(j)+m,ind(k)+n) = au(m,n)
               hamil(ind(k)+n,ind(j)+m) = au(m,n)       
               overl(ind(j)+m,ind(k)+n) = bu(m,n)
               overl(ind(k)+n,ind(j)+m) = bu(m,n)
             end do
           end do
         end do
       end do

! MG+Guanhua_QC_UW1005: add corresponding part into SCF cycle 
! for external charges : extra shift shiftE 
      if (.not.lcdko) then
        if ((EXT.eq.'CH') .or. (EXT.eq.'EF')) then
! Guanhua: get the contribution from external MM charges
       call externalshift(nn,x,    shiftE,shiftE2,qsccewc,qsccews,
     $                    erfct,del,pts)
        endif
      endif

! QC-GSBP: The charge independent part for GSBP-SCC is OMEGAGS, which
! has already been computed in gsbpscc.src. Simply use it whereever
! shiftE appears
! setup for SCF cycle (why freeze maxiter?!:-) QC.
! related to broden_mixing scheme?!
!
       if (doscf) then
!        if (mysccid.eq.0) then 
!          open(70,file='scfsum',form='formatted',status='unknown') 
!          close(70, status='delete')
!        endif
         almix = 0.2
         maxiter = 400 
         eelold = 1.0d10
       else
          maxiter = 1
       endif

! JZ_UW12
       SCC_CALLED = .true.
       if (SMBP_GRAD) then
         maxiter_save = maxiter
         maxiter = 1
       endif

!-----------------------------------------------------------------------
! Guishan_Zheng 03/19/2013: 
! Initial guess from DXL-BOMD and other electronic dynamics algorithms
! The first (IDAMP) steps use very tight SCF convergence criteria, then
! the criterion switches to the input value
       If(IGuess.ne.0.and.IMD.eq.1) then
          if(igeometry.eq.1) scftol0 = scftol
          if(igeometry.le.idamp) scftol = scftight
          if(igeometry.gt.idamp) scftol = scftol0
          call dxlc(igeometry,iguess,nndim,nn,qzero,qmat,qaux)
       endif
!-----------------------------------------------------------------------
! MG_QC_UW1207: initialize oldcharges, MG_UW1210 (ldep)
       if (doscf) then
         if (lldep) then
           do i=1,lldim
             qlold(i)=ql(i)
           enddo
         else  
           do i = 1,nn 
             qmold(i) = qmat(i)
           end do
         endif
         if (lcolspin) then
           do i = 1,lldim
             qlupold(i)   = qlup(i)
             qldownold(i) = qldown(i)
           enddo
         endif
       endif
       
      e_dftd3 = 0.0d0
      if (lsccdftd2.or.lsccdftd3) then

        ! Initialize energy.

        if (lgrad) then 

          ! Initialize (the necessary part of) the gradient.
          do i = 1, nn
            g_dftd3(1,i) = 0.0d0
            g_dftd3(2,i) = 0.0d0
            g_dftd3(3,i) = 0.0d0
          enddo
          call calc_g_dftd3(x, e_dftd3, g_dftd3, nn)

        else
            call calc_e_dftd3(x, e_dftd3, nn)

        endif 
        
      endif
!      write(*,*) "eglcao: starting scf..."

! MG_QC_UW1205: calc gamma once before scc-cycle, MG_UW1210 (ldep)
       if (doscf) then
         call get_gammamat(nn,x,izp,uhubb,uhder,boxsiz,period,
     &                    lldep,lmax,gammamat,gammader)
       endif
! end
! QC: control printing
       if (prnlev.ge.6) then       
        write(outu,*) 'initial charges:'
        write(outu,'(1000(5(F10.5,X),/))') qmat(1:nn)
       endif
! GZ 10/19/2014 initialize the iteration number of DIIS
       ndiis = 0 
       
!  *************************
!  SCF cycle starts here
!  *************************
       
       do niter = 1,maxiter 
         miter = niter
       
! save old charges:  MG_QC_UW1207 moved to outside of SCF cycle 
! this is taken care of around broyden/mixer call
         if (doscf) then
           do i = 1,nn 
             qmold(i) = qmat(i)
           end do
         endif
       
       
! charge-independent part of H and S
!
       do j = 1,nn 
         indj  = ind(j)
         indj1 = ind(j+1)
         do k = 1,j 
           indk  = ind(k)
           indk1 = ind(k+1)
           
           do n = 1,indk1-indk 
             do m = 1,indj1-indj 
               a(indj+m,indk+n) = hamil(indj+m,indk+n)
               b(indj+m,indk+n) = overl(indj+m,indk+n)
             end do
           end do
         end do
       end do
       
! zero shift : we do that in HAMILSHIFT 
!
!         do i = 1,nn
!           shift(i) = 0.0d0
! MG_QC_UW1205
!           shift3(i)= 0.0d0
!           shift3A(i) = 0.0d0
! end
!         end do
       
! add charge dependent terms (Hubbard, etc. )
! start by defining shift 
! MG+Guanhua_QC_UW1103: note the change of arguments here
! MG_QC_UW1207: calculate also spinshift for lcolspin
! MG_UW1210 (ldep)
         if (doscf) then
           call HAMILSHIFT(nn,izp,gammamat,gammader,
     &       qmat,qzeroscc,shift,shift3,shift3A,shift3B,indl,spinshift)
!cc add external charges! CHARMM
! MG+Guanhua_QC_UW1206: KO
      if (lcdko) then
          if ((EXT.eq.'CH').or. (EXT.eq.'EF'))then 
          call externalshift(nn,x,    shiftE,shiftE2,qsccewc,qsccews,
     $                    erfct,del,pts)
!           do i = 1,nn
!           shift(i) = shift(i) - shiftE(i)
!! minus, weil Z positiv sind!, positive DeltaQ meinen aber negative Ladung
!           end do
          endif 
      endif

! Add in the Hubbard derivative term@@@Will add the fitted Gaussian para later @@@
!      QC_UW_08: Charge-dependent on-site Hubbard 
       if (luhder) then
!        write(*,*) "charge-derivative"
         do i = 1, nn 
          qdif(i) = (qmat(i)-qzeroscc(izp(i),4))
!         QC_UW0305: additional charge-dependence term
          if (luhgau) then
            qtmp=exp(-alp_hgau*(qdif(i)-q0_hgau)*(qdif(i)-q0_hgau))
          else
            qtmp=0.0d0
          endif
          uhuq(i) = uhder(izp(i),1) + v0_hgau*qtmp
          uhuqd(i) = -v0_hgau*qtmp*2.d0*alp_hgau*(qdif(i)-q0_hgau)

         enddo    
       endif   


! >>>> QC_GSBP: add GSBP contributions to the matrix elements
!          write(*,*) "calling gsbpshift..."
           IF (QGSBPSCC) 
     $       CALL GSBPSHIFT(nn,izp,qmat,qzeroscc,gamags,omegags,shift)
! >>>> XIAO ZHU & QC: add PB contributions to the matrix elements
           IF (QPBSCC)
     $       CALL PBSHIFT(nn,rf_qm,rf_mm,shift)



c K.W. 2014-04-14 SCF part of CPE calculation,
c c/f Kaminski et al., JPC A 116 (2012), 9131.
           if (docpe) then
             if (lldep) then
               write (outu,*) 'Sorry, CPE does not work with l-dependent
     &          hubbards at the moment'
               stop
             endif

#if KEY_DFTBMKL==1
             call cpe_scf(nn,qmat,qzeroscc,izp,x,e_cpe,shiftCPE,
     &       C,dipol,cpedipol,cpedipabs,uhder,uhubb)
#endif
           endif

!
! update hamiltonian matrix 
!
      if (lldep) then !MG_UW1210 (ldep)
        do i = 1,nn 
        do li = 1,lmax(izp(i))
        do mi = 1,2*li-1
           mu = ind(i)+(li-1)**2+mi
           do j = 1,i
           do lj = 1,lmax(izp(j))
           do mj = 1,2*lj-1
              nu = ind(j)+(lj-1)**2+mj
              a(mu,nu)     = a(mu,nu)+0.5d0*overl(mu,nu)
     &        *(shift(i,li)+shift(j,lj)
     &          -(shiftE(i)+shiftE(j)+shiftE2(i)+shiftE2(j)))
           end do
           end do
           end do
        end do
        end do
        end do
      else ! not ldep
        if (docpe) then
          do i = 1,nn 
           do li = 1,lmax(izp(i))**2 
            do j = 1,i
             do lj = 1,lmax(izp(j))**2 
                  a(ind(i)+li,ind(j)+lj) = a(ind(i)+li,ind(j)+lj) 
     &            +0.5*overl(ind(i)+li,ind(j)+lj)*(shift(i,1)+shift(j,1)
     &            -(shiftE(i)+shiftE(j)+shiftE2(i)+shiftE2(j))) ! MG+Guanhua_QC_UW1206: KO
     &            -0.5d0*overl(ind(i)+li,ind(j)+lj)*(shiftCPE(i)
     &            +shiftCPE(j))
             end do
            end do
           end do
          end do
        else
          do i = 1,nn 
           do li = 1,lmax(izp(i))**2 
            do j = 1,i
             do lj = 1,lmax(izp(j))**2 
                  a(ind(i)+li,ind(j)+lj) = a(ind(i)+li,ind(j)+lj) 
     &            +0.5*overl(ind(i)+li,ind(j)+lj)*(shift(i,1)+shift(j,1)
     &            -(shiftE(i)+shiftE(j)+shiftE2(i)+shiftE2(j))) ! MG+Guanhua_QC_UW1206: KO
             end do
            end do
           end do
          end do
        endif
      endif !not ldep

      endif ! doscf

! QC: UW_0305: Include charge dependent Hubbard
! MG_UW1210: ldep not implemented with luhder (sccdftbini.src)
       if (luhder) then
!       write(*,*) "update Hamiltonian -charge "
         do i = 1,nn
           do li= 1,lmax(izp(i))**2
             do j= 1,i
               do lj= 1,lmax(izp(j))**2
                 a(ind(i)+li,ind(j)+lj)=a(ind(i)+li,ind(j)+lj)
     &           +0.25*overl(ind(i)+li,ind(j)+lj)*
     &             ( uhuq(i)*(qmat(i)-qzeroscc(izp(i),4))**2 +
     &               uhuq(j)*(qmat(j)-qzeroscc(izp(j),4))**2 +
     $              uhuqd(i)*(qmat(i)-qzeroscc(izp(i),4))**3/3.0 +
     $              uhuqd(j)*(qmat(j)-qzeroscc(izp(j),4))**3/3.0 )
               end do  
             end do  
           end do  
         end do   
       endif    
            
! QC_UW08: end

! MG+Guanhua_QC_UW1101: add terms related to scc3rd
! MG_UW1210: (ldep)
       if (lscc3rd) then
         if (lldep) then
           do i = 1,nn 
           do li = 1,lmax(izp(i))
           do mi = 1,2*li-1
             mu = ind(i)+(li-1)**2+mi
             do j = 1,i
             do lj = 1,lmax(izp(j))
             do mj = 1,2*lj-1
               nu = ind(j)+(lj-1)**2+mj
               a(mu,nu)     = a(mu,nu)+0.5d0*overl(mu,nu)
     &         *1.0d0/3.0d0*(shift3(i,li)+
     &           shift3A(i,li)+shift3(j,lj)+shift3A(j,lj)+
     &           shift3B(i)+shift3B(j))
             end do
             end do
             end do
           end do
           end do
           end do
         else ! not ldep
           do i = 1,nn
             do li = 1,lmax(izp(i))**2
               do j = 1,i 
                 do lj = 1,lmax(izp(j))**2
                   a(ind(i)+li,ind(j)+lj) = a(ind(i)+li,ind(j)+lj)
     &             +0.5d0*overl(ind(i)+li,ind(j)+lj)
     &             *1.0d0/3.0d0*(2.0d0*shift3(i,1)+shift3A(i,1)
     &             +2.0d0*shift3(j,1)+shift3A(j,1))
                 enddo
               enddo
             enddo
           enddo
         endif
       endif
! end            

! MG_QC_UW1207: add spinshift for lcolspin (collinear spin)
      if (lcolspin) then
        ! add spin contribution for spinup
        ! add spin contribution for spindown (additional matrix)
        do i = 1,nn 
        do li = 1,lmax(izp(i))
        do mi = 1,2*li-1
           mu = ind(i)+(li-1)**2+mi
           do j = 1,i
           do lj = 1,lmax(izp(j))
           do mj = 1,2*lj-1
              nu = ind(j)+(lj-1)**2+mj
              adown(mu,nu) = a(mu,nu)-0.5d0*overl(mu,nu)
     &          *(spinshift(i,li)+spinshift(j,lj))
              a(mu,nu)     = a(mu,nu)+0.5d0*overl(mu,nu)
     &          *(spinshift(i,li)+spinshift(j,lj))
           end do
           end do
           end do
        end do
        end do
        end do
      endif
! end MG_QC_UW1207

!
! GHO transformation of H and S matrices ... PJ 7/2004
!
      if (qlink) then
!
! pack H, and S to lower triangle form
!
        k = 0
        do i = 1, norbs
           do j = 1, i
              k = k + 1
              fao(k) = a(i,j)
              sao(k) = b(i,j)
           end do
        end do
!
! transform S and H into hb basis for QM link atom, also
! get the auxiliary orbital energies as diagonal elements
! of the H matrix in the hybrid basis. Note that eauxhb
! for the S transform is only dummy parameters, but for
! the H transformation, it contains the auxililary orbital
! energies, which will be included in the TB band structure
! energies.
!
        call ftofhb(sao,shb,bt,natqm,nqmlnk,norbs,nfirst,nlast,eauxhb)
        call ftofhb(fao,fhb,bt,natqm,nqmlnk,norbs,nfirst,nlast,eauxhb)
!
! tranfer H and S in HB basis to 2-d arries in working space
!
        k = 0
        do i = 1, norbhb
           do j = 1, i
              k = k + 1
              ahb(i,j) = fhb(k)
              bhb(i,j) = shb(k)
           end do
        end do
!
! GHO diagonalization, solve HC=SCE in active hybrid basis
!
        call ewevge(mdim,mdim,norbhb,ahb,bhb,ev,h,1,-1,ier)
!
! append and transform auxiliary hybrid orbitals to AO basis
!
        k = 0
        do j = 1, norbhb
           do i = 1, norbhb
              k = k + 1
              chb(k) =  ahb(i,j)
           end do
        enddo
        call ctrasf(chb, cao, norbs)
        k = 0
        do j = 1, norbs
           do i = 1, norbs
              k = k + 1
              a(i,j) = cao(k)
           end do
        enddo
!
! update the corresponding auxiliary orbital energies
!
         k = norbhb
         do i = 1, nqmlnk
            do j = 1, 3
               k = k + 1
               ev(k) = eauxhb(j,i)
            end do
         end do
      else
         
! solution of eigenvalue problem 

! QC_UW1017: Based on XL_QC_UW15: NBO
! Stores the Fock matrix if we need NBO, 
! 'cause 'a' becomes MO vectors after diagonalization
           if(qnbo) then
             do m=1,ndim
               do n=1,m
                 ham(m,n) = a(m,n)
                 ham(n,m) = ham(m,n)
               enddo
             enddo
! for open-shell, need to store the spin-down case
             if(lcolspin) then
              do m=1,ndim
                do n=1,m
                  hamdown(m,n) = adown(m,n)
                  hamdown(n,m) = hamdown(m,n)
                enddo
              enddo
             endif
           endif
!
! XL_QC_UW1211: implement multiple diagnolizer
#if KEY_DFTBMKL==1
           if (dmeth .eq. 0) then
             call ewevge(MDIM,MDIM,ndim,a,b,ev,h,1,-1,ier)
           elseif (dmeth .eq. 1) then
             call ewevge_dsygv(MDIM,MDIM,ndim,a,b,ev,h,1,-1,ier)
           elseif (dmeth .eq. 2) then
             call ewevge_dsygvd(MDIM,MDIM,ndim,a,b,ev,h,1,-1,ier)
           endif
#else
           call ewevge(MDIM,MDIM,ndim,a,b,ev,h,1,-1,ier)
#endif

! MG_QC_UW1207: spinpolarized case: solve also for down-spin electrons 
         if(lcolspin) then
             do i=1,ndim
               do j=1,i
                 b(j,i) = 0.0d0
                 b(i,j) = overl(i,j)
               enddo
             enddo
! XL_QC_UW1211: implement multiple diagnolizer
#if KEY_DFTBMKL==1
             if (dmeth .eq. 0) then
               call ewevge(MDIM,MDIM,ndim,adown,b,evdown,h,1,-1,ier2)
             elseif (dmeth .eq. 1) then
               call ewevge_dsygv(MDIM,MDIM,ndim,adown,b,evdown,
     &                                                   h,1,-1,ier2)
             elseif (dmeth .eq. 2) then
               call ewevge_dsygvd(MDIM,MDIM,ndim,adown,b,evdown,
     &                                                   h,1,-1,ier2)
             endif
#else
             call ewevge(MDIM,MDIM,ndim,adown,b,evdown,h,1,-1,ier2)
#endif
         endif

! ... PJ 7/2004
      endif
       
         if ((ier .ne. 0).or.(ier2.ne.0)) then 
           print *,' ewevge: ier =',ier
! QC: exit gracefully
!          stop
           call WRNDIE(-5,'<EGLCAO>','SCCDFTB not converged.')
         endif
!
! Machine accuracy 
!
         dacc = 4*racc
!
! form the GHO density matrix ... PJ 7/2004
!
         if (qlink) then
!
! add the fractional occupations for GHO auxililary hybrid orbitals
! this equals to an evenly distribution of the MM charge of GHO
! boundary B over three auxiliary orbitals
!
             call FERMI(nelhb,telec,norbhb,ev,dacc,occ,efermi)
             k = norbhb
             do i = 1, nqmlnk
                do j = 1, 3
                   k = k + 1
                   occ(k) = 1.0-qmatmq(i)/3.0
                end do
             end do
!
! form the total density matrix and energy weighted density
! matrix in HB basis for GHO gradient
!
             call cexpd(chb,norbs,nqmlnk)
             call pwho(pho,who,chb,occ,ev,norbs)
         else
!
! Calculate occupation (occ) and fermi energy (efermi)
! using fermi-distribution
! MG_QC_UW1207: note additional argument!
         if (lcolspin) then
           call FERMI(nelup,telec,ndim,ev,dacc,occ,efermiup,1.0d0)
           call FERMI(neldown,telec,ndim,evdown,dacc,occdown,
     &                                             efermidown,1.0d0)
         else
           call FERMI(nel,telec,ndim,ev,dacc,occ,efermi,2.0d0)
         endif
! end MG_QC_UW1207
         
!
! ... PJ 7/2004
         endif 
         
! sum of occupied eigenvalues
!
         eel = 0.0d0
         do i = 1,ndim 
!
! for GHO, we have to loop over all orbitals, becuase
! auxiliary orbitals are also occupied ... PJ 7/2004
!
           if (.not. qlink) then
              if (occ(i) .lt. dacc) goto 75
           end if
!
           eel = eel + occ(i)*ev(i)
         end do
75       continue
! MG_QC_UW1207
         if (lcolspin) then
           do i = 1,ndim
             if (occdown(i).lt.dacc) goto 76
             eel = eel + occdown(i)*evdown(i)
           enddo
76         continue
         endif
! end MG_QC_UW1207

!
! Lowest unoccupied level
!
!         nstart = i  !MG_QC_UW1207: not necessary...

! determine mulliken charges , the charge of the whole system
! and the mulliken 
         if(doscf .or. atomic) then
           ! MG_QC_UW1207: note also change of arguments
           ! first spinpolarized
           if (lcolspin) then
             call MULLIKEN(nn,x,izp,qmatup,qzeroscc,qmulliup,qlup,
     &                     qtotup,ndim,dacc,occ,a,
     &                     overl,lmax,ind,dipolup,dipabs)
             call MULLIKEN(nn,x,izp,qmatdown,qzeroscc,qmullidown,qldown,
     &                     qtotdown,ndim,dacc,occdown,adown,
     &                     overl,lmax,ind,dipoldown,dipabs)
             qtot=qtotup+qtotdown
             do i=1,nn
               qmat(i) = qmatup(i)+qmatdown(i)
             enddo
             do i=1,lldim
               ql(i) = qlup(i)+qldown(i)
             enddo
             do i=1,ndim
               qmulli(i) = qmulliup(i)+qmullidown(i)
             enddo
             dipabs=0.0d0 ! norm of dipole moment
             do i=1,3
               dipol(i)=dipolup(i)+dipoldown(i)
               dipabs = dipabs + dipol(i)**2
             enddo
             dipabs=dsqrt(dipabs)
           ! now spinunpolarized
           else
             call MULLIKEN(nn,x,izp,qmat,qzeroscc,qmulli,ql,qtot,
     &              ndim,dacc,occ,a,overl,lmax,ind,dipol,dipabs)
           endif
           ! MG_UW1210: if charge is getting printed out we need this
           IF (LMULIK) THEN
             do j=1,nn
               qmulik(j)=qzeroscc(izp(j),4) - qmat(j)
               !write(*,'(1x,"MULIK>",I5,2F8.3)') j,qmulik(j),qmat(j)
             enddo
           ENDIF
           if (dabs(qtot-nel).gt.1.0d-06) then
             if (wrnlev.ge.-5)
     &         write(outu,*) "Warning: actual number of electrons:",qtot
     &                   ,"instead of ",nel
           endif 
         endif
       
! output spektra, occupation numbers, mulliken charges 
!
! QC: Disable for the moment 
         if (mysccid.eq.0.and.atomic) 
     &     call outspec(nn,nbeweg,ndim,ind,lmax,izp,ev,occ,efermi,
     &                qmat,qmulli,qtot,dipol,dipabs,
     &                lcolspin,evdown,occdown,qmatup,qmatdown,qmulliup,
     &                qmullidown,qtotup,qtotdown,efermiup,efermidown)
       
         if (evector) then
           ! MG_QC_UW1207
           if (lcolspin) then
             call outspineigenvectors(a,ev,occ,adown,evdown,occdown,
     &                                ind,nn)
           else 
             call outeigenvectors(a,ev,occ,ind,nn)
           endif
         endif
        
       
! complete calculation of electronic energy:
! charge-dependent energy contribution
! warning: this will only lead to the right result if convergence
! has been reached
!
         if (doscf) then
           ecoul = 0.0d0
           ecoul3=0.0d0 ! MG+Guanhua_QC_UW1205
           eext = 0.0d0
           if (lldep) then !MG_UW1210
             do i=1,nn
               do li=1,lmax(izp(i))
                 indli=indl(i)+li
                 ecoul=ecoul+shift(i,li)*(ql(indli)+qzeroscc(izp(i),li))
                 if (lscc3rd) then
                   ecoul3=ecoul3+shift3(i,li)*qzeroscc(izp(i),li)
     &                          +(shift3A(i,li)+shift3B(i))*ql(indli)
                 endif
               enddo
             enddo
           else ! no ldep
             do i = 1,nn
               ecoul = ecoul + shift(i,1)*(qmat(i)+qzeroscc(izp(i),4))
               if (lscc3rd) then ! MG+Guanhua_QC_UW1205
                 ecoul3= ecoul3+shift3(i,1)*(qmat(i)+qzeroscc(izp(i),4))
     &                   + shift3A(i,1)*qmat(i)
               endif
             end do
           endif !ldep
           if (docpe) then
             e_delta_cpe = 0.0d0
             do aa = 1,nn
               e_delta_cpe = e_delta_cpe + shiftCPE(aa) * qmat(aa)
             end do
           endif

! Guanhua_UW1206 KO: need to redo externalshift calculation because now
!cccccc CHARMM
! Guanhua_UW1206 KO: need to redo externalshift calculation because now 
! the qmat is changed
      if (lcdko) then
          call externalshift(nn,x,    shiftE,shiftE2,qsccewc,qsccews,
     $                    erfct,del,pts)
      endif

           if((EXT.eq.'CH').or. (EXT.eq.'EF'))then
           do i = 1,nn
           eext = eext + shiftE(i)*qzeroscc(izp(i),4)
     $                 + shiftE2(i)*qmat(i) ! Guanhua_QC_UW1005: KO
           end do
           endif
!cccccc
! >>>>     QC_GSBP contribution. The GAMAGS terms are already included
! >>>>     in the eel and ecoul terms, similar to the gamma term from 
! >>>>     sccdftb
! >>>>     But the OMEGAGS terms need to be modified similar to 
! >>>>     the CHARMM contribution
           egsbpscc=0.0d0
           IF (QGSBPSCC) THEN
             do i=1,nn
               egsbpscc = egsbpscc + 
     $                    OMEGAGS(i)*(qzeroscc(izp(i),4)-qmat(i))
     $                                         *factor
             enddo
           ENDIF

!          XIAO ZHU & QC for PBSCC
           epbscc=0.0d0
           IF (QPBSCC) THEN
             do i=1,nn
               epbscc = epbscc +
     $                    rf_mm(i)*(qzeroscc(izp(i),4)-qmat(i))
     $                                         *factor
             enddo
           ENDIF

!          UW_0605: Hubbard derivative contribution@@
           eqhu = 0.0d0
           if (luhder) then
             do i = 1,nn
              eqhu = eqhu  +
     &                  uhuq(i)*(qmat(i)-qzeroscc(izp(i),4))**3/6.d0
     &            - 0.5*uhuq(i)*(qmat(i)-qzeroscc(izp(i),4))**2
     &                         * qmat(i)
     $            -    uhuqd(i)*(qmat(i)-qzeroscc(izp(i),4))**3/6.d0
     $                         * qmat(i)

             end do
            endif

!        MG_QC_UW1207: spinpolarization term
!        etot=sum_i n_i epsilon_i - ..... - 1/2*sum_a sum_l sum_l' p_al p_al' W_all'  = ... -1/2*sum_a sum_l p_al spinshift(a,l)
           espinpol=0.0d0
           if (lcolspin) then
              do i=1,nn
                do li=1,lmax(izp(i))
                  indli=indl(i)+li
                  espinpol = espinpol + spinshift(i,li)*
     &                      ( qlup(indli) - qldown(indli) )
                 enddo
               enddo
           endif
 
           
! MG_QC_UW1206 change factor before eext, because calculated differently (due to KO)
!          eel = eel-0.5*ecoul + 0.5*eext
           eel = eel-0.5*ecoul + eext + 0.5*egsbpscc + eqhu 
!          xiao zhu and qc for SCCPB
     $           +0.5*epbscc
! MG+Guanhua_QC_UW1101
     &         -ecoul3/3.0d0
! MG_QC_UW1207
     &         -0.5d0*espinpol
! Anders Christensen 2015
     &         + e_cpe + e_delta_cpe
! Anders Christensen 2015
!      &         + e_dftd3

!          write(*,*) "EEL>",eel,ecoul,eext,egsbpscc,eqhu
! >>>>     QC_GSBP end
         endif
   
! print energy, check convergence, call broyden mixing
!
         if (doscf) then
!          if (mysccid.eq.0) then 
!            open(70,file='scfsum',form='formatted',status='unknown')
!            write(70,*) 'iter: ',niter,', E= ',eel
!            close(70) 
!          endif

!          Haibo_QC_UW0609 Tighter SCF Convergence based on Mulliken
           if (lscfchg) then
             maxqmuldev=0.0d0
             do i = 1, ndim
                dqmulli = qmulli(i) -qmulliold(i)
                qmulliold(i) = qmulli(i)
                if ( abs(dqmulli) .ge. maxqmuldev)  then
                   maxqmuldev = abs(dqmulli)
                endif
             enddo
!            write(*,*) "Conv?",niter,maxqmuldev,qmullitol,
!    $                          abs(eel-eelold),scftol

             if ((abs(eel-eelold) .lt. scftol).and.
     $          (maxqmuldev .lt. qmullitol)) goto 70
           else
             if (abs(eel-eelold) .lt. scftol) goto 70
           endif

           if (mysccid.eq.0) then
            if ( writeout .ge. 1 ) then
             write(72,*) atomic,' iter: ',niter,', E= ',eel
            endif
            if ( writeout .ge. 2 ) then
             write(73,*) ' iter: ',niter 
             ! MG_QC_UW1207
             if (lcolspin) then
               write(73,*) "spin up:"
               write(73,'(10F12.6)') (qmatup(i),i=1,nn)
               write(73,*) "spin down:"
               write(73,'(10F12.6)') (qmatdown(i),i=1,nn)
               write(73,*) "total:"
             endif
             write(73,'(10F12.6)') (qmat(i),i=1,nn)
             write(74,*) ' iter: ',niter 
             ! MG_QC_UW1207
             if (lcolspin) then
               write(74,*) "spin up:"
               write(74,'(10F12.6)') (occ(i),ev(i),i=1,ndim)
               write(74,*) "spin down:"
               write(74,'(10F12.6)') (occdown(i),evdown(i),i=1,ndim)
             else 
               write(74,'(10F12.6)') (occ(i),ev(i),i=1,ndim)
             endif
            endif
           endif
           eelold = eel
!
!   broyden mixing
!
! MG_QC_UW1207: mixer overwrites qmat and qmold with new charges for the next scc-cycle
! for lcolspin, it also takes care of ql,qlold,qlup,qlupold,...
#if KEY_DFTBMKL==1
           if(sccmix.eq.3) then
             call diis(0,niter,mdiis,ndiis,nn,nndim,MxCDim,qmold,
     &                 qmat,qdiis,qall,bdiis,cdiis,0)
             do i = 1,nn 
               qmold(i) = qmat(i)
             end do
           else
             call mixer(niter,almix,nn,qmold,qmat,lcolspin,ndim,lldim,
     &            lldep,lmax,izp,indl,qlold,ql,qlupold,qldownold,qlup,
     &            qldown,sccmix,gens,wsy) ! Guanhua_puja_QC_UW1212
           endif
#endif
#if KEY_DFTBMKL==0
             call mixer(niter,almix,nn,qmold,qmat,lcolspin,ndim,lldim,
     &            lldep,lmax,izp,indl,qlold,ql,qlupold,qldownold,qlup,
     &            qldown,sccmix,gens,wsy) ! Guanhua_puja_QC_UW1212
#endif

         end if
       end do
70     continue

!   QC: give at least a warning if maxit reached

       if (niter.ge.maxiter) write(outu,*) "EGLCAO> Maxiter Reached."
       if (prnlev.ge.6) then 
        write(outu,'(A,X,I4,X,A,E12.5,2(X,A,I5))') 
     $    'EGLCAO> SCF Iterations:',  niter, 
     $   'with tolerance:',scftol, 'at geometry',igeometry,'with imd=',
     $   IMD
        write(outu,*) 'converged charges:'
        write(outu,'(1000(5(F10.5,X),/))') qmat(1:nn)
       endif
       
! MG+Guanhua_QC_UW1103: note the change of arguments here       
! MG_QC_UW1207: calculate also spinshift for lcolspin
       if (doscf) then
         call HAMILSHIFT(nn,izp,gammamat,gammader,
     &     qmat,qzeroscc,shift,shift3,shift3A,shift3B,indl,spinshift)
       endif
       
! end of SCF
!
! MG_QC_UW1206: shiftE is added to force separately!  -- MG1210 otherwise we would also have to take into account ldep here! (add for every shift(i,li))
!cc add EXTERNAL CHARGES CHARMM
!          if ((EXT.eq.'CH').or. (EXT.eq.'EF'))then
!           do i = 1,nn
!           shift(i) = shift(i) - shiftE(i)
!           enddo
!          endif
! minus, weil Z positiv sind!, positive DeltaQ meinen aber negative Ladung  
! >>>> QC_GSBP: add GSBP contributions to the matrix elements
           IF (QGSBPSCC) 
     $       CALL GSBPSHIFT(nn,izp,qmat,qzeroscc,gamags,omegags,shift)

! >>>> XIAO ZHU & QC: add PB contributions to the matrix elements
           IF (QPBSCC)
     $       CALL PBSHIFT(nn,rf_qm,rf_mm,shift)


!      if (atomic) then       
       
!
! calculation of atomic energies
! MG_QC_UW1207: this is not compatible with lcolspin (UNPE) at the moment!
! MG_UW1210: also not with ldep!
!      call atomenerg(nn,ndim,ind,izp,shift,
!    &      x,hamil,a,occ,doscf,period,boxsiz,xinvbox,nlat)
!      endif
       
! start with repulsive energy
       erep = 0.0d0
       
!
! Calculate repulsive energy
!
       call repulsive(nn,izp,period,x,boxsiz,xinvbox,nlat,erep)
       
! total energy 
!
       e = eel + erep + e_dftd3
!      write(*,*) "Repulsive",eel,erep

       if (niter.ge.maxiter) then
        write(outu,*) 'iter: ',niter,', E= ',e,', Eel= ',eel
     c   ,'SCCDFTB NOT CONVERGED: maxiter',maxiter
!       else
!       if (mysccid.eq.0) then
!       write(71,*) 'iter: ',niter,', E= ',e,', Eel= ',eel
!       endif
       endif

! QC_UW_1017: based on XL_QC_UW15: NBO

       if (qnbo) call wrtnbo47(nn,ndim,x,ind,overl,occ,a,ham,
     &                         occdown,adown,hamdown)

! QC_UW_1017 End: NBO

! start with gradient
        
       do j = 1,nn 
         do i = 1,3 
           gr(i,j) = 0.0d0
! Guanhua_QC_UW1101 
           hgrad(i,j) = 0.0d0
! end
         end do
       end do

!   move here lgrad
       if(lgrad) then
! DISERPSION
! K.W. 2014-05-07 calculation of CPE gradient
         if (docpe) then
#if KEY_DFTBMKL==1
           call cpe_grad(nn,izp,x,qmat,qzeroscc,C,grad_cpe,uhubb,
     $     uhder, ndim,a, lmax,ind,occ, shiftCPE)
#endif

         end if
       if (dispers) then
        call  dispersion_egr(nn,x,hgrad,boxsiz,nlat,period)
        do j=1,nn
         do i=1,3
           gr(i,j) = gr(i,j) + hgrad(i,j)
         end do
        end do
!   add dispersion energy to total:
        e = e + Edis
       endif
! END DISPERSION       
! BEGIN EXTERNAL CHARGES CHARMM
      if((EXT.eq.'CH').or. (EXT.eq.'EF'))then
!      call externalchgrad(nn,x,izp,hgrad,qmat,
       call externalchgrad(nn,x,    hgrad,     
     $                     qsccewc,qsccews,qscqewc,qscqews,
     $                    erfct,del,pts)
        do j=1,nn
         do i=1,3
           gr(i,j) = gr(i,j) + hgrad(i,j)
         end do
        end do
      endif
! END EXTERNAL CHRARGES
!      if(lgrad) then
!
! Calculate gradient of the repulsive energy
!
       call repulsivegrd(nn,nbeweg,izp,period,x,boxsiz,
     &                   xinvbox,nlat,hgrad)
       do j=1,nn 
         do i=1,3 
           gr(i,j) = gr(i,j) + hgrad(i,j)
         end do
       end do
       
      if (luhder) then
        do i=1,nn
         shift(i,1) = shift(i,1) + 0.5*uhuq(i)*
     &     (qmat(i) - qzeroscc(izp(i),4))**2
     $     + uhuqd(i)*(qmat(i) - qzeroscc(izp(i),4))**3/6.d0
        enddo
      endif

! gradient 
!
! MG+Guanhua_QC_UW1205: changed the arguments here (for 3rd order and KO)
! MG_QC_UW1207: consider also spin-polarization!, MG_UW1210 (ldep)
       call usualgrd(nn,nbeweg,ndim,izp,lmax,ind,period,doscf,
     &         x,ev,a,b,occ,shift,shift3,shift3A,shift3B,shiftE,shiftE2,
     &         dacc,boxsiz,xinvbox,hgrad,lldep,lcolspin,1.0d0,spinshift)
                    
       do j=1,nn 
        do i=1,3
         gr(i,j) = gr(i,j) + hgrad(i,j)
        end do
       end do

! MG_QC_UW1207, MG_1210 (ldep)
       if (lcolspin) then
         call usualgrd(nn,nbeweg,ndim,izp,lmax,ind,period,doscf,x,evdown
     &    ,adown,b,occdown,shift,shift3,shift3A,shift3B,shiftE,shiftE2,
     &     dacc,boxsiz,xinvbox,hgrad,lldep,lcolspin,-1.0d0,spinshift)
           do j=1,nn 
            do i=1,3
             gr(i,j) = gr(i,j) + hgrad(i,j)
            end do
           end do
       endif
       
! here are the contributions due to gamma if in scf mode
       if (doscf) then
! MG+Guanhua_QC_UW1205: changed the arguments here, MG_UW1210 (ldep)
         call GAMMAGRAD(nn,nbeweg,x,izp,uhubb,uhder,
     &                  qmat,qzeroscc,ql,indl,boxsiz,period,hgrad,
     &                  erfct,del,pts)
           do j=1,nn 
            do i=1,3 
               gr(i,j) = gr(i,j) + hgrad(i,j)
            end do
           end do
         endif



c K.W. 2014-05-02; CPE gradient calculation
         if (docpe) then
           do j = 1,nn
               gr(1,j) = gr(1,j) + grad_cpe(1,j)
               gr(2,j) = gr(2,j) + grad_cpe(2,j)
               gr(3,j) = gr(3,j) + grad_cpe(3,j)
           end do
         endif

         if (lsccdftd2.or.lsccdftd3) then
          do i = 1, nn

            gr(1,i) = gr(1,i) + g_dftd3(1,i) 
            gr(2,i) = gr(2,i) + g_dftd3(2,i) 
            gr(3,i) = gr(3,i) + g_dftd3(3,i) 
          enddo
        endif 
        
      ! endif

! >>>> QC_GSBP: the GSBP_SCC contribution to the gradident
!      those involve overlap derivatives (i.e., terms
!      depend on charge variations) are already handled in
!      usualgrd because GSBP components have been included in SHIFT.
!      Those depend on FIXED QM Mulliken charges(and GSBP basis vectors)
!      are handled conveniently in misc/gsbpscc.src/sccgsbp4 following
!      very closely the way that MM GSBP component works. 
! <<<< QC_GSBP: done.
!
! constraints? QC: Jan. 2010 - take out
!
!      if(constr2) then
!       do i=1,conat(1)
!        cai=conat(i+1)
!        spro=convec(1,i)*gr(1,cai) 
!        spro=spro+convec(2,i)*gr(2,cai) 
!        spro=spro+convec(3,i)*gr(3,cai) 
!        gr(1,cai)=gr(1,cai)-spro*convec(1,i)
!        gr(2,cai)=gr(2,cai)-spro*convec(2,i)
!        gr(3,cai)=gr(3,cai)-spro*convec(3,i)
!       end do
!      endif
!
! End of gradient calculation , only done when not in mode 6
!
       endif

! JZ_UW12: Restore maxiter value after SMBP gradient calc       
       if (SMBP_GRAD) maxiter = maxiter_save

       end

! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
!
! QC_UW_1017: Based on XL_QC_UW15: NBO

       subroutine wrtnbo47(nn,ndim,x,ind,overl,occ,a,ham,
     &                     occdown,adown,hamdown)
                 !call nbo(nn,izp,ndim,occ,a,overl,lmax,ind,nel,qzeroscc,ham)
!      Need MDIM, NNDIM, qzeroscc, izp, nel 
       use sccdftbsrc, izp=>izp2
!      Need SCCZIN
       use sccdftb, only: SCCZIN
       use chm_kinds
       use consta
       use stream

       integer nn,ndim
       real*8 x(3,*)
       integer ind(NNDIM+1)
       real*8 overl(MDIM,MDIM)
       real*8 occ(MDIM),a(MDIM,MDIM),ham(MDIM,MDIM)
       real*8 occdown(MDIM),adown(MDIM,MDIM),hamdown(MDIM,MDIM)

       integer IMO

       integer elebasis(MDIM)
       character(len=10) s_nn,s_ndim,s_num, s_format
       character(len=500)s_center, s_center_num, s_label, s_label_num
       real*8 overlap(MDIM),density(MDIM,MDIM),dens(MDIM)
       real*8 h(MDIM)

       integer i,j,m,n,nelement


!      Prepare NBO47 file for subsequent NBO analysis

          IMO = 91
          ! convert nn and ndim to string
          write(s_nn,'(i5)')nn
          write(s_ndim,'(i5)')ndim

          !change the file name later
          OPEN(IMO,FILE='nbo.47',STATUS='REPLACE')
          !QC_UW1710: If UNPE .ne. ZERO then we need to write the OPEN Keyword
          !    in the GENNBO part.
          if (lcolspin) then 
            write(OUTU,*) "NBO> WRITE NBO47 FILE FOR OPEN SHELL CASE"
          write(imo,'(x,A)') '$GENNBO  NATOMS='//trim(adjustl(s_nn))
     &            //' NBAS='//trim(adjustl(s_ndim))//' UPPER OPEN BODM 
     &    FORMAT=PRECISE $END'
          else
            write(OUTU,*) "NBO> WRITE NBO47 FILE FOR CLOSED SHELL CASE"
          write(imo,'(x,A)') '$GENNBO  NATOMS='//trim(adjustl(s_nn))
     &            //'  NBAS='//trim(adjustl(s_ndim))//' UPPER BODM 
     &    FORMAT=PRECISE $END'
          endif 
          write(imo, '(x,A)') '$NBO $END'
          write(imo, '(x,A)') '$COORD'
          write(imo, '(x,A)') 'Title Feel free to revise'
          
          do i=1,nn
!           write(imo,'(2i5,3f12.6)')elenuc(i),int(qzeroscc(izp(i),4)),
            write(imo,'(2i5,3f12.6)')INT(scczin(i)),
     &                               int(qzeroscc(izp(i),4)),
     &                          x(1,i)*BOHRR,x(2,i)*BOHRR, x(3,i)*BOHRR
          enddo
          write(imo, '(x,A)') '$END'
          write(imo, '(x,A)') '$BASIS'
          s_center = 'CENTER = '
          do i=1,nn
            do j=1,ndim
              elebasis(j) = i
            enddo
            write(s_num,'(i5)')ind(i+1)-ind(i)
            s_format = '('//trim(adjustl(s_num))//'i5)'
            s_format = trim(adjustl(s_format))
            write(s_center_num,s_format)
     &          (elebasis(j),j=1,ind(i+1)-ind(i))
            s_center = trim(adjustl(s_center))//'    '
     &                 //trim(adjustl(s_center_num))
          enddo
          write(imo, '(x,A)') trim(s_center)
          s_label = 'LABEL =  '
          do i=1,nn
            if(ind(i+1)-ind(i) == 1) s_label_num = '1'
            if(ind(i+1)-ind(i) == 4) s_label_num = '1  101  102  103'
            if(ind(i+1)-ind(i) == 9) s_label_num = 
     &                     '1  101  102  103  251  252  253  254  255'
            s_label = trim(adjustl(s_label))//'    '//s_label_num
          enddo
          write(imo, '(x,A)') trim(s_label)
          write(imo, '(x,A)') '$END'

          !overlap matrix
          !QC: let's convert to the upper triangle
          !QC: It's actually a bit strange [looks like lower triangle]
          nelement=0
!         write (outu,*) "Writing Overlap:",ndim,nelement
          do m=1,ndim
!            do n=m,ndim
             do n=1,m
               nelement=nelement + 1
!              overlap((m-1)*ndim+n) = overl(m,n)
               overlap(nelement) = overl(m,n)
             enddo
          enddo

          !QC: we are going to write the precise format
          write(imo,'(x,A)')'$OVERLAP'
!         more = mod(ndim*ndim,3)
!         line = (ndim*ndim-more)/3
!         do m=1,line
!           write(imo,'(3f16.12)')overlap((m-1)*3+1),overlap((m-1)*3+2),
!    &                          overlap((m-1)*3+3)
!         enddo
!         if(more .ne. 0) then
!           if(more .eq. 1) write(imo,'(f16.12)')overlap(line*3+1)
!           if(more .eq. 2) write(imo,'(2f16.12)')overlap(line*3+1),
!    &                                        overlap(line*3+2)
!         endif
!         write(imo,'(4e20.12)') (overlap(i),i=1,ndim*ndim)
          write(imo,'(4e20.12)') (overlap(i),i=1,nelement)
          write(imo,'(x,A)')'$END'
!         write (outu,*) "Done writing Overlap:",ndim,nelement

!         write(outu,*) "NDIM> ",ndim
!         do i=1,ndim
!           write(outu,*) "OCC> ",i,occ(i)
!         enddo
!         if (lcolspin) then
!          do i=1,ndim
!            write(outu,*) "OCC> ",i,occdown(i)
!          enddo
!         endif
          ! density matrix
          density = 0.0
          do m=1,ndim
            do n=1,ndim
               do i=1,ndim
                 density(m,n)=density(m,n)+occ(i)*a(m,i)*a(n,i)
               enddo
            enddo
          enddo

!         QC: convert to upper triangle
          nelement = 0
!         write (outu,*) "Writing Density:",ndim,nelement
          do m=1,ndim
!           do n=m,ndim
            do n=1,m
              nelement = nelement + 1
!             dens((m-1)*ndim+n) = density(m,n)
              dens(nelement    ) = density(m,n)
            enddo
          enddo

          write(imo,'(x,A)')'$DENSITY'
!         do m=1,line
!           write(imo,'(3f16.12)')dens((m-1)*3+1),dens((m-1)*3+2),
!    &                          dens((m-1)*3+3)
!         enddo
!         if(more .ne. 0) then
!           if(more .eq. 1) write(imo,'(f16.12)')dens(line*3+1)
!           if(more .eq. 2) write(imo,'(2f16.12)')dens(line*3+1),
!    &                                      dens(line*3+2)
!         endif

          write(imo,'(4e20.12)') (dens(i),i=1,nelement)
!         write (outu,*) "Done writing Density:",ndim,nelement

          ! For open-shell, we need to write the density matrix for the
          ! beta spin as well
 
          if (lcolspin) then
           density = 0.0
           do m=1,ndim
             do n=1,ndim
                do i=1,ndim
                  density(m,n)=density(m,n)+
     &                        occdown(i)*adown(m,i)*adown(n,i)
                enddo
             enddo
           enddo

           nelement = 0  
!          write (outu,*) "Writing Density:",ndim,nelement
           do m=1,ndim
!            do n=m,ndim
             do n=1,m
               nelement = nelement + 1
               dens(nelement    ) = density(m,n)
             enddo
           enddo

!          do m=1,line
!           write(imo,'(3f16.12)')dens((m-1)*3+1),dens((m-1)*3+2),
!    &                          dens((m-1)*3+3)
!          enddo
!          if(more .ne. 0) then
!           if(more .eq. 1) write(imo,'(f16.12)')dens(line*3+1)
!           if(more .eq. 2) write(imo,'(2f16.12)')dens(line*3+1),
!    &                                      dens(line*3+2)
!          endif
          write(imo,'(4e20.12)') (dens(i),i=1,nelement)
!         write (outu,*) "Done writing Density:",ndim,nelement

          endif
          write(imo,'(x,A)')'$END'
  
          ! FOCK matrix
          nelement = 0
!         write (outu,*) "Writing Fock   :",ndim,nelement
          do m=1,ndim
!           do n=m,ndim
            do n=1,m
              nelement = nelement + 1
!             h((m-1)*ndim+n) = ham(m,n)
              h(nelement    ) = ham(m,n)
            enddo
          enddo
     
          write(imo,'(x,A)')'$FOCK'
!         do m=1,line
!           write(imo,'(3f16.12)')h((m-1)*3+1),h((m-1)*3+2),
!    &                      h((m-1)*3+3)
!         enddo
!         if(more .ne. 0) then
!           if(more .eq. 1) write(imo,'(f16.12)')h(line*3+1)
!           if(more .eq. 2) write(imo,'(2f16.12)')h(line*3+1),
!    &                                        h(line*3+2)
!         endif
          write(imo,'(4e20.12)') (h(i),i=1,nelement)
!         write (outu,*) "Done writing Fock:",ndim,nelement

          !For open-shell, write the FOCK for beta spin
          if (lcolspin) then
           nelement = 0
!          write (outu,*) "Writing Fock   :",ndim,nelement
           do m=1,ndim
!            do n=m,ndim
             do n=1,m
               nelement = nelement + 1
               h(nelement    ) = hamdown(m,n)
             enddo
           enddo
     
!          do m=1,line
!           write(imo,'(3f16.12)')h((m-1)*3+1),h((m-1)*3+2),
!    &                      h((m-1)*3+3)
!          enddo
!          if(more .ne. 0) then
!            if(more .eq. 1) write(imo,'(f16.12)')h(line*3+1)
!            if(more .eq. 2) write(imo,'(2f16.12)')h(line*3+1),
!    &                                        h(line*3+2)
!          endif
           write(imo,'(4e20.12)') (h(i),i=1,nelement)
!          write (outu,*) "Done writing Fock:",ndim,nelement

          endif
          write(imo,'(x,A)')'$END'
          close(91)

          end

