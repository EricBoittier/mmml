module cpe

    use chm_kinds
    use sccdftb, only: maxcen
    use sccdftbsrc, only: nndim

    implicit none
    
    double precision, dimension(6,4) :: cpepar_asc
    double precision, dimension(nndim,4) :: cpepar
    logical :: cpe_initialized = .false.

contains

  ! subroutine init_cpepar_asc

  !   implicit none

  !   integer :: u
  !   integer :: i, j

  !   double precision :: test

  !   u = 100

  !   open(unit = u, file = '/home/andersx/cpe.prm', &
  !     status = 'old', action = 'read')

  !   do i = 1, 6
  !       do j = 1, 4
  !           read(u,*) test
  !           ! write (*,*) "asc -- Reading in CPE parameter:", i, j, test
  !           cpepar_asc(i,j) = test

  !       end do
  !   end do
  !   
  !   close(u)

  ! end subroutine init_cpepar_asc

  subroutine init_cpepar(nn)

    use sccdftbsrc, only: lcpe0, lcpeq
    use psf, only: amass

    implicit none
    integer, intent(in) :: nn
    integer :: i
    if (lcpeq) then
      do i = 1,nn
        select case ( nint(amass(i)) )
          case (1) ! hydrogen
            cpepar(i,1) = 2.25514337801d0
            cpepar(i,2) = 0.856656353725d0
            cpepar(i,3) = 0.379639959606d0
            cpepar(i,4) = 0.37964127194d0

          case (12) ! carbon
            cpepar(i,1) = 1.47830882936d0
            cpepar(i,2) = 0.00486169565562d0
            cpepar(i,3) = 1.08623341882d0
            cpepar(i,4) = 2.35302538146d0

          case (16) ! oxygen
            cpepar(i,1) = 4.32274727143d0
            cpepar(i,2) = 0.0451795053077d0
            cpepar(i,3) = 3.4832795805d0
            cpepar(i,4) = 3.60500013486d0

          case (14) ! nitrogen
            cpepar(i,1) = 2.02928832645d0
            cpepar(i,2) = 0.32382207953d0
            cpepar(i,3) = 1.65113027173d0
            cpepar(i,4) = 2.29215452723d0

          case (32) ! sulfur
            cpepar(i,1) = 3.28533122396d0
            cpepar(i,2) = 1.86613640340d0
            cpepar(i,3) = 17.5557124853d0
            cpepar(i,4) = 1884.97885873d0

          case default
            print *, 'element ', i, ' is not recognized!'
            stop
        end select
      end do
    elseif (lcpe0) then
      do i = 1,nn
        select case ( nint(amass(i)) )
          case (1) ! hydrogen
            cpepar(i,1) = 1.33556595062d0
            cpepar(i,2) = 0.0d0
            cpepar(i,3) = 0.131508039276d0
            cpepar(i,4) = 5.37139270614d0

          case (12) ! carbon
            cpepar(i,1) = 1.23312078401d0
            cpepar(i,2) = 0.0d0
            cpepar(i,3) = 2.14685675866d0
            cpepar(i,4) = 6.50021713292d0

          case (16) ! oxygen
            cpepar(i,1) = 53.4194245215d0
            cpepar(i,2) = 0.0d0
            cpepar(i,3) = 3.55071484086d0
            cpepar(i,4) = 3.61753346033d0

          case (14) ! nitrogen
            cpepar(i,1) = 5.3496610575d0
            cpepar(i,2) = 0.0d0
            cpepar(i,3) = 5.84902091186d0
            cpepar(i,4) = 5.84955727703d0

          case (32) ! sulfur
            cpepar(i,1) = 1.4068335838d0
            cpepar(i,2) = 0.0d0
            cpepar(i,3) = 3.18339512453d0
            cpepar(i,4) = 3.18360536875d0

          case default
            print *, 'element ', i, ' is not recognized!'
            stop
        end select
      end do
    else
        write (*,*) "UNKNOWN CPE VERSION?"
    endif

  end subroutine init_cpepar

#if KEY_DFTBMKL==1

  subroutine cpe_scf(nn,qmat,qzero,izp,x,e_cpe,shiftcpe, &
                    c,dipol,cpedipol,cpedipabs,uhder,uhubb)
                    ! c,dipol,cpedipol,cpedipabs,uhder,uhubb,cpepar,field)
  ! ************************************************************************
  ! compute contribution of cpe response to the dftb hamiltonian
  ! implemented by s. kaminski jpc a. 2012 (116), 9131-9141.
  !
  ! there are two types of coulomb type integrals to be evaluated:
  ! 'm' is over 1 cpe basis function and 1 dftb3 basis function
  ! 'n' is over 2 cpe basis functions
  ! by dftb3 basis function i mean the slater function for the charge
  ! density fluctiations \delta\rho; cpe basis functions are gaussians,
  ! which are used to represent the cpe response density (eq.12, eq. 17).
  ! for each atom, there are 3 cpe basis functions, which are similar to
  ! p-orbitals (one can think of them as polarization functions).
  ! ***********************************************************************
  ! k.w. 2014-05-01

    use sccdftbsrc, only: nndim, maxtyp

    implicit none

    !-> input
    integer, intent(in) :: nn
    integer, dimension(nndim), intent(in) :: izp
    double precision, dimension(nndim), intent(in) :: qmat
    double precision, dimension(3), intent(in) :: dipol
    ! double precision, dimension(3), intent(in) :: field
    double precision, dimension(3,nndim), intent(in) :: x
    double precision, dimension(maxtyp,3), intent(in) :: uhder, uhubb
    double precision, dimension(maxtyp,4), intent(in) :: qzero

    ! output ->
    double precision, intent(out) :: e_cpe,cpedipabs      ! response energy contribution
    double precision, dimension(3*nn), intent(out) :: c   ! response dipole coefficients (zxy)
    double precision, dimension(nn), intent(out)   :: shiftcpe
    double precision, dimension(3), intent(out)    :: cpedipol
    ! double precision, dimension(nndim,4),intent(out) :: cpepar

    ! local
    integer :: i,j,k,l,i_k,i_l,i1,i2,j1,j2
    double precision, dimension(3) :: rab,ga,dga,gb,dgb,gc,dgc,gd,dgd,y
    double precision, dimension(3,3) :: p, pa, pb, dyda
    double precision, dimension(3,1) :: pc, pd, p2
    double precision, dimension(nn) :: tau, q_qm
    double precision, dimension(3*nn,3*nn) :: a, dadqa, dadqb, ai
    double precision, dimension(3*nn,nn) :: mmat, dmdqb
    double precision, dimension(3*nn) :: dmdqa, m, zw
    double precision :: s,f,za,zb,zc,zd,r2,r,b1,b2,br1,br2,c1,c2,j00,d00dr,d00dr2,j10
    double precision :: dedr,rlo,rhi,j11,e1,e2,dedb,dbdqa,dbdqb,d00db,dedqa,dedqb,de2db2
    double precision :: db2dqd,db2dqc,d10dqd,d10dqc,dedrdb,d00dr2db,d00drdb,d10db,d10dqa
    double precision :: d10dqb,d11dqa,d11dqb,dzadqa,dzbdqb,dzcdqc,dzddqd
    double precision, dimension(3), parameter :: xsz = (/ 0.4068833884920483d0, &
                                                          0.09179674341953627d0, &
                                                          3.515225231758639d0 /)
    double precision, dimension(3), parameter :: xsc = (/ 0.3620527755096057d0, &
                                                          0.6262050528632612d0, &
                                                          0.01174217162750757d0 /)
    ! double precision, parameter :: sqrt_pi = 1.77245385
    double precision, parameter :: sqrt_pi = 1.7724538509055
    double precision, parameter :: TO_DEBYE = 2.541765d0
    ! double precision, parameter :: to_debye = 2.541765d0


    do i = 1,nn
      q_qm(i) = qzero(izp(i),4) - qmat(i)
      ! exponent of the dftb slater functions
      tau(i) = (16.0d0/5.0d0) * (uhubb(izp(i),1) + uhder(izp(i),1) * q_qm(i))

      ! write (*,*) "TAU_DFTB", i, tau(i)
      ! write (*,*) "Qqm_DFTB", i, q_qm(i)

      ! asc: read masses from amass
      ! select case ( nint(xm(izp(i))) )
      ! select case ( nint(amass(i)) )
      !   case (1) ! hydrogen
      !     cpepar(i,1) = cpepar_asc(1,1) 
      !     cpepar(i,2) = cpepar_asc(1,2)
      !     cpepar(i,3) = cpepar_asc(1,3)
      !     cpepar(i,4) = cpepar_asc(1,4)

      !   case (12) ! carbon
      !     cpepar(i,1) = cpepar_asc(2,1) 
      !     cpepar(i,2) = cpepar_asc(2,2)
      !     cpepar(i,3) = cpepar_asc(2,3)
      !     cpepar(i,4) = cpepar_asc(2,4)

      !   case (16) ! oxygen
      !     cpepar(i,1) = cpepar_asc(3,1)
      !     cpepar(i,2) = cpepar_asc(3,2)
      !     cpepar(i,3) = cpepar_asc(3,3)
      !     cpepar(i,4) = cpepar_asc(3,4)

      !   case (14) ! nitrogen
      !     cpepar(i,1) = cpepar_asc(4,1)
      !     cpepar(i,2) = cpepar_asc(4,2)
      !     cpepar(i,3) = cpepar_asc(4,3)
      !     cpepar(i,4) = cpepar_asc(4,4)

      !   case (32) ! sulfur
      !     cpepar(i,1) = cpepar_asc(5,1)
      !     cpepar(i,2) = cpepar_asc(5,2)
      !     cpepar(i,3) = cpepar_asc(5,3)
      !     cpepar(i,4) = cpepar_asc(5,4)

      !   case (31) ! phosphorous
      !     cpepar(i,1) = cpepar_asc(6,1)
      !     cpepar(i,2) = cpepar_asc(6,2)
      !     cpepar(i,3) = cpepar_asc(6,3)
      !     cpepar(i,4) = cpepar_asc(6,4)

      !   case default
      !     print *, 'element ', i, ' is not recognized!'
      !     stop
      ! end select
    end do

    ! initialize some matrices involved in the integral evaluation
    a     = 0.d0
    ai    = 0.d0
    mmat  = 0.d0
    dmdqa = 0.d0
    dmdqb = 0.d0
    dadqa = 0.d0
    dadqb = 0.d0

    ! evaluate the second order matrix (see paper for reference)
    do i = 1,nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2
      za = cpepar(i,1) * exp( cpepar(i,2) * q_qm(i) )
      dzadqa = cpepar(i,2) * za

      do j = 1,nn
        j1 = (j-1)*3 + 1
        j2 = j1 + 2

        zb = cpepar(j,1) * exp( cpepar(j,2) * q_qm(j) )
        dzbdqb = cpepar(j,2) * zb

        rab = x(:,i) - x(:,j)

        ! ssjpp_intq (i have no idea, what that means, i just copied it)
        ga = xsz * za**2
        gb = xsz * zb**2
        dga = xsz * 2.0d0*za * dzadqa
        dgb = xsz * 2.0d0*zb * dzbdqb

        ! ggjpp_intq
        p = 0.d0
        pa = 0.d0
        pb = 0.d0

        ! r2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
        r2 = dot_product(rab,rab)

        if (r2 < 1.0d-25) then
          e1 = 0.d0
          dedqa = 0.d0
          dedqb = 0.d0

          do k = 1,3
            do l = 1,3
              b1 = sqrt( ga(k) * gb(l) / ( ga(k) + gb(l) ) )
              dbdqa = dga(k) * (gb(l) / (ga(k) + gb(l)))**2 / (2.0d0*b1)
              dbdqb = dgb(l) * (ga(k) / (ga(k) + gb(l)))**2 / (2.0d0*b1)
              c1 = xsc(k) * xsc(l)
              e1 = e1 + c1 * 4.0d0 * b1**3 / ( 3.0d0 * sqrt_pi )
              dedb = c1 * 12.0d0 * b1**2 / ( 3.0d0 * sqrt_pi )
              dedqa = dedqa + dedb * dbdqa
              dedqb = dedqb + dedb * dbdqb
            end do
          end do

          do k = 1,3
            p(k,k) = e1
            pa(k,k) = dedqa
            pb(k,k) = dedqb
          end do
        else

          r = sqrt(r2)
          y = rab/r

          do k = 1,3
            do l = 1,3
             dyda(k,l) = -y(k)*y(l)/r
            end do
            dyda(k,k) = dyda(k,k) + 1.d0/r
          end do

          j10 = 0.d0
          j11 = 0.d0
          d10dqa = 0.d0
          d10dqb = 0.d0
          d11dqa = 0.d0
          d11dqb = 0.d0

          do k = 1,3
            do l = 1,3
              b1      = sqrt( ga(k) * gb(l) / (ga(k) + gb(l)) )
              dbdqa   = dga(k) * (gb(l) / (ga(k) + gb(l)))**2 / (2.0d0*b1)
              dbdqb   = dgb(l) * (ga(k) / (ga(k) + gb(l)))**2 / (2.0d0*b1)
              br1     = b1 * r
              c1      = xsc(k) * xsc(l)
              e1      = (2.d0/sqrt_pi) * b1 * exp(-br1*br1)
              dedr    = -2*b1*br1 * e1
              dedb    = (2.0d0/sqrt_pi) * (1.0d0-2.0d0*br1*br1) * exp(-br1*br1)
              dedrdb  = -4.0d0*br1*e1 - 2.0d0*b1*br1*dedb
              j00     = erf(br1)/r
              d00dr   = e1/r - j00/r
              d00dr2  = dedr/r - e1/r2 - d00dr/r + j00/r2
              d00db   = (2.0d0/sqrt_pi) * exp(-br1*br1)
              d00drdb = dedb/r - d00db/r
              d00dr2db= dedrdb/r - dedb/r2 - d00drdb/r + d00db/r2
              j10     = j10 + c1 * d00dr
              j11     = j11 + c1 * d00dr2

              d10dqa  = d10dqa + c1 *  d00drdb * dbdqa
              d10dqb  = d10dqb + c1 *  d00drdb * dbdqb
              d11dqa  = d11dqa + c1 * d00dr2db * dbdqa
              d11dqb  = d11dqb + c1 * d00dr2db * dbdqb
            end do
          end do

          do k = 1,3
            i_k = mod(k+1,3)+1
            do l = 1,3
              i_l = mod(l+1,3)+1

              p(k,l)  =  j11 * y(i_k) * (-y(i_l)) - j10 * dyda(i_k,i_l)
              pa(k,l) =  d11dqa * y(i_k) * (-y(i_l)) - d10dqa * dyda(i_k,i_l)
              pb(k,l) =  d11dqb * y(i_k) * (-y(i_l)) - d10dqb * dyda(i_k,i_l)

            end do
          end do
        end if

        a(i1:i2, j1:j2)     = p
        dadqa(i1:i2, j1:j2) = pa
        dadqb(i1:i2, j1:j2) = pb

      end do
    end do

    ! first order matrix
    do i = 1,nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2

      zc = cpepar(i,1) * exp(cpepar(i,2) * (q_qm(i)))
      dzcdqc = cpepar(i,2) * zc

      do j = 1,nn
        rab = x(:,i) - x(:,j)
        zd  = tau(j)
        dzddqd = (16.0d0/5.0d0)*(uhder(izp(j),1))

        ! ssjps_intq
        gc  = xsz * zc**2
        gd  = xsz * zd**2
        dgc = xsz * 2.0d0*zc * dzcdqc
        dgd = xsz * 2.0d0*zd * dzddqd

        ! ggjps_intq
        p2 = 0.d0
        pc = 0.d0
        pd = 0.d0

        r2  = dot_product(rab,rab)

        if (r2 < 1.0d-25) then
          p2 = 0.d0
          pc = 0.d0
          pd = 0.d0
        else
          r = sqrt(r2)
          y = rab / r
          j10 = 0.d0
          d10dqc = 0.0d0
          d10dqd = 0.0d0

          do k = 1,3
            do l = 1,3
              b2      = sqrt( gc(k) * gd(l) / (gc(k) + gd(l)))
              db2dqc  = dgc(k) * (gd(l) / (gc(k) + gd(l)))**2 / (2.0d0*b2)
              db2dqd  = dgd(l) * (gc(k) / (gc(k) + gd(l)))**2 / (2.0d0*b2)
              br2     = b2 * r
              c2      = xsc(k) * xsc(l)
              e2      = (2.d0/sqrt_pi) * b2 * exp(-br2*br2)
              j00     = erf(br2) / r
              d00dr   = e2/r - j00/r
              d00db   = (2.0d0/sqrt_pi) * exp(-br2*br2)
              de2db2  = (2.0d0/sqrt_pi) * (1.0d0 - 2.0d0*br2*br2) * exp(-br2*br2)
              d00drdb = de2db2/r - d00db/r
              j10     = j10 + c2 * d00dr
              d10db   = c2 * d00drdb
              d10dqc  = d10dqc + d10db * db2dqc
              d10dqd  = d10dqd + d10db * db2dqd
            end do
          end do

          p2(1,1) = p2(1,1) + j10 * y(3)
          p2(2,1) = p2(2,1) + j10 * y(1)
          p2(3,1) = p2(3,1) + j10 * y(2)
          pc(1,1) = pc(1,1) + d10dqc * y(3)
          pc(2,1) = pc(2,1) + d10dqc * y(1)
          pc(3,1) = pc(3,1) + d10dqc * y(2)
          pd(1,1) = pd(1,1) + d10dqd * y(3)
          pd(2,1) = pd(2,1) + d10dqd * y(1)
          pd(3,1) = pd(3,1) + d10dqd * y(2)
        endif

        rlo = cpepar(i,3) + cpepar(j,3)
        rhi = cpepar(i,4) + cpepar(j,4)

        ! switching function
        if (r .ge. rhi) then
          s = 0.d0
        else if (r .le. rlo) then
          s = 1.d0
        else
          f = (rhi-r) / (rhi-rlo)
          s = 10.d0*f**3 - 15.d0*f**4 + 6.d0*f**5
        end if
        s = 1.d0 - s

        p2 = s * p2
        pc = s * pc
        pd = s * pd

        mmat(i1:i2, j)  = mmat(i1:i2, j)  + p2(1:3,1)
        dmdqa(i1:i2)    = dmdqa(i1:i2)    + pc(1:3,1) * q_qm(j)
        dmdqb(i1:i2, j) = dmdqb(i1:i2, j) + pd(1:3,1) * q_qm(j)

      end do
    end do

    call invert_eta_cpe(3*nn, a, ai)

    e_cpe = 0.d0
    shiftcpe = 0.d0
    c     = 0.d0

    m = matmul(mmat,q_qm)

    ! External electric field
    ! do i=1,nn
    !    j = (i-1)*3
    !    m(j+1) = m(j+1) - field(3) ! Z
    !    m(j+2) = m(j+2) - field(1) ! x
    !    m(j+3) = m(j+3) - field(2) ! y
    ! end do

    c = -matmul(ai,m)
    e_cpe = dot_product(c,m) + 0.5d0 * dot_product(c,matmul(a,c))

    do i = 1, nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2

      shiftcpe(i) = shiftcpe(i) + dot_product(c,mmat(:,i)) + dot_product(c(i1:i2),dmdqa(i1:i2))

      do j = 1,nn
        shiftcpe(j) = shiftcpe(j) + dot_product(c(i1:i2),dmdqb(i1:i2,j))
      end do
    end do

    do i = 1,nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2

      do j = 1,nn
        j1 = (j-1)*3 + 1
        j2 = j1 + 2

        shiftcpe(i) = shiftcpe(i) + 0.50d0 * dot_product( c(i1:i2), matmul(dadqa(i1:i2,j1:j2), c(j1:j2)) )
        shiftcpe(j) = shiftcpe(j) + 0.50d0 * dot_product( c(i1:i2), matmul(dadqb(i1:i2,j1:j2), c(j1:j2)) )
      end do
    end do

! corrected dipole moment
    zw = 0.0d0

    do i = 1,nn
      j = (i-1)*3
      zw(1) = zw(1) + c(j+1)
      zw(2) = zw(2) + c(j+2)
      zw(3) = zw(3) + c(j+3)
    end do

    ! cpedipol = 0.0d0
    ! cpedipol(1) = dipol(1) + (zw(2) * to_debye)
    ! cpedipol(2) = dipol(2) + (zw(3) * to_debye)
    ! cpedipol(3) = dipol(3) + (zw(1) * to_debye)

    ! ! norm of the dipole moment
    ! cpedipabs = sqrt(cpedipol(1) * cpedipol(1) +&
    !                  cpedipol(2) * cpedipol(2) +&
    !                  cpedipol(3) * cpedipol(3))

    cpedipol = 0.0d0
    cpedipol(1) = zw(2)
    cpedipol(2) = zw(3)
    cpedipol(3) = zw(1)

    ! norm of the dipole moment
    cpedipabs = sqrt(cpedipol(1) * cpedipol(1) +&
                     cpedipol(2) * cpedipol(2) +&
                     cpedipol(3) * cpedipol(3))
    
  end subroutine cpe_scf


  ! subroutine cpe_grad(nn,izp,x,qmat,qzero,cpepar,c,grad_cpe, uhubb, uhder, ndim,a, lmax,ind,occ, dedqi)
  subroutine cpe_grad(nn,izp,x,qmat,qzero,c,grad_cpe, uhubb, uhder, ndim,a, lmax,ind,occ, dedqi)
  ! compute the contribution of cpe to the gradient
  ! re-implementation of steve kaminski's code
  ! k.w. 2014-05-02

    use sccdftbsrc, only: nndim, maxtyp, ldim, mdim, racc

    implicit none

    ! -> input
    integer, dimension(nndim), intent(in) :: izp
    integer, intent(in) :: nn
    integer, intent(in) :: ndim
    integer, intent(in), dimension(maxtyp):: lmax
    integer, intent(in), dimension(nndim+1) :: ind
    
    double precision, dimension(mdim,mdim) :: a
    double precision, dimension(nndim), intent(in) :: qmat
    double precision, dimension(maxtyp,4), intent(in) :: qzero
    double precision, dimension(maxtyp,3), intent(in) :: uhubb, uhder
    double precision, dimension(3*nn), intent(in) :: c
    ! Not really inout, but we edit the coordinates and reset them
    ! underway, so inout is required, unfortunately.
    double precision, dimension(3,nndim), intent(inout) :: x
    ! double precision, dimension(nndim,4), intent(in) :: cpepar
    double precision, dimension(mdim), intent(in) :: occ
    double precision, dimension(nn), intent(in) :: dedqi ! Same as shiftCPE in eglcao.f

    ! output ->
    double precision, dimension(3,nndim), intent(out) :: grad_cpe

    ! local
    integer :: i,j,k,l,m,n,i_k,i_l,i1,i2,j1,j2

    double precision, dimension(3), parameter :: xsz = (/ 0.4068833884920483d0, &
                                                          0.09179674341953627d0, &
                                                          3.515225231758639d0 /)
    double precision, dimension(3), parameter :: xsc = (/ 0.3620527755096057d0, &
                                                          0.6262050528632612d0, &
                                                          0.01174217162750757d0 /)
    double precision, parameter :: sqrt_pi = 1.77245385090551588191942755656d0
    double precision, dimension(3) ::  y, dedra, rab, ga, gb
    double precision, dimension(3,1) :: p2
    double precision, dimension(3,3) :: dyda, p
    double precision, dimension(3,3,1) :: dp2
    double precision, dimension(3,3,3) :: dyda2(3,3,3), dp
    double precision, dimension(nn) :: q_qm, zeta_qm
    double precision :: j00,d00dr,d00dr2,d00dr3,j10,j11,j21,r2,r3,s,f,sf,ds,dsf
    double precision :: dedr,dedr2,dedr4,e1,e2,c1,c2,b1,b2,br1,br2,r,rlo,rhi,za,zb

    double precision, parameter :: deltax = 0.00001d0

    ! First index = (X/Y/Z), second index = Atom id, third = charge id
    double precision, dimension(3,ndim,ndim) :: cpe_dqdr
    double precision, dimension(ldim,ldim) :: s_plus,s_minus
    double precision, dimension(ldim,ldim) :: h_plus,h_minus

    double precision :: cpe_dsdr
    double precision :: cpe_sum
    double precision :: rcdx
    double precision :: dgrs
    double precision :: xhelp
    double precision :: dacc
    double precision :: c_mui,c_nui

    double precision :: dedqdqdr

    integer :: izpj,izpk,indj,indk
    integer :: inda,izpa,indb,izpb,lmaxa,lmaxb
    integer :: mu,nu
    integer :: ma,la,mb,lb,xyz
    integer :: aa,ab

    rcdx = 1.0d0/deltax

    dacc = 4.0d0 * racc

    ! Clear CPE dq/dr matrix
    grad_cpe = 0.0d0

    cpe_dqdr = 0.0d0

    do i = 1,nn
      q_qm(i) = qzero(izp(i),4) - qmat(i)
    end do

    do i = 1,nn
      zeta_qm(i)  = (16.0d0/5.0d0) *(uhubb(izp(i),1) + uhder(izp(i),1) * q_qm(i))
    end do

    ! 2nd order gradient
    do i = 1,nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2
      za = cpepar(i,1) * exp(cpepar(i,2) * q_qm(i))

      do j = 1,nn
        j1 = (j-1)*3 + 1
        j2 = j1 + 2
        zb = cpepar(j,1) * exp(cpepar(j,2) * q_qm(j))

        rab = x(:,i) -x(:,j)

        ga  = xsz * za**2
        gb  = xsz * zb**2
        p   = 0.0d0
        dp  = 0.0d0

        r2  = dot_product(rab,rab)

        e1 = 0.d0
        if (r2 < 1.0d-25) then

          do k = 1,3
            do l =1,3
               b1 = sqrt( ga(k) * gb(l) / (ga(k) + gb(l)) )
               c1 = xsc(k) * xsc(l)
               e1 = e1 + c1 * 4.d0 * b1**3 / (3.d0 * sqrt_pi)
            end do
          end do

          do k = 1,3
            p(k,k) = e1
          end do
        else
          r  = sqrt(r2)
          r3 = r * r2
          y  = rab / r

          do k = 1,3
            do l = 1,3
              dyda(k,l) = -y(k) * y(l) / r
            end do
            dyda(k,k) = dyda(k,k) + 1.d0 / r
          end do

          dyda2 = 0.0d0

          do k = 1,3
            do l = 1,3
              do m = 1,3
                dyda2(k,l,m) = -dyda(k,m) * y(l)/r  &
                               -y(k) * dyda(l,m)/r + (y(k) * y(l)/r2) * y(m)
              end do
            end do
            do l = 1,3
              dyda2(k,k,l) = dyda2(k,k,l) - y(l)/r2
            end do
          end do

          j10    = 0.0d0
          j11    = 0.0d0
          j21    = 0.0d0
          d00dr  = 0.0d0
          d00dr2 = 0.0d0
          d00dr3 = 0.0d0

          do k = 1,3
            do l = 1,3
              b1  = sqrt( ga(k) * gb(l) / (ga(k) + gb(l)) )
              br1 = b1 * r
              c1  = xsc(k) * xsc(l)
              j00 = erf(br1)/r
              e1  = (2.0d0/sqrt_pi) * b1  * exp(-br1*br1)
              dedr = -2.0d0*b1*br1 * e1
              dedr2 = -2.0d0*b1*b1 * e1 - 2.0d0*b1*br1 * dedr
              d00dr = e1/r - j00/r
              d00dr2 = dedr/r - e1/r2 - d00dr/r + j00/r2
              d00dr3 = dedr2/r - dedr/r2 - dedr/r2 + 2.0d0*e1/r3 &
                     - d00dr2/r + d00dr/r2 + d00dr/r2 - 2.0d0*j00/r3
              j10    = j10 + c1 * d00dr
              j11    = j11 + c1 * d00dr2
              j21    = j21 + c1 * d00dr3
            end do
          end do

          do k = 1,3
            i_k = mod(k+1,3) + 1
            do l = 1,3
              i_l = mod(l+1,3) + 1
              p(l,k) = j11 * y(i_l) * (-y(i_k)) - j10 * dyda(i_l,i_k)
            end do
          end do

          do k = 1,3
            i_k = mod(k+1,3) + 1
            do l = 1,3
              i_l = mod(l+1,3) + 1
              do m = 1,3
                dp(m,l,k) = j21 * y(m) * y(i_l) * (-y(i_k)) + j11 &
                          * dyda(i_l,m) * (-y(i_k)) + j11 * y(i_l) &
                          * (-dyda(i_k,m)) - j11 * y(m) * dyda(i_l,i_k) &
                          - j10 * dyda2(i_l,i_k,m)
              end do
            end do
          end do
        end if

        do k = 1,3
          dedra(k) = 0.5d0 * dot_product( c(i1:i2), matmul(dp(k,:,:),c(j1:j2)) )
        end do

        grad_cpe(:,i) = grad_cpe(:,i) + dedra
        grad_cpe(:,j) = grad_cpe(:,j) - dedra

      end do
    end do

    ! 1st order gradient
    do i = 1,nn
      i1 = (i-1)*3 + 1
      i2 = i1 + 2
      za = cpepar(i,1) * exp(cpepar(i,2) * q_qm(i))

      do j = 1,nn
        zb  = zeta_qm(j)
        rab = x(:,i) - x(:,j)
        ga  = xsz * za**2
        gb  = xsz * zb**2
        p2  = 0.0d0
        dp2 = 0.0d0

        r2  = dot_product(rab,rab)

        if (r2 < 1.0d-25) then
          p2 = 0.0d0
          dp2 = 0.0d0
        else
          r = sqrt(r2)
          y = rab/r

          do k = 1,3
            do l = 1,3
              dyda(k,l) = -y(k)*y(l)/r
            end do
              dyda(k,k) = dyda(k,k) + 1.d0/r
          end do

          j10 = 0.0d0
          j11 = 0.0d0
          j00 = 0.0d0
          d00dr = 0.0d0
          d00dr2 = 0.0d0

          do k = 1,3
            do l = 1,3
               b2 = sqrt(ga(k) * gb(l) / (ga(k) + gb(l)))
               br2 = b2 * r
               c2 = xsc(k) * xsc(l)
               e2 = (2.0d0/sqrt_pi) * b2 * exp(-br2*br2)
               dedr4  = -2.0d0*b2*br2 * e2
               j00    = erf(br2)/r
               d00dr  = e2/r - j00/r
               d00dr2 = dedr4/r - e2/r2 - d00dr/r + j00/r2
               j10    = j10 + c2 * d00dr
               j11    = j11 + c2 * d00dr2
            end do
          end do

          p2(1,1) = j10 * y(3)
          p2(2,1) = j10 * y(1)
          p2(3,1) = j10 * y(2)

          do k = 1,3
             dp2(k,1,1) = j11 * y(k) * y(3) + j10 * dyda(3,k)
             dp2(k,2,1) = j11 * y(k) * y(1) + j10 * dyda(1,k)
             dp2(k,3,1) = j11 * y(k) * y(2) + j10 * dyda(2,k)
          end do
        end if

        ! switching function
        rlo = cpepar(i,3) + cpepar(j,3)
        rhi = cpepar(i,4) + cpepar(j,4)

        if (r .ge. rhi) then
          s  = 0.0d0
          ds = 0.0d0
        else if (r .le. rlo) then
          s  = 1.0d0
          ds = 0.0d0
        else
          f  = (rhi-r)/(rhi-rlo)
          ds = -1.0d0 / (rhi-rlo)
          sf = 10.0d0 * f**3 - 15.0d0 * f**4 + 6.0d0 * f**5
          dsf = 30.0d0 * f**2 - 60.0d0 * f**3 + 30.0d0 * f**4
          s = sf
          ds = dsf * ds
        end if

        s  = 1.0d0 - s
        ds = - ds

        do k = 1,3
          dedra(k) = s * dot_product( c(i1:i2),dp2(k,:,1)*q_qm(j) )
        end do

        if (r > 0.0d0) then
          e2 = dot_product( c(i1:i2), p2(:,1)*q_qm(j) )
          dedra = dedra + ( ds * rab/r ) * e2
        end if

        grad_cpe(:,i) = grad_cpe(:,i) + dedra
        grad_cpe(:,j) = grad_cpe(:,j) - dedra

      end do
    end do

    ! q_a charge index
    do aa = 1,nn
      inda=ind(aa)
      izpa=izp(aa)
      lmaxa=lmax(izpa)

      ! atom b index
      do ab = 1,nn

        if (aa.eq.ab) cycle

        indb=ind(ab)
        izpb=izp(ab)
        lmaxb=lmax(izpb)

        ! Loop over occupied orbitals
        do i = 1,ndim 
           
          ! Stop looping over occupied orbitals if 
          ! occupation gets very small
          if (occ(i) .lt. dacc) cycle
          ! Hint: i = x or y or z
          do xyz = 1,3
            xhelp = x(xyz,ab)
            x(xyz,ab) = xhelp + deltax
            call slkmatrices(aa,ab,x,h_plus,s_plus)
            x(xyz,ab) = xhelp - deltax
            call slkmatrices(aa,ab,x,h_minus,s_minus)
            x(xyz,ab) = xhelp

            ! mu in a 
            do la = 1,lmaxa
              do ma=1,2*la-1

                m = (la-1)**2 + ma
                mu = m + inda

                ! nu in b
                do lb = 1,lmaxb
                  do mb = 1,2*lb-1

                    n = (lb-1)**2 + mb
                    nu = n + indb
    
                    dgrs = (s_plus(m,n) - s_minus(m,n))*rcdx * 0.5d0 &
                        * occ(i) * a(nu,i) * a(mu,i)

                    cpe_dqdr(xyz,ab,aa) = cpe_dqdr(xyz,ab,aa) + dgrs
                    cpe_dqdr(xyz,aa,aa) = cpe_dqdr(xyz,aa,aa) - dgrs

                  end do 
                end do 
              end do 
            end do 
          end do 
        end do 
      end do 
    end do 


    do k = 1,nn
      do xyz = 1,3
        dedqdqdr = 0.0d0
        do aa = 1,nn
          dedqdqdr = dedqdqdr + cpe_dqdr(xyz,k,aa) * dedqi(aa)
        end do
        grad_cpe(xyz,k) = grad_cpe(xyz,k) - dedqdqdr
      end do
    end do

  end subroutine cpe_grad



  subroutine invert_eta_cpe(n,a,ai)
    implicit none

    ! -> input
    integer, intent(in) :: n
    ! double precision, dimension(n,n), intent(out) :: a
    double precision, dimension(n,n), intent(inout) :: a

    ! output ->
    double precision, dimension(n,n), intent(out) :: ai

    ! local
    integer :: info,lwork,i,j
    double precision, parameter :: tol = 1.0d-10
    double precision, dimension(n,n) :: u, vt, wm
    double precision, dimension(n) :: w
    double precision, dimension(:), allocatable :: work
    double precision :: query

    ! logical :: use_svd = .false.
    logical :: use_svd = .false.

    ai = a
    info = 0

    if (.not. use_svd) then
      ! cholesky factorization of matrix ai
      call dpotrf('U', n, ai(1,1), n, info)
      if (info < 0) then
        write (6,*) 'dpotrf: argument had an illegal value:', info
      else if (info > 0) then
        write (6,*) 'dpotrf: factorization not completed:', info
      else
        ! inversion of the cholesky factorized matrix, however
        ! why only the first element???
        call dpotri('U', n, ai(1,1), n, info)
        if (info < 0) then
          write (6,*) 'dpotri: argument had an illegal value:', info
        else if (info > 0) then
          write (6,*) 'dpotri: factorization not completed:', info
          use_svd = .true.
        else
          do j = 2,n
            do i = 1,j-1
              ai(j,i) = ai(i,j)
            end do
          end do
        end if
      end if
    end if

    if (use_svd) then
      lwork = -1
      call dgesvd("A","A",n,n,ai(1,1),n,w(1),u(1,1),n,vt(1,1),n,query,lwork,info)
      if (info /= 0) then
        write (6, *) 'dgesvd not completed'
        stop
      end if

      lwork = nint(query)
      allocate( work(lwork) )
      w  = 0.d0
      u  = 0.d0
      vt = 0.d0

      call dgesvd("A","A",n,n, ai(1,1),n,w(1),u(1,1),n,vt(1,1), n, work(1), lwork, info)
      wm = 0.d0
      do i=1,n
        if ( dabs(w(i)) > tol) then
          wm(i,i) = 1.0d0 / w(i)
        end if
      end do

      ai = matmul( transpose(vt), matmul(wm,transpose(u)) )

    end if
  end subroutine invert_eta_cpe


#endif

end module cpe
