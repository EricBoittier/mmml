module pmeutil
  use chm_kinds
#if KEY_LJPME==1
  use iso_c_binding, only: c_ptr, c_null_ptr
#endif
  implicit none

  !  NFFT1,NFFT2,NFFT3 - the dimensions of the charge grid array
  !  SIZFFTAB - permanent 3d fft table storage
  !  SIZFFWRK - temporary 3d fft work storage
  !  FFTABLE,FFWORK - fft arrays

  real(chm_real),allocatable,dimension(:,:),save :: &
       theta1,theta2,theta3,dtheta1,dtheta2,dtheta3

  real(chm_real),allocatable,dimension(:),save :: &
       fft1_table,fft2_table,fft3_table,ffwork, &
       bsp_mod1,bsp_mod2,bsp_mod3
  integer,dimension(15),save :: ifac1,ifac2,ifac3

  integer,save :: NCorePME
  integer,save :: nfft1,nfft2,nfft3,sizfftab,sizffwrk
  integer, parameter :: MAX_PME_NODES=512
  integer, dimension(0:MAX_PME_NODES),save ::  &
       nxyslab,nxzslab,mxystart,mxzstart,rtaskt
  integer,save :: forder
  logical, dimension(0:MAX_PME_NODES),save :: me_first

  real(chm_real), allocatable, dimension(:),save ::  &
       transbuf1,transbuf2

  integer,save :: n3all,n2all,n3left,n2left,ntxyslab,ntxzslab
  integer,save :: mxyslabs,mxzslabs
  integer,save :: maxwk, inda, indz, tmpme, stmpme

  integer,save :: ierr_allocate

#if KEY_LJPME==1
  ! DPME ACS/FCP 2/2016
  logical :: QLJPME = .false. ! master flag for ljpme
  logical :: QCPME = .false. ! whether to use helPME for cPME electrostatics
  real(chm_real) :: DKAPPA ! kappa to use for dispersion term in ljpme
  real(chm_real) :: DKAPPA2 ! kappa^2 to use for dispersion term in ljpme
  real(chm_real) :: DKAPPA4 ! kappa^4 to use for dispersion term in ljpme
  real(chm_real) :: DKAPPA6 ! kappa^6 to use for dispersion term in ljpme
  real(chm_real) :: R6_INV_CUT ! 1/ rcut^6
  real(chm_real) :: DOPCUT ! 1 - (1 + rcut^2 dkappa^2 + 0.5 rcut^4 dkappa^4)*EXP(-rcut^2 dkappa^2)
  real(chm_real4) :: R6_INV_CUT_SP ! 1/ rcut^6
  real(chm_real4) :: DOPCUT_SP ! 1 - (1 + rcut^2 dkappa^2 + 0.5 rcut^4 dkappa^4)*EXP(-rcut^2 dkappa^2)
  integer :: DFFTX, DFFTY, DFFTZ, DORDER !  dispersion grid sizes and spline order
  ! The ksum extents for the compressed PME algorithm
  integer :: dmax,dmaxx,dmaxy,dmaxz
  integer :: qmax,qmaxx,qmaxy,qmaxz
  real(chm_real) :: DEWVIRIAL(9)
  type(c_ptr) :: dpmeInstance = c_null_ptr ! LJPME using helPME
  type(c_ptr) :: qpmeInstance = c_null_ptr ! electrostatic PME using helPME
#endif



contains


  !****************************************************************
  !                  ALLOCATE_BSPLINE
  !****************************************************************
  subroutine allocate_bspline(natom,nattot)
    integer,intent(in) :: natom,nattot
    integer :: alloc_err
    allocate(theta1(forder,nattot),theta2(forder,nattot), &
         theta3(forder,nattot), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to alloc theta arrays"

    allocate(dtheta1(forder,natom+1),dtheta2(forder,natom+1), &
         dtheta3(forder,natom+1), &
         stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to alloc dth arrays"

    return
  end subroutine allocate_bspline


  !****************************************************************
  !                  DEALLOCATE_BSPLINE
  !****************************************************************
  subroutine deallocate_bspline()
    integer :: alloc_err
    deallocate(dtheta1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate dtheta1"
    deallocate(dtheta2,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate dtheta2"
    deallocate(dtheta3,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate dtheta3"
    deallocate(theta1,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate theta1"
    deallocate(theta2,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate theta2"
    deallocate(theta3,stat=alloc_err)
    if(alloc_err /= 0 ) write(0,*)"unable to deallocate theta3"
    return
  end subroutine deallocate_bspline

  !****************************************************************
  !                  GET_BSPLINE
  !****************************************************************
  subroutine get_bspline(numatoms,fr1,fr2,fr3,xnsymm)

    ! this routine generates b-spline coefficients
    !
    !---------------------------------------------------------------------
    ! input:
    !      numatoms: number of atoms
    !      fr1,fr2,fr3 the scaled and shifted fractional coords
    !      order: the order of spline interpolation
    ! output
    !      theta1,theta2,theta3: the spline coeff arrays
    !      dtheta1,dtheta2,dtheta3: the 1st deriv of spline coeff arrays
    !---------------------------------------------------------------------

    integer numatoms,order
    real(chm_real) :: fr1(*),fr2(*),fr3(*)
    integer xnsymm

    real(chm_real) :: w
    integer n,i,itrans

    n=numatoms
    do itrans=2,xnsymm
       do i = 1,numatoms
          n=n+1
          w = fr1(n)-int(fr1(n))
          call fill_bspline(w,order,theta1(1,n),dtheta1(1,i))
          w = fr2(n)-int(fr2(n))
          call fill_bspline(w,order,theta2(1,n),dtheta2(1,i))
          w = fr3(n)-int(fr3(n))
          call fill_bspline(w,order,theta3(1,n),dtheta3(1,i))
       enddo
    enddo

    do n = 1,numatoms
       w = fr1(n)-int(fr1(n))
       call fill_bspline(w,order,theta1(1,n),dtheta1(1,n))
       w = fr2(n)-int(fr2(n))
       call fill_bspline(w,order,theta2(1,n),dtheta2(1,n))
       w = fr3(n)-int(fr3(n))
       call fill_bspline(w,order,theta3(1,n),dtheta3(1,n))
    enddo

    return
  end subroutine get_bspline

  !****************************************************************
  !                  FILL_BSPLINE
  !****************************************************************
  subroutine fill_bspline(w,order,array,darray)
    !---------- use standard b-spline recursions: see doc file
    use number, only: one, zero
    implicit none

    real(chm_real), intent(in) :: w
    integer, intent(in) :: order
    real(chm_real), dimension(1:order), intent(inout) :: array, darray

    integer :: j, k
    real(chm_real) :: div

    !--- do linear case
    array(order) = zero
    array(2) = w
    array(1) = one - w

    !--- compute standard b-spline recursion
    do k = 3,order-1
       div = one / (k-1)
       array(k) = div*w*array(k-1)
       do j = 1,k-2
          array(k-j) = div*((w+j)*array(k-j-1) + (k-j-w)*array(k-j))
       enddo
       array(1) = div*(1-w)*array(1)
    enddo

    !--- perform standard b-spline differentiation
    darray(1) = -array(1)
    do j = 2,order
       darray(j) = array(j-1) - array(j)
    enddo

    !--- one more recursion
    k=order
    div = one / (k-1)
    array(k) = div*w*array(k-1)
    do j = 1,k-2
       array(k-j) = div*((w+j)*array(k-j-1) + (k-j-w)*array(k-j))
    enddo
    array(1) = div*(1-w)*array(1)
    return
  end subroutine fill_bspline

  !****************************************************************
  !                        LOAD_BSP_MOD
  !****************************************************************
  subroutine load_bsp_mod()
    ! this routine loads the moduli of the inverse dft of the b splines
    ! bsp_mod1-3 hold these values, nfft1-3 are the grid dimensions,
    ! Order is the order of the B spline approx.
    use memory, only: chmalloc, chmdealloc
    use number, only: zero
    implicit none

    real(chm_real), allocatable, dimension(:) :: bsp_arr
    real(chm_real), dimension(forder) :: array, darray
    integer :: maxn, limit

    maxn = max(nfft1, nfft2, nfft3)
    call chmalloc("pmeutil.src", "load_bsp_mod", "bsp_arr", maxn, crl=bsp_arr)
    call fill_bspline(zero, forder, array, darray)
    bsp_arr(1:maxn) = zero

    limit = min(maxn, forder + 1)
    bsp_arr(2:limit) = array(1:(limit - 1))

    call dftmod(bsp_mod1, bsp_arr, nfft1)
    call dftmod(bsp_mod2, bsp_arr, nfft2)
    call dftmod(bsp_mod3, bsp_arr, nfft3)

    call chmdealloc("pmeutil.src", "load_bsp_mod", "bsp_arr", maxn, crl=bsp_arr)
    return
  end subroutine load_bsp_mod

  !****************************************************************
  !                        DFTMOD
  !****************************************************************
  SUBROUTINE DFTMOD(BSP_MOD,BSP_ARR,NFFT)

    ! Computes the modulus of the discrete fourier transform of bsp_arr,
    !  storing it into bsp_mod

  use number
  use consta
    integer,intent(in) :: nfft
    real(chm_real) :: bsp_mod(nfft),bsp_arr(nfft)


    integer j,k
    real(chm_real) :: sum1,sum2,arg,nfftr,arg1

    nfftr=twopi/nfft

    do k = 1,nfft
       sum1 = zero
       sum2 = zero
       arg1 = (k-1)*nfftr
       do j = 1,nfft
          arg = arg1*(j-1)
          sum1 = sum1 + bsp_arr(j)*cos(arg)
          sum2 = sum2 + bsp_arr(j)*sin(arg)
       enddo
       bsp_mod(k) = sum1**2 + sum2**2
    enddo
    do k = 1,nfft
       if ( bsp_mod(k)  <  rsmall ) &
            bsp_mod(k) = half*(bsp_mod(k-1) + bsp_mod(k+1))
    enddo
    do k = 1,nfft
       bsp_mod(k) = one/bsp_mod(k)
    enddo
    return
  end subroutine dftmod

  !****************************************************************
  !                        GET_SC_FRACT
  !****************************************************************
  subroutine get_sc_fract(fr1,fr2,fr3,cg1, &
       natom,nattot,x,y,z,recip, &
       xnsymm,maxsyml,xsymop,cg)

    ! This routine generates scales fractional coordinates.
    !
    ! INPUT:
    !      NATOM:  number of atoms
    !      NATTOT: number of atoms in unit cell including symmetry operations
    !      X,Y,Z:  arrays of cartesian coords
    !      RECIP:  the 3x3 array of reciprocal vectors stored as columns
    !      XNSYMM: The number of assymetric units in the crystal file
    !      MAXSYML: usually 192 (local copy, dimens has maxsym)
    !      XSYMOP: symmetry operator for all assymetric units
    ! OUTPUT:
    !     FR1,FR2,FR3 - the scaled and shifted fractional coords
    !     cg1:     charge array for all atoms in unit cell
    use number

    integer,intent(in) :: natom,nattot, xnsymm,maxsyml
    real(chm_real),intent(in) ::  X(natom),Y(natom),Z(natom),cg(natom)
    real(chm_real),intent(in) :: recip(3,3)
    integer,intent(in) :: xsymop(3,4,xnsymm)
    real(chm_real) ::  cg1(nattot), &
         fr1(nattot),fr2(nattot),fr3(nattot)

    integer n,m,itrans,i,j
    real(chm_real) :: wrk11,wrk21,wrk31,wrk12,wrk22,wrk32
    real(chm_real) :: wrk13,wrk23,wrk33,wrk14,wrk24,wrk34
    real(chm_real) :: rfact

    !--- convert to fractional coordinates
    wrk11=recip(1,1)
    wrk21=recip(2,1)
    wrk31=recip(3,1)
    wrk12=recip(1,2)
    wrk22=recip(2,2)
    wrk32=recip(3,2)
    wrk13=recip(1,3)
    wrk23=recip(2,3)
    wrk33=recip(3,3)
    do n = 1,natom
       fr1(n) = x(n)*wrk11+y(n)*wrk21+z(n)*wrk31
       fr2(n) = x(n)*wrk12+y(n)*wrk22+z(n)*wrk32
       fr3(n) = x(n)*wrk13+y(n)*wrk23+z(n)*wrk33
       cg1(n) = cg(n)
    enddo

    !--- generate other symmetric positions based on crystal data
    rfact = one / maxsyml
    m=natom
    do itrans=2,xnsymm
       wrk11=xsymop(1,1,itrans)
       wrk21=xsymop(2,1,itrans)
       wrk31=xsymop(3,1,itrans)
       wrk12=xsymop(1,2,itrans)
       wrk22=xsymop(2,2,itrans)
       wrk32=xsymop(3,2,itrans)
       wrk13=xsymop(1,3,itrans)
       wrk23=xsymop(2,3,itrans)
       wrk33=xsymop(3,3,itrans)
       wrk14=xsymop(1,4,itrans)*rfact
       wrk24=xsymop(2,4,itrans)*rfact
       wrk34=xsymop(3,4,itrans)*rfact

       do n = 1,natom
          m=m+1
          fr1(m) = fr1(n)*wrk11 + fr2(n)*wrk12 + fr3(n)*wrk13 + wrk14
          fr2(m) = fr1(n)*wrk21 + fr2(n)*wrk22 + fr3(n)*wrk23 + wrk24
          fr3(m) = fr1(n)*wrk31 + fr2(n)*wrk32 + fr3(n)*wrk33 + wrk34
          cg1(m) = cg(n)
       enddo
    enddo

    !--- Scale and shift the coordinates
    do n = 1,nattot
       fr1(n) = nfft1*(fr1(n) - anint(fr1(n)) + half)
       fr2(n) = nfft2*(fr2(n) - anint(fr2(n)) + half)
       fr3(n) = nfft3*(fr3(n) - anint(fr3(n)) + half)
    enddo

    return
  end subroutine get_sc_fract

  integer function GetNCorePME()
!     Returns log_2(NNOD())
  use parallel

    NCorePME = 1
    do while ( NCorePME < numnod )
       NCorePME = NCorePME * 2
    enddo
    
    if(NCorePME > numnod) then
       NCorePME = NCorePME/2
    endif
    
    GetNCorePME = NCorePME
    return
  end function GetNCorePME


  !****************************************************************
  !                        GET_FFTDIMS
  !****************************************************************
  subroutine get_fftdims(nfftdim1,nfftdim2,nfftdim3, &
       nfftable,nffwork)

  use stream
  use parallel
    integer nfftdim1,nfftdim2,nfftdim3,nffwork,nfftable
    integer n,nfftmax

    nfftmax = max(nfft1,nfft2,nfft3)
    nfftdim1 = nfft1
    n = nfft1/2
    if ( nfft1 == 2*n )then
       nfftdim1 = nfft1/2+2
    else
       call WRNDIE(-5,'<PMEUTIL>GET_FFTDIMS', &
            "For RealComplex FFT fftx must be even")
    endif

    nfftdim2 = nfft2
    n = nfft2/2
    if ( nfft2 == 2*n )nfftdim2 = nfft2+1
    nfftdim3 = nfft3
    n = nfft3/2
    if ( nfft3 == 2*n )nfftdim3 = nfft3+1
    nfftable = 4*nfftmax + 15
    nffwork = nfftmax

    !---   Parallel fft using pubfft in this file
    !---    pubfft does not need work space, 2D fft needs a complex
    !---    type work vector size nfftdim2. For some reason it is
    !---    doubled after the return to PMESH_SETUP before  is called
    !---    instead of now.
    nffwork = nfftdim2

    !--- Now add work space for the x-z slabs
    nffwork = nffwork + nfftdim1*nfft3*(nfft2/GetNCorePME() + 1)

    return
  end subroutine get_fftdims


  !****************************************************************
  !                        PLL_FFT_SETUP
  !****************************************************************
  subroutine pll_fft_setup(ldx,ldx2,ldx3)
    !---*********************************************************************
    !---            PLL_FFT_SETUP
    !---     Only run by MASTER PE at present
    !---   Sets the values of the variables in PME_PAR.FCM
    !---      These have the parameters of the distributed work in the
    !---      fft's and the other ewald routines.
    !---*********************************************************************
  use parallel
    integer n1, n2, n3, ldx, ldx2, ldx3, i, j,ntot
    integer base,base0,count,mynod0,idle,nRealCore

    !-----------------------------------------------------------------------
    !      Calculate the work distribution:
    !         each PE should get approx. the same number of slabs in the
    !            x-y direction.
    !         n3  =  total number of slabs
    !         n3all = number of slabs each pe gets
    !         n3left= number of slabs left over
    !                 these are given to the forst n3left PEs as
    !                 an increment to their slabs (n3all + 1)
    !         ntxyslab = total number of allocated spots in a slab
    !                    this is the number of complex values that has to
    !                    be passed per slab to each PE for 2D transforming
    !         nxyslab(ipe) = number of slabs ipe will have to do.
    !         mxyslabs = number of slabs this PE will have to do.
    !         mxzstart(ipe) = last slab of the previous pe.
    !
    !       Same type of variables for the xz slabs.
    !          n2all, n2left, ntxzslab,mxzslabs,nxzslab(0:255), mxzstart(0:255)
    !----------------------------------------------------------------------
    n1=nfft1
    n2=nfft2
    n3=nfft3
    maxwk = 2*n2
    ntxyslab = ldx * ldx2 *2
    ntxzslab = ldx * n3 *2

    if(numnod <= 1 ) then
       n3all=n3
       n3left=0
       n2all=n2
       n2left=0
       indz=maxwk

       nxyslab(0)  = n3all
       mxystart(0) = 0
       mxyslabs    = n3all

       nxzslab(0)  = n2all
       mxzstart(0) = 0
       mxzslabs    = n2all

       return
    endif

#if KEY_PARALLEL==1 /*pll*/
    nRealCore = GetNCorePME()
    NCorePME = nRealCore
    idle = 0

    if(mynod .ge. nRealCore) then 
       idle = 1
    endif    

    n3all  = n3/nRealCore
    n3left = mod(n3,nRealCore)

    n2all  = n2/nRealCore
    n2left = mod(n2,nRealCore)
    indz  = maxwk

    mxystart(0) = 0
    if(n3left == 0)then
       nxyslab(0) = n3all
       do i = 1, nRealCore-1
          nxyslab(i) = n3all
          mxystart(i) = mxystart(i-1) + nxyslab(i-1)
       enddo

       do i = nRealCore, numnod-1
          nxyslab(i) = 0
          mxystart(i) = mxystart(i-1) + nxyslab(i-1)
       enddo
    else
       nxyslab(0) = n3all + 1
       do i = 1, n3left-1
          nxyslab(i) = n3all+1
          mxystart(i) = mxystart(i-1) + nxyslab(i-1)
       enddo
       do i=max(n3left,1),nRealCore-1
          nxyslab(i) = n3all
          mxystart(i) = mxystart(i-1) + nxyslab(i-1)
       enddo
       do i=nRealCore,numnod-1
          nxyslab(i) = 0
          mxystart(i) = mxystart(i-1) + nxyslab(i-1)
       enddo
    endif

    mxyslabs = nxyslab(mynod)

    mxzstart(0) = 0
    if(n2left == 0)then
       nxzslab(0) = n2all
       do i = 1, nRealCore-1
          nxzslab(i) = n2all
          mxzstart(i) = mxzstart(i-1) + nxzslab(i-1)
       enddo
       do i = nRealCore, numnod-1
          nxzslab(i) = 0
          mxzstart(i) = mxzstart(i-1) + nxzslab(i-1)
       enddo
    else
       nxzslab(0) = n2all + 1
       do i=1,n2left-1
          nxzslab(i) = n2all+1
          mxzstart(i) = mxzstart(i-1) + nxzslab(i-1)
       enddo
       do i=max(n2left,1),nRealCore-1
          nxzslab(i) = n2all
          mxzstart(i) = mxzstart(i-1) + nxzslab(i-1)
       enddo
       do i=nRealCore,numnod-1
          nxzslab(i) = 0
          mxzstart(i) = mxzstart(i-1) + nxzslab(i-1)
       enddo
    endif

    mxzslabs = nxzslab(mynod)

    !--- Check if parallel run is power-of-2 number of processors.
    !--- the transpose in the parallel FFT currently requires
    !--- this restriction for setting up the communications
    !--- pattern list: rtaskt() and me_first() arrays

!    base=1
!    do while(numnod > base)
!       base=2*base
!    enddo
!    if(numnod < base) &
!         call WRNDIE(-5,'<PME>','non 2**N PE count in transpose')


   if(idle == 1) return


    count=0
    base=nRealCore
    base0=0
10  base=base/2
    mynod0=mynod-base0
    if(mynod0 >= base)then
       do i=0,base-1
          count=count+1
          me_first(count)=.false.
          rtaskt(count)=mod(mynod0-i,base)+base0
       enddo
       base0=base0+base
    else
       do i=0,base-1
          count=count+1
          me_first(count)=.true.
          rtaskt(count)=mod(mynod0+i,base)+base+base0
       enddo
    endif
    if(base > 2)goto 10
    count=count+1
    mynod0=mynod-base0
    if(mynod0 == 1)then
       me_first(count)=.false.
       rtaskt(count)=mod(nCorePME+mynod-1,nCorePME)
    else
       me_first(count)=.true.
       rtaskt(count)=mod(mynod+1,nCorePME)
    endif


    return
#endif /* (pll)*/
  end subroutine pll_fft_setup

  !
  !   +----------------------------------------------------------------+
  !   |**************************************************************  |
  !   |   *****************************************************        |
  !   |       ********************************************             |
  !   |              REAL - COMPLEX FFT STUFF HERE                     |
  !   |       ********************************************             |
  !   |    ****************************************************        |
  !   |**************************************************************  |
  !   +----------------------------------------------------------------+
  !
  !
  !**************************************************************
  !         Author: Michael F. Crowley
  !                 TSRI
  !                 July 2000
  !**************************************************************

  subroutine fft3d0rc(isign,scale, &
       x, ldx,ldx2,tmpy,alpha,beta)

    use new_timer,only:timer_start,timer_stop,T_FFTcomm  

  use number
  use parallel
#if KEY_PARALLEL==1 /*pll*/
#if KEY_CMPI==0
    use mpi    
#endif
    integer transbufsiz
#endif /* (pll)*/

    real(chm_real) ::  x(*)
    real(chm_real) ::  scale
    integer isign,  isys
    integer ldx,ldx2, n1, n2, n3
    integer mxz,mxy

    integer i, k, k0, ks
    integer j, ja, ja00, jz, jz0, jz00, jj, jidx, jtask
    !      common /zzewaldint/ i, k, k0, ks &
    !           , j,      ja,     ja00,  jz &
    !           , jz0,    jz00,   jj,    jidx &
    !           , jtask


    real(chm_real) :: tmpy(*),alpha(*),beta(*)
    integer n1x

    !
    !--------------------------------------------------------------
    !         startup: initialize tables for all three dimensions
    !
    n1=nfft1
    n2=nfft2
    n3=nfft3

    indz = 2*n2
    n1x = n1/2

    if(isign  == 0)then

       call fft2drc(0,n1,n2,one, x,ldx,ffwork,0, &
            tmpy,alpha,beta)
       call cffti(n3,fft3_table,ifac3)
#if KEY_PARALLEL==1 && KEY_ASYNC_PME==0
       call psync()           
#endif
       return
    endif

    !-----------------------------------------------------------------------
    ! each PE should do their 2D ffts now
    !-----------------------------------------------------------------------
    do j = 1, mxyslabs
       jj = (j-1)*ntxyslab+1

       call fft2drc(isign, n1, n2, ONE,  &
            x(jj),ldx,    &
            ffwork, isys &
            ,tmpy, &
            alpha,beta &
            )

    enddo
    !----------------------------------------------------------------------

#if KEY_PARALLEL==1
    if(numnod > 1)then
       mxz=(n2-1)/nCorePME + 1
       mxy=(n3-1)/nCorePME + 1
       transbufsiz=MXZ*MXY*LDX*2   !*4
       allocate(transbuf1(transbufsiz),transbuf2(transbufsiz))
       call timer_start(T_FFTcomm)                          
       call XY_ZX_transp(ffwork(indz+1),x,ldx,n3, &
            transbuf1,transbuf2, &
            transbufsiz)
       call timer_stop(T_FFTcomm)                          
       deallocate(transbuf1,transbuf2)
    endif
#endif 
    !----------------------------------------------------------------------
    do ks = 0,nxzslab(mynod)-1
       ja00 = (mxzstart(mynod)+ks)*ldx*2
       jz00 = ks*ntxzslab + 2*mxystart(mynod)
       do j = 0,nxyslab(mynod)-1
          jz = jz00 + 2*j +1
          ja = ja00 + j*ntxyslab+1
          do i = 0,n1x
             ffwork(indz+jz+i*n3*2) = x(ja+i*2)
             ffwork(indz+jz+1+i*n3*2) = x(ja+i*2+1)
          enddo
       enddo
    enddo
    !-----------------------------------------------------------------------
    !         END of TRANSPOSE
    !-----------------------------------------------------------------------
    !    Now do Z-FFTs
    !-----------------------------------------------------------------------
    do k = 0,mxzslabs-1
       k0 = k*ntxzslab
       do j = 0, n1x
          jidx=k0 + j*n3*2 +1
          call cfftf(n3, ffwork(indz+jidx), fft3_table,ifac3)
       enddo
    enddo
    !*************************************************
    !     Leave it   DISTRIBUTED AS Z-X SLABS
    !*************************************************
    do k = 1,ntxzslab*mxzslabs
       x(k) = ffwork(indz+k)
    enddo
    return




100 format(10e10.3)
101 format(i3,10f5.1)
4500 format(2e10.3)
  end subroutine fft3d0rc

  !**************************************************************
  !                   FFT3D_ZXYRC Complex-to-Real
  !**************************************************************
  !         Author: Michael F. Crowley
  !                 Pittsburgh Supercomputing Center
  !                 Oct 20, 1994
  !**************************************************************

  subroutine fft3d_zxyrc(isign, scale, &
       x, ldx,ldx2, y, ldy, ldy2,tmpy,alpha,beta)

    !****************************************************************
    !
    !  isign = 1 for forward fft
    !        =-1 for reverse fft
    !        = 0 initializes table for all three dimensions
    !  n1, n2, n3   dimensions of the fft to be performed
    !  scale  data will be scaled by this number on return
    !         see ccfft3d from Cray for details of how to use this one
    !  x, ldx, ldx2  complex 3-d array
    !                the input array with declared dimensions ldx and ldx2
    !                in the first two dimensions
    !  y, ldy, ldy2  complex 3-d array output 3D array
    !  table  real size 2*(n1+n2+n3) (for Cray CCFFT)
    !  ffwork   workspace size 4*( max(n1,n2,n3) ) (for Cray CCFFT)
    !                     + ldx * ldx2 * 2 * (n3/numnod + 1)
    !                     + ldx * n3   * 2 * (n2/numnod + 1)
    !  isys   use 0 (read cray docs on ccfft)
    !*************************************************************

    use new_timer,only:timer_start,timer_stop,T_FFTcomm    

    use number
    use parallel
#if KEY_PARALLEL==1 /*pll*/
#if KEY_CMPI==0
    use mpi     
#endif
    integer transbufsiz
#endif /* (pll)*/

    real(chm_real) :: x(*), y(*)
    integer ldx,ldx2, ldy, ldy2, n1, n2, n3
    integer k, k0, ks
    integer ja, ja00, jz, jz0, jz00, jj, jidx, jtask
    integer  i, j, isign, ndim, isys
    real(chm_real) ::  scale
    integer mxz,mxy

    integer n1x

    real(chm_real) :: alpha(*),beta(*),tmpy(*)

    !--------------------------------------------------------------
    !         startup: no initialization possible in this routine

    n1=nfft1
    n2=nfft2
    n3=nfft3
    if(isign  == 0)then
       write(6,*)"fft3d_zxy ERROR, cannot do an isign of 0, I QUIT"
       stop
    endif

    if (isign  /=  1 ) then
       write(6,*)'isign for 2nd fft should be 1'
       stop
    endif

#if KEY_PARALLEL==0
    mxyslabs=n3
    mxzslabs=n2
    ntxyslab = ldx * ldx2 *2
    ntxzslab = ldx * n3 *2
    indz = 2*n2
#endif 
    n1x=n1/2
    !***********************************************************************
    !       DISTRIBUTED AS Z-X SLABS, put the data into z area of work
    !***********************************************************************
    do k = 1,ntxzslab*mxzslabs
       ffwork(indz+k) = x(k)
    enddo
    !***********************************************************************
    !**** DO Z FFTs NOW **************************************************
    !***********************************************************************
    do k = 0,mxzslabs-1
       k0 = k*ntxzslab
       do j = 0, n1x
          jidx=k0 + j*n3*2 +1
          call cfftb(n3, ffwork(indz+jidx), fft3_table,ifac3)
       enddo
    enddo

    !***********************************************************************
    !*****  REDISTRIBUTE INTO XY SLABS *************************************
    !***********************************************************************
    ! Redistribute the data with strided shmem getsc
    !
    ! split into two do loops so that the same PEs are not trying
    ! to read each other at the same time.
    !***********************************************************************

#if KEY_PARALLEL==1
    !      call MPI8BARRIER(mpi_comm_world, ierr)
    !      call MPI_BARRIER(mpi_comm_world, ierr)
#if KEY_ASYNC_PME==0
    call psync()                                          
#endif

    if(numnod > 1)then
       MXZ=(N2-1)/nCorePME + 1
       MXY=(N3-1)/nCorePME + 1
       transbufsiz=MXZ*MXY*LDX*2    !*4
       allocate(transbuf1(transbufsiz*2))
       call timer_start(T_FFTcomm)                          
       call zx_xy_transp(x,ffwork(indz+1),ldx,n3,transbuf1, &
            transbufsiz)
       call timer_stop(T_FFTcomm)                           
       deallocate(transbuf1)
    endif
#endif 
    do ks = 0,nxzslab(mynod)-1
       ja00 = (mxzstart(mynod)+ks)*ldx*2
       jz00 = ks*ntxzslab + 2*mxystart(mynod)
       do j = 0,nxyslab(mynod)-1
          jz = jz00 + 2*j +1
          ja = ja00 + j*ntxyslab+1
          do i = 0,n1x
             x(ja+i*2) = ffwork(indz+jz+i*n3*2)
             x(ja+i*2+1) = ffwork(indz+jz+1+i*n3*2)
          enddo
       enddo
    enddo

    !-----------------------------------------------------------------------
    ! each PE should do their 2D ffts now
    !-----------------------------------------------------------------------

    do j = 1, mxyslabs
       jj = (j-1)*ntxyslab+1
       call fft2drc(isign, n1, n2, ONE,  &
            x(jj),ldx, ffwork, isys &
            ,tmpy, &
            alpha,beta &
            )
    enddo
#if KEY_PARALLEL==1
    !       call MPI8BARRIER(mpi_comm_world,ierr)
    !       call MPI_BARRIER(mpi_comm_world,ierr)
#if KEY_ASYNC_PME==0
    call psync()                             
#endif
#endif 

    return
100 format(10e10.3)
101 format(i3,10f5.1)
4500 format(2e10.3)
  end  subroutine fft3d_zxyrc

  !**************************************************************
  !**************************************************************
  !**************************************************************
  !
  !************ 2D  FFT real-complex-real ***********************
  !
  !**************************************************************
  !     sgi fft and  pubfft implementation for REAL-to-COMPLEX fft
  !**************************************************************
  !      subroutine fft2drc
  !   Complex version May 22 1995
  !**************************************************************
  subroutine fft2drc(isign,n1, n2, scale, x, ldx,  &
       work, isys, &
       tmpy,alpha,beta &
       )
    !**************************************************************
    !     1 PE non-distributed 2 dim. FFT
    !         calls a 1 dim FFT (CCFFT for the 1)
    !
    !         Author: Michael F. Crowley
    !                 Pittsburgh Supercomputing Center
    !                 Oct 20, 1994
    !**************************************************************
  use number
  use consta
  use parallel
    real(chm_real) :: work(*)
    real(chm_real) :: scale
    integer isys,isign,n1,n2,n3,ldx
    integer i, idx, idy, j, jy
    integer n1rc,kr,kkr,ki,kki,idt,n1x
    integer j1,j2,j3,j4,j1rl,j1im,j2rl,j2im,j3rl,j3im,j4rl,j4im,jdx
    integer istar
    real(chm_real) :: x(2, 0:ldx-1, 0:n2-1)
    real(chm_real) :: tmpy(2,0:ldx-1)
    real(chm_real) :: alpha(0:n1/2),beta(0:n1/2),a,b,c,d,pi2n,theta

    !---------------------------------------------------------------
    real(chm_real) :: en1

    en1=n1
    pi2n=two*pi/en1
    n1x=n1/2
    !=================================================
    !             initialize
    !
    if(isign == 0)then
       if(mod(n1,2)  /=  0)then
          call WRNDIE(-3,'<PME>', &
               'NEED factor 2 value for fftx for RC fft in fft2drc.')
       endif

       call cffti(n1x,fft1_table,ifac1)
       call cffti(n2,fft2_table,ifac2)

       return
    endif
    !-----------------------------------------------------
    do i=0,n1x-1
       theta=pi2n*i
       alpha(i) = cos(theta)
       beta(i)  = sin(theta)
    enddo
    !---------------------------------------------------------------
    !       Backward fft real to complex
    !---------------------------------------------------------------
    if(isign == -1)then
       !---------------------------------------------------------------
       !  First the x direction, the data is already contiguous
       !
       !-----------------------------------------------------------------------

       do j = 0,n2-1
          do i = 0, n1/2-1
             tmpy(1,i)=x(1,i,j)
             tmpy(2,i)=x(2,i,j)
          enddo

          call cfftf(n1x, tmpy(1,0), fft1_table,ifac1)

          do i = 1, n1x-1
             a =  .5d0*(tmpy(1,i)+tmpy(1,n1x-i)) ! Real F even
             b =  .5d0*(tmpy(2,i)-tmpy(2,n1x-i)) ! Imag F even
             c =  .5d0*(tmpy(2,i)+tmpy(2,n1x-i)) ! Real F odd
             d = -.5d0*(tmpy(1,i)-tmpy(1,n1x-i)) ! Imag F odd
             x(1,i,j)      = a + alpha(i)*c + beta(i)*d
             x(2,i,j)      = b + alpha(i)*d - beta(i)*c
          enddo
          !--------------------------------------------
          !     DC and nyquist
          x(1,0,j)=tmpy(1,0)+tmpy(2,0)
          x(2,0,j)=0.
          x(1,n1x,j)=tmpy(1,0)-tmpy(2,0)
          x(2,n1x,j)=0.
       enddo
       !-----------------------------------------------------------------------
       !     Now in the y direction, the data is in y now and
       !     we will put it into a contiguous 1D array first, transform,
       !     then put it back.
       !     ibegw should be adjusted to be thesize of the work
       !     area necessary for the machine specific fft.
       !     for pubfft, there is no work area used so ibegw=0
       !
       do j1 = 0, n1x
          i=2*j1+1
          istar=2*(n1-j1)
          do j2 = 0, n2-1
             j=2*j2+1
             work(j)   = x(1,j1,j2)
             work(j+1) = x(2,j1,j2)
          enddo
          call cfftf(n2, work(1), fft2_table,ifac2)
          do j2 = 0, n2-1
             j=2*j2+1
             x(1,j1,j2) = work(j)
             x(2,j1,j2) = work(j+1)
          enddo
       enddo
       !---------------------------------------------------------------
    else
       !---------------------------------------------------------------
       !     Forward fft complex to real
       !---------------------------------------------------------------
       !-----------------------------------------------------------------------
       !  Now in the y direction, the data is in y now and
       !     we will put it into a contiguous 1D array first, transform,
       !           then put it back.
       !     ibegw should be adjusted to be thesize of the work
       !           area necessary for the machine specific fft.
       !           for pubfft, there is no work area used so ibegw=0

       do i = 0, n1x
          do j = 0, n2-1

             work(2*j+1) = x(1,i,j)
             work(2*j+2) = x(2,i,j)
          enddo
          call cfftb(n2, work(1), fft2_table,ifac2)
          do j = 0, n2-1
             x(1,i,j) = work(2*j+1)
             x(2,i,j) = work(2*j+2)
          enddo
       enddo

       do j = 0,n2-1
          do i = 1, n1x-1
             a =  (x(1,i,j)+x(1,n1x-i,j)) ! Real F even
             b =  (x(2,i,j)-x(2,n1x-i,j)) ! Imag F even
             c =  (x(2,i,j)+x(2,n1x-i,j)) ! F odd contrib
             d =  (x(1,i,j)-x(1,n1x-i,j)) ! F odd contrib
             tmpy(1,i)      = a - alpha(i)*c - beta(i)*d
             tmpy(2,i)      = b + alpha(i)*d - beta(i)*c
          enddo
          tmpy(1,0) = (x(1,0,j)+x(1,n1x,j))
          tmpy(2,0) = (x(1,0,j)-x(1,n1x,j))
          call cfftb(n1x, tmpy(1,0), fft1_table,ifac1)
          do i = 0, n1x-1
             x(1,i,j)=tmpy(1,i)
             x(2,i,j)=tmpy(2,i)
          enddo
       enddo
       return
    endif

    return
  end subroutine fft2drc

#if KEY_PARALLEL==1 /*parallel_transpose*/
  !****************************************************************
  !     XY_ZX_TRANSP
  !****************************************************************
  subroutine XY_ZX_transp(targ,src,ldx,n3,tmp,tmp2,tmpsiz)
    use parallel
    use stream
#if KEY_MPI==1
    use mpi     
#endif
  
    integer tmpsiz
    real(chm_real) :: targ(*), src(*), tmp(tmpsiz),tmp2(tmpsiz)
    integer n3, ldx, ktask, numval,rtask
    integer i,j,k,ks,k0,k00,jtask,jjtask
    INTEGER check,kex(0:128,0:128)

    integer ierr

    if(numnod > MAX_PME_NODES)                    &
         CALL WRNDIE(-5,'<PME>','Too many nodes.')

    !--------------------------------------------------------
    !  --Start MATRIX TRANSPOSE--
!    do jjtask = 1,numnod-1
    do jjtask = 1,NCorePME-1
       if(mynod >= NCorePME) then
#if KEY_ASYNC_PME==0
          call psync()
#endif 
          cycle
       endif

       jtask = rtaskt(jjtask)
       numval = 0
       if(jtask /= mynod) then
          do i = 0,nxzslab(jtask)-1
             k00 = (mxzstart(jtask) + i )* ldx*2
             do j = 0, ldx-1
                k0 = k00 + j*2
                do ks = 0,nxyslab(mynod)-1
                   k = k0 + ks * ntxyslab + 1
                   numval = numval+1
                   tmp(numval) = src(k)
                   numval = numval+1
                   tmp(numval) = src(k+1)
                enddo
             enddo
          enddo
#if KEY_ASYNC_PME==0 /*async_io*/
          if(me_first(jjtask))then
#if KEY_MPI==1 /*mpitr*/
             IF(NUMVAL > 0) &
                  CALL MPI_SEND(tmp(1),numval,MPI_DOUBLE_PRECISION, &
                  jtask, 11, COMM_CHARMM, ierr)
#else /*  (mpitr)*/
             if(numval > 0) &
                  call GSEN(jtask, mynod,  tmp,  numval*8)
#endif /* (mpitr)*/
             call xy_zx_tr_rcv(targ,ldx,n3,tmp2,jtask)
             call psync()
          else
             call xy_zx_tr_rcv(targ,ldx,n3,tmp2,jtask)
#if KEY_MPI==1 /*mpitr2*/
             IF(NUMVAL > 0) &
                  CALL MPI_SEND(tmp(1),numval,MPI_DOUBLE_PRECISION, &
                  jtask, 11, COMM_CHARMM, ierr)
#else /* (mpitr2)*/
             if(numval > 0) &
                  call GSEN(jtask, mynod,  tmp,  numval*8)
#endif /* (mpitr2)*/
             call psync()
          endif
#else /* (async_io)*/
          CALL XY_ZX_TR_COMM(targ,ldx,n3,tmp2,jtask,tmp(1),numval)
#endif /* (async_io)*/
       endif
    enddo
    return
  end subroutine xy_zx_transp


  !******************************************************************
  !     XY_ZX_TR_COMM
  !******************************************************************
  subroutine xy_zx_tr_comm(targ,ldx,n3,foo,ktask,tsnd,lsnd)
  use parallel
  use machutil,only:eclock

    logical flag
    real(chm_real) :: targ(*), foo(*),tsnd(*),timmer
    integer n3, ldx, ktask, num_recv, numval,lsnd
    integer i,j,ks,k0,k00,jtask, ibuff,ifoo
    integer istart, iend

    timmer=eclock()
    numval = 2*ldx*nxzslab(mynod)*nxyslab(ktask) 
    CALL GRECSEN(ippmap(ktask),11,foo(1),numval,tsnd(1),lsnd)
    ! --MATRIX TRANSPOSE--
    numval = 0
    do i = 0,mxzslabs-1
       k00 = i*ntxzslab + mxystart(ktask)*2
       do j = 0, ldx-1
          k0 = k00 + j * n3*2
          do ks = 1,nxyslab(ktask)*2,2
             numval = numval+1
             targ(ks + k0) = foo(numval)
             numval = numval+1
             targ(ks + k0+1) = foo(numval)
          enddo
       enddo
    enddo
    tmeri(timecomm)=tmeri(timecomm)+eclock()-timmer
    return
  end subroutine xy_zx_tr_comm
  !****************************************************************
  !     XY_ZX_TR_RCV
  !****************************************************************
  subroutine xy_zx_tr_rcv(targ,ldx,n3,buf,ktask)
    use parallel
#if KEY_MPI==1
    use mpi 
#endif
    logical flag
    real(chm_real) :: targ(*), buf(*)
    integer n3, ldx, ktask, num_recv, numval
    integer i,j,ks,k0,k00
    integer istart, iend

#if KEY_MPI==1
    integer status(mpi_status_size),ierr
#endif 


#if KEY_MPI==1 /*mpi*/
    numval = 2*ldx*nxzslab(mynod)*nxyslab(ktask)
    if(numval == 0)return
    call MPI_RECV(buf(1), numval, MPI_DOUBLE_PRECISION,  &
         ktask, 11,  &
         COMM_CHARMM, status, ierr )
#else /*    (mpi)*/
    numval = 8*( 2*ldx*nxzslab(mynod)*nxyslab(ktask) )
    call GREC(ktask,   ktask,   buf(1), numval)
#endif /*   (mpi)*/

    ! --MATRIX TRANSPOSE--
    ! --MATRIX TRANSPOSE--
    numval = 0
    do i = 0,mxzslabs-1
       k00 = i*ntxzslab + mxystart(ktask)*2
       do j = 0, ldx-1
          k0 = k00 + j * n3*2
          do ks = 1,nxyslab(ktask)*2,2
             numval = numval+1
             targ(ks + k0) = buf(numval)
             numval = numval+1
             targ(ks + k0+1) = buf(numval)
          enddo
       enddo
    enddo
    return
  end subroutine xy_zx_tr_rcv


  !****************************************************************
  !     ZX_XY_TRANSP
  !****************************************************************
  subroutine zx_xy_transp(targ,src,ldx,n3,buf,bufsiz)
    use parallel
#if KEY_MPI==1
    use mpi     
#endif
    integer bufsiz
    real(chm_real) :: targ(*), src(*), buf(bufsiz,2)
    integer n3, ldx, ktask, rtask, numval
    integer i,j,k,ks,k0,k00,jtask, stmpm2
    INTEGER check,kex(0:128,0:128)
    integer ierr

#if KEY_MPIFFT==1
    integer ireq,isnd_stat(MPI_STATUS_SIZE)         
#endif
    integer jjtask

    if(numnod > MAX_PME_NODES)                    &
         CALL WRNDIE(-5,'<PME>','Too many nodes.')

    !----------------------------------------------------

    !     index into the buf array for the receive routine
    ! --MATRIX TRANSPOSE--
    stmpm2=stmpme/2+1
    do jjtask = 1,nCorePME-1
       if(mynod >= NCorePME) then
#if KEY_ASYNC_PME==0
          call psync()
#endif 
          cycle
       endif
    
       jtask = rtaskt(jjtask)
       numval = 0
       if(jtask /= mynod) then
          do i = 0,nxyslab(jtask)-1
             k00 = (mxystart(jtask) + i)*2
             do j = 0, nxzslab(mynod)-1
                k0 = k00 + j*ntxzslab
                do ks = 0,ldx-1
                   k = k0 + ks * n3*2 + 1
                   numval = numval+1
                   buf(numval,1) = src(k)
                   numval = numval+1
                   buf(numval,1) = src(k+1)
                enddo
             enddo
          enddo
#if KEY_ASYNC_PME==0 /*async_comm*/
          if(me_first(jjtask))then
#if KEY_MPI==1
             IF(NUMVAL > 0) &
                  CALL MPI_SEND(buf(1,1),numval,MPI_DOUBLE_PRECISION, &
                  jtask, 11, COMM_CHARMM, ierr)
#else /**/
             if (numval > 0) &
                  call GSEN(jtask, mynod,  buf(1,1),  numval*8)
#endif 
             call zx_trans_recv(targ,src,ldx,n3,buf(1,2),jtask)
             call psync()
          else
             call zx_trans_recv(targ,src,ldx,n3,buf(1,2),jtask)
#if KEY_MPI==1
             IF(NUMVAL > 0) &
                  CALL MPI_SEND(buf(1,1),numval,MPI_DOUBLE_PRECISION, &
                  jtask, 11, COMM_CHARMM, ierr)
#else /**/
             if (numval > 0) &
                  call GSEN(jtask, mynod,  buf(1,1),  numval*8)
#endif 
             call psync()
          endif
#else /* (async_comm)*/
          !.mh_fix            CALL ZX_TRANS_COMM(targ,src,ldx,n3,tmp(stmpm2),jtask,
          !.mh_fix     $           tmp(1),numval)
          CALL ZX_TRANS_COMM(targ,src,ldx,n3,buf(1,2),jtask, &
               buf(1,1),numval)
#endif /* (async_comm)*/
       endif
    enddo
    return
  end subroutine zx_xy_transp


  !******************************************************************
  !     ZX_TRANS_COMM
  !******************************************************************
  subroutine zx_trans_comm(targ,src,ldx,n3,foo,ktask,tsnd,lsnd)
  use parallel
  use machutil,only:eclock
    logical flag

    real(chm_real) :: targ(*), src(*), foo(*),tsnd(*),timmer
    integer n3, ldx, m1, m2, m3, ktask, num_recv, numval,lsnd
    integer i,j,k,ks,k0,k00,jtask, ibuff,ifoo
    integer istart, iend

    numval = 2*ldx*nxyslab(mynod)*nxzslab(ktask)
    if(numval == 0)return
    !
    timmer=eclock()

    call grecsen(ippmap(ktask),12,foo(1),numval,tsnd(1),lsnd)

    numval = 0
    ! --MATRIX TRANSPOSE--
    do i = 0,mxyslabs-1
       k00 = i*ntxyslab + mxzstart(ktask)*ldx*2
       do j = 0,nxzslab(ktask)-1
          k0 = k00 + j*ldx*2
          do ks = 1,ldx*2,2
             numval = numval+1
             targ(k0+ks) = foo(numval)
             numval = numval+1
             targ(k0+ks+1) = foo(numval)
          enddo
       enddo
    enddo
    !
    tmeri(timecomm)=tmeri(timecomm)+eclock()-timmer

    return
  end subroutine zx_trans_comm

  !****************************************************************
  !     ZX_TRANS_RECV
  !****************************************************************
  subroutine zx_trans_recv(targ,src,ldx,n3,buf,ktask)
    use parallel
#if KEY_MPI==1
    use mpi     
#endif
    logical flag

    real(chm_real) :: targ(*), src(*), buf(*)
    integer n3, ldx, m1, m2, m3, ktask, num_recv, numval
    integer i,j,k,ks,k0,k00
    integer istart, iend

#if KEY_MPI==1
    integer status(mpi_status_size),ierr
#endif 

    numval = 2*ldx*nxyslab(mynod)*nxzslab(ktask)

    if(numval == 0)return
#if KEY_MPI==1 /*mpi*/
#if KEY_MPIFFTZ==1 /*mpifftz*/
    call MPI_RECV(buf(1), numval, MPI_DOUBLE_PRECISION,  &
         ktask, ktask,  &
         COMM_CHARMM, status, ierr )
#else /*  (mpifftz)*/
    call MPI_RECV(buf(1), numval, MPI_DOUBLE_PRECISION,  &
         ktask, 11,  &
         COMM_CHARMM, status, ierr )
#endif /* (mpifftz)*/
#else /*  (mpi)*/
    call GREC(ktask,   ktask,   buf(1), 8*numval)
#endif /* (mpi)*/
    numval = 0
    ! --MATRIX TRANSPOSE--
    ! --MATRIX TRANSPOSE--
    do i = 0,mxyslabs-1
       k00 = i*ntxyslab + mxzstart(ktask)*ldx*2
       do j = 0,nxzslab(ktask)-1
          k0 = k00 + j*ldx*2
          do ks = 1,ldx*2,2
             numval = numval+1
             targ(k0+ks) = buf(numval)
             numval = numval+1
             targ(k0+ks+1) = buf(numval)
          enddo
       enddo
    enddo
    !-----------------------------------------------------------------------
    !
    return
  end subroutine zx_trans_recv
#endif /* (parallel_transpose)*/

#if KEY_CRAY_1DFFT==0 /*cray_main*/
  SUBROUTINE CFFTB(N,C,WSAVE,ifac)
    !
    integer,intent(in) :: n
    REAL(chm_real) :: C(*),WSAVE(4*n)
    integer,intent(inout) :: ifac(15)
    !
    INTEGER IW1
    !
    IF (N  ==  1) RETURN
    IW1 = N+N+1
    CALL CFFTB1(N,C,WSAVE,WSAVE(IW1),ifac)
    RETURN
  END subroutine cfftb

  SUBROUTINE CFFTF(N,C,WSAVE,ifac)
    !
    integer,intent(in) :: n
    real(chm_real) :: C(*),WSAVE(4*n)
    integer,dimension(15),intent(inout) :: ifac
    !
    INTEGER IW1, IW2
    !
    IF (N  ==  1) RETURN
    IW1 = N+N+1
    CALL CFFTF1(N,C,WSAVE,WSAVE(IW1),ifac)
    RETURN
  END subroutine cfftf

  SUBROUTINE CFFTI(N,table,ifac)
    !
    integer, intent(in) :: n
    real(chm_real) :: table(4*n)
    integer,intent(inout) :: ifac(15)
    !
    INTEGER IW1
    !
    IF (N  ==  1) RETURN
    IW1 = N+N+1
    CALL CFFTI1(N,table(iw1),ifac)
    RETURN
  END subroutine cffti

  SUBROUTINE CFFTB1(N,C,CH,WA,IFAC)
    !
    real(chm_real) :: CH(*),C(*),WA(*)
    INTEGER IFAC(*)
    !
    INTEGER IDL1, NA, NF, IP, IW, IDOT, IX2, IX3, IX4, I, N, NAC
    INTEGER IDO, K1, L1, L2, N2
    !
    NF = IFAC(2)
    NA = 0
    L1 = 1
    IW = 1
    DO K1=1,NF
       IP = IFAC(K1+2)
       L2 = IP*L1
       IDO = N/L2
       IDOT = IDO+IDO
       IDL1 = IDOT*L1
       IF (IP  ==  4) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IF (NA  ==  0) THEN
             CALL PASSB4(IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
          ELSE
             CALL PASSB4(IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  2) THEN
          IF (NA  ==  0) THEN
             CALL PASSB2(IDOT,L1,C,CH,WA(IW))
          ELSE
             CALL PASSB2(IDOT,L1,CH,C,WA(IW))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  3) THEN
          IX2 = IW+IDOT
          IF (NA  ==  0) THEN
             CALL PASSB3(IDOT,L1,C,CH,WA(IW),WA(IX2))
          ELSE
             CALL PASSB3(IDOT,L1,CH,C,WA(IW),WA(IX2))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  5) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IX4 = IX3+IDOT
          IF (NA  ==  0) THEN
             CALL PASSB5(IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ELSE
             CALL PASSB5(IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ENDIF
          NA = 1-NA
       ELSE
          CALL WRNDIE(-5,'<CFFTB1>','Illegal prime factor for PME FFT')
       ENDIF
       L1 = L2
       IW = IW+(IP-1)*IDOT
    ENDDO
    IF (NA  ==  0) RETURN
    N2 = N+N
    DO I=1,N2
       C(I) = CH(I)
    ENDDO
    RETURN
  END subroutine cfftb1

  SUBROUTINE CFFTF1(N,C,CH,WA,IFAC)
    !
    real(chm_real) :: CH(*),C(*),WA(*)
    INTEGER IFAC(*)
    !
    INTEGER IDL1, NA, NF, IP, IW, IDOT, IX2, IX3, IX4, I, N, NAC
    INTEGER IDO, K1, L1, L2, N2
    !
    NF = IFAC(2)
    NA = 0
    L1 = 1
    IW = 1
    DO K1=1,NF
       IP = IFAC(K1+2)
       L2 = IP*L1
       IDO = N/L2
       IDOT = IDO+IDO
       IDL1 = IDOT*L1
       IF (IP  ==  4) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IF (NA  ==  0) THEN
             CALL PASSF4(IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
          ELSE
             CALL PASSF4(IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  2) THEN
          IF (NA  ==  0) THEN
             CALL PASSF2(IDOT,L1,C,CH,WA(IW))
          ELSE
             CALL PASSF2(IDOT,L1,CH,C,WA(IW))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  3) THEN
          IX2 = IW+IDOT
          IF (NA  ==  0) THEN
             CALL PASSF3(IDOT,L1,C,CH,WA(IW),WA(IX2))
          ELSE
             CALL PASSF3(IDOT,L1,CH,C,WA(IW),WA(IX2))
          ENDIF
          NA = 1-NA
       ELSE IF (IP  ==  5) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IX4 = IX3+IDOT
          IF (NA  ==  0) THEN
             CALL PASSF5(IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ELSE
             CALL PASSF5(IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ENDIF
          NA = 1-NA
       ELSE
          CALL WRNDIE(-5,'<CFFTF1>','Illegal prime factor for PME FFT')
       ENDIF
       L1 = L2
       IW = IW+(IP-1)*IDOT
    ENDDO
    IF (NA  ==  0) RETURN
    N2 = N+N
    DO I=1,N2
       C(I) = CH(I)
    ENDDO
    RETURN
  END subroutine cfftf1

  subroutine cffti1(n,wa,ifac)
    !
  use number
  use consta
    integer n
    real(chm_real) :: wa(*)
    integer, intent(inout) :: ifac(15)
    logical ok


    integer ib, ld, ii, nf, ip, nl, nq, nr, idot
    integer ntry, i, j, ido, ipm, i1, k1, l1, l2
    real(chm_real) ::  fi, arg, argh, argld
    integer ntryh(4)
    data ntryh(1),ntryh(2),ntryh(3),ntryh(4)/3,4,2,5/

    ! Workaround for Intel IA64 compiler bug.
    integer :: ifac_tmp(15)

    nl = n
    nf = 0
    j = 0
    !  101 CONTINUE
    ok = .true.
    ok_loop: do while(ok)
       j = j+1
       if (j <= 4) then
          ntry = ntryh(j)
       else
          ntry = ntry+2
       endif
       !  104 continue
       nl_loop: do while(nl /= 1)
          if(mod(nl,ntry) /= 0) cycle ok_loop
          nf = nf+1
          ifac(nf+2) = ntry
          nl = nl/ntry
          if (ntry  ==  2) then
             if (nf  /=  1) then
                !                  do i=nf+1,3,-1
                !                     ifac(i+1)=ifac(i)
                !                  enddo
                ! Workaround for Intel IA64 compiler bug misvectorizing the above loop.
                ifac_tmp(1:15)=ifac(1:15)
                do i=nf+1,3,-1
                   ifac(i+1)=ifac_tmp(i)
                enddo

                ifac(3) = 2
             endif
          endif
       enddo nl_loop
       ok=.false.
    enddo ok_loop
    !      IF (NL  /=  1) GO TO 104
    ifac(1) = n
    ifac(2) = nf

    argh = twopi/n
    i = 2
    l1 = 1

    do k1=1,nf
       ip = ifac(k1+2)
       ld = 0
       l2 = l1*ip
       ido = n/l2
       idot = ido+ido+2
       ipm = ip-1
       do j=1,ipm
          i1 = i
          wa(i-1) = one
          wa(i) = zero
          ld = ld+l1
          fi = zero
          !            argld = real(ld,chm_real)*argh
          argld = ld*argh
          do ii=4,idot,2
             i = i+2
             fi = fi+one
             arg = fi*argld
             wa(i-1) = cos(arg)
             wa(i) = sin(arg)
          enddo
          if (ip  >  5) then
             wa(i1-1) = wa(i-1)
             wa(i1) = wa(i)
          endif
       enddo
       l1 = l2
    enddo
    return
  end subroutine cffti1

  SUBROUTINE PASSB2(IDO,L1,CC,CH,WA1)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,2,L1), CH(IDO,L1,2),WA1(*)
    !
    real(chm_real) :: TI2, TR2
    !
    IF (IDO  <=  2) THEN
       DO K=1,L1
          CH(1,K,1) = CC(1,1,K)+CC(1,2,K)
          CH(1,K,2) = CC(1,1,K)-CC(1,2,K)
          CH(2,K,1) = CC(2,1,K)+CC(2,2,K)
          CH(2,K,2) = CC(2,1,K)-CC(2,2,K)
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             CH(I-1,K,1) = CC(I-1,1,K)+CC(I-1,2,K)
             TR2 = CC(I-1,1,K)-CC(I-1,2,K)
             CH(I,K,1) = CC(I,1,K)+CC(I,2,K)
             TI2 = CC(I,1,K)-CC(I,2,K)
             CH(I,K,2) = WA1(I-1)*TI2+WA1(I)*TR2
             CH(I-1,K,2) = WA1(I-1)*TR2-WA1(I)*TI2
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passb2

  SUBROUTINE PASSB3(IDO,L1,CC,CH,WA1,WA2)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,3,L1), CH(IDO,L1,3), WA1(*), WA2(*)
    !
    real(chm_real) :: CI2, TAUI, CI3, DI2, DI3, CR2, TAUR, CR3
    real(chm_real) :: DR2, DR3, TI2, TR2
    DATA TAUR,TAUI /-.5D0,.866025403784439D0/
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TR2 = CC(1,2,K)+CC(1,3,K)
          CR2 = CC(1,1,K)+TAUR*TR2
          CH(1,K,1) = CC(1,1,K)+TR2
          TI2 = CC(2,2,K)+CC(2,3,K)
          CI2 = CC(2,1,K)+TAUR*TI2
          CH(2,K,1) = CC(2,1,K)+TI2
          CR3 = TAUI*(CC(1,2,K)-CC(1,3,K))
          CI3 = TAUI*(CC(2,2,K)-CC(2,3,K))
          CH(1,K,2) = CR2-CI3
          CH(1,K,3) = CR2+CI3
          CH(2,K,2) = CI2+CR3
          CH(2,K,3) = CI2-CR3
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TR2 = CC(I-1,2,K)+CC(I-1,3,K)
             CR2 = CC(I-1,1,K)+TAUR*TR2
             CH(I-1,K,1) = CC(I-1,1,K)+TR2
             TI2 = CC(I,2,K)+CC(I,3,K)
             CI2 = CC(I,1,K)+TAUR*TI2
             CH(I,K,1) = CC(I,1,K)+TI2
             CR3 = TAUI*(CC(I-1,2,K)-CC(I-1,3,K))
             CI3 = TAUI*(CC(I,2,K)-CC(I,3,K))
             DR2 = CR2-CI3
             DR3 = CR2+CI3
             DI2 = CI2+CR3
             DI3 = CI2-CR3
             CH(I,K,2) = WA1(I-1)*DI2+WA1(I)*DR2
             CH(I-1,K,2) = WA1(I-1)*DR2-WA1(I)*DI2
             CH(I,K,3) = WA2(I-1)*DI3+WA2(I)*DR3
             CH(I-1,K,3) = WA2(I-1)*DR3-WA2(I)*DI3
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passb3

  SUBROUTINE PASSB4(IDO,L1,CC,CH,WA1,WA2,WA3)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,4,L1),CH(IDO,L1,4),WA1(*),WA2(*),WA3(*)
    !
    real(chm_real) :: CI2, CI3, CI4, CR2, CR3, CR4
    real(chm_real) :: TI1, TI2, TI3, TI4, TR1, TR2, TR3, TR4
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TI1 = CC(2,1,K)-CC(2,3,K)
          TI2 = CC(2,1,K)+CC(2,3,K)
          TR4 = CC(2,4,K)-CC(2,2,K)
          TI3 = CC(2,2,K)+CC(2,4,K)
          TR1 = CC(1,1,K)-CC(1,3,K)
          TR2 = CC(1,1,K)+CC(1,3,K)
          TI4 = CC(1,2,K)-CC(1,4,K)
          TR3 = CC(1,2,K)+CC(1,4,K)
          CH(1,K,1) = TR2+TR3
          CH(1,K,3) = TR2-TR3
          CH(2,K,1) = TI2+TI3
          CH(2,K,3) = TI2-TI3
          CH(1,K,2) = TR1+TR4
          CH(1,K,4) = TR1-TR4
          CH(2,K,2) = TI1+TI4
          CH(2,K,4) = TI1-TI4
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TI1 = CC(I,1,K)-CC(I,3,K)
             TI2 = CC(I,1,K)+CC(I,3,K)
             TI3 = CC(I,2,K)+CC(I,4,K)
             TR4 = CC(I,4,K)-CC(I,2,K)
             TR1 = CC(I-1,1,K)-CC(I-1,3,K)
             TR2 = CC(I-1,1,K)+CC(I-1,3,K)
             TI4 = CC(I-1,2,K)-CC(I-1,4,K)
             TR3 = CC(I-1,2,K)+CC(I-1,4,K)
             CH(I-1,K,1) = TR2+TR3
             CR3 = TR2-TR3
             CH(I,K,1) = TI2+TI3
             CI3 = TI2-TI3
             CR2 = TR1+TR4
             CR4 = TR1-TR4
             CI2 = TI1+TI4
             CI4 = TI1-TI4
             CH(I-1,K,2) = WA1(I-1)*CR2-WA1(I)*CI2
             CH(I,K,2) = WA1(I-1)*CI2+WA1(I)*CR2
             CH(I-1,K,3) = WA2(I-1)*CR3-WA2(I)*CI3
             CH(I,K,3) = WA2(I-1)*CI3+WA2(I)*CR3
             CH(I-1,K,4) = WA3(I-1)*CR4-WA3(I)*CI4
             CH(I,K,4) = WA3(I-1)*CI4+WA3(I)*CR4
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passb4

  SUBROUTINE PASSB5(IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,5,L1), CH(IDO,L1,5),  &
         WA1(*), WA2(*), WA3(*), WA4(*)
    !
    real(chm_real) :: CI2, CI3, DI2, CI4, DI3, CI5, DI4, DI5,  &
         CR2, CR3, DR2
    real(chm_real) :: CR4, DR3, CR5, DR4, DR5,  &
         TI2, TI3, TI4, TI5, TR2, TR3
    real(chm_real) :: TR4, TR5, TI11, TI12, TR11, TR12
    !
    DATA TR11,TI11,TR12,TI12 /.309016994374947D0, .951056516295154D0, &
         -.809016994374947D0,.587785252292473D0/
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TI5 = CC(2,2,K)-CC(2,5,K)
          TI2 = CC(2,2,K)+CC(2,5,K)
          TI4 = CC(2,3,K)-CC(2,4,K)
          TI3 = CC(2,3,K)+CC(2,4,K)
          TR5 = CC(1,2,K)-CC(1,5,K)
          TR2 = CC(1,2,K)+CC(1,5,K)
          TR4 = CC(1,3,K)-CC(1,4,K)
          TR3 = CC(1,3,K)+CC(1,4,K)
          CH(1,K,1) = CC(1,1,K)+TR2+TR3
          CH(2,K,1) = CC(2,1,K)+TI2+TI3
          CR2 = CC(1,1,K)+TR11*TR2+TR12*TR3
          CI2 = CC(2,1,K)+TR11*TI2+TR12*TI3
          CR3 = CC(1,1,K)+TR12*TR2+TR11*TR3
          CI3 = CC(2,1,K)+TR12*TI2+TR11*TI3
          CR5 = TI11*TR5+TI12*TR4
          CI5 = TI11*TI5+TI12*TI4
          CR4 = TI12*TR5-TI11*TR4
          CI4 = TI12*TI5-TI11*TI4
          CH(1,K,2) = CR2-CI5
          CH(1,K,5) = CR2+CI5
          CH(2,K,2) = CI2+CR5
          CH(2,K,3) = CI3+CR4
          CH(1,K,3) = CR3-CI4
          CH(1,K,4) = CR3+CI4
          CH(2,K,4) = CI3-CR4
          CH(2,K,5) = CI2-CR5
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TI5 = CC(I,2,K)-CC(I,5,K)
             TI2 = CC(I,2,K)+CC(I,5,K)
             TI4 = CC(I,3,K)-CC(I,4,K)
             TI3 = CC(I,3,K)+CC(I,4,K)
             TR5 = CC(I-1,2,K)-CC(I-1,5,K)
             TR2 = CC(I-1,2,K)+CC(I-1,5,K)
             TR4 = CC(I-1,3,K)-CC(I-1,4,K)
             TR3 = CC(I-1,3,K)+CC(I-1,4,K)
             CH(I-1,K,1) = CC(I-1,1,K)+TR2+TR3
             CH(I,K,1) = CC(I,1,K)+TI2+TI3
             CR2 = CC(I-1,1,K)+TR11*TR2+TR12*TR3
             CI2 = CC(I,1,K)+TR11*TI2+TR12*TI3
             CR3 = CC(I-1,1,K)+TR12*TR2+TR11*TR3
             CI3 = CC(I,1,K)+TR12*TI2+TR11*TI3
             CR5 = TI11*TR5+TI12*TR4
             CI5 = TI11*TI5+TI12*TI4
             CR4 = TI12*TR5-TI11*TR4
             CI4 = TI12*TI5-TI11*TI4
             DR3 = CR3-CI4
             DR4 = CR3+CI4
             DI3 = CI3+CR4
             DI4 = CI3-CR4
             DR5 = CR2+CI5
             DR2 = CR2-CI5
             DI5 = CI2-CR5
             DI2 = CI2+CR5
             CH(I-1,K,2) = WA1(I-1)*DR2-WA1(I)*DI2
             CH(I,K,2) = WA1(I-1)*DI2+WA1(I)*DR2
             CH(I-1,K,3) = WA2(I-1)*DR3-WA2(I)*DI3
             CH(I,K,3) = WA2(I-1)*DI3+WA2(I)*DR3
             CH(I-1,K,4) = WA3(I-1)*DR4-WA3(I)*DI4
             CH(I,K,4) = WA3(I-1)*DI4+WA3(I)*DR4
             CH(I-1,K,5) = WA4(I-1)*DR5-WA4(I)*DI5
             CH(I,K,5) = WA4(I-1)*DI5+WA4(I)*DR5
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passb5

  SUBROUTINE PASSF2(IDO,L1,CC,CH,WA1)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,2,L1), CH(IDO,L1,2), WA1(*)
    !
    real(chm_real) :: TI2, TR2
    !
    IF (IDO  <=  2) THEN
       DO K=1,L1
          CH(1,K,1) = CC(1,1,K)+CC(1,2,K)
          CH(1,K,2) = CC(1,1,K)-CC(1,2,K)
          CH(2,K,1) = CC(2,1,K)+CC(2,2,K)
          CH(2,K,2) = CC(2,1,K)-CC(2,2,K)
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             CH(I-1,K,1) = CC(I-1,1,K)+CC(I-1,2,K)
             TR2 = CC(I-1,1,K)-CC(I-1,2,K)
             CH(I,K,1) = CC(I,1,K)+CC(I,2,K)
             TI2 = CC(I,1,K)-CC(I,2,K)
             CH(I,K,2) = WA1(I-1)*TI2-WA1(I)*TR2
             CH(I-1,K,2) = WA1(I-1)*TR2+WA1(I)*TI2
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passf2

  SUBROUTINE PASSF3(IDO,L1,CC,CH,WA1,WA2)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,3,L1), CH(IDO,L1,3), WA1(*), WA2(*)
    !
    real(chm_real) :: CI2, TAUI, CI3, DI2, DI3, CR2, TAUR
    real(chm_real) :: CR3, DR2, DR3, TI2, TR2
    !
    DATA TAUR,TAUI /-.5d0,-.866025403784439d0/
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TR2 = CC(1,2,K)+CC(1,3,K)
          CR2 = CC(1,1,K)+TAUR*TR2
          CH(1,K,1) = CC(1,1,K)+TR2
          TI2 = CC(2,2,K)+CC(2,3,K)
          CI2 = CC(2,1,K)+TAUR*TI2
          CH(2,K,1) = CC(2,1,K)+TI2
          CR3 = TAUI*(CC(1,2,K)-CC(1,3,K))
          CI3 = TAUI*(CC(2,2,K)-CC(2,3,K))
          CH(1,K,2) = CR2-CI3
          CH(1,K,3) = CR2+CI3
          CH(2,K,2) = CI2+CR3
          CH(2,K,3) = CI2-CR3
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TR2 = CC(I-1,2,K)+CC(I-1,3,K)
             CR2 = CC(I-1,1,K)+TAUR*TR2
             CH(I-1,K,1) = CC(I-1,1,K)+TR2
             TI2 = CC(I,2,K)+CC(I,3,K)
             CI2 = CC(I,1,K)+TAUR*TI2
             CH(I,K,1) = CC(I,1,K)+TI2
             CR3 = TAUI*(CC(I-1,2,K)-CC(I-1,3,K))
             CI3 = TAUI*(CC(I,2,K)-CC(I,3,K))
             DR2 = CR2-CI3
             DR3 = CR2+CI3
             DI2 = CI2+CR3
             DI3 = CI2-CR3
             CH(I,K,2) = WA1(I-1)*DI2-WA1(I)*DR2
             CH(I-1,K,2) = WA1(I-1)*DR2+WA1(I)*DI2
             CH(I,K,3) = WA2(I-1)*DI3-WA2(I)*DR3
             CH(I-1,K,3) = WA2(I-1)*DR3+WA2(I)*DI3
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passf3

  SUBROUTINE PASSF4(IDO,L1,CC,CH,WA1,WA2,WA3)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,4,L1),CH(IDO,L1,4),WA1(*),WA2(*),WA3(*)
    !
    real(chm_real) :: CI2, CI3, CI4, CR2, CR3, CR4
    real(chm_real) :: TI1, TI2, TI3, TI4, TR1, TR2, TR3, TR4
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TI1 = CC(2,1,K)-CC(2,3,K)
          TI2 = CC(2,1,K)+CC(2,3,K)
          TR4 = CC(2,2,K)-CC(2,4,K)
          TI3 = CC(2,2,K)+CC(2,4,K)
          TR1 = CC(1,1,K)-CC(1,3,K)
          TR2 = CC(1,1,K)+CC(1,3,K)
          TI4 = CC(1,4,K)-CC(1,2,K)
          TR3 = CC(1,2,K)+CC(1,4,K)
          CH(1,K,1) = TR2+TR3
          CH(1,K,3) = TR2-TR3
          CH(2,K,1) = TI2+TI3
          CH(2,K,3) = TI2-TI3
          CH(1,K,2) = TR1+TR4
          CH(1,K,4) = TR1-TR4
          CH(2,K,2) = TI1+TI4
          CH(2,K,4) = TI1-TI4
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TI1 = CC(I,1,K)-CC(I,3,K)
             TI2 = CC(I,1,K)+CC(I,3,K)
             TI3 = CC(I,2,K)+CC(I,4,K)
             TR4 = CC(I,2,K)-CC(I,4,K)
             TR1 = CC(I-1,1,K)-CC(I-1,3,K)
             TR2 = CC(I-1,1,K)+CC(I-1,3,K)
             TI4 = CC(I-1,4,K)-CC(I-1,2,K)
             TR3 = CC(I-1,2,K)+CC(I-1,4,K)
             CH(I-1,K,1) = TR2+TR3
             CR3 = TR2-TR3
             CH(I,K,1) = TI2+TI3
             CI3 = TI2-TI3
             CR2 = TR1+TR4
             CR4 = TR1-TR4
             CI2 = TI1+TI4
             CI4 = TI1-TI4
             CH(I-1,K,2) = WA1(I-1)*CR2+WA1(I)*CI2
             CH(I,K,2) = WA1(I-1)*CI2-WA1(I)*CR2
             CH(I-1,K,3) = WA2(I-1)*CR3+WA2(I)*CI3
             CH(I,K,3) = WA2(I-1)*CI3-WA2(I)*CR3
             CH(I-1,K,4) = WA3(I-1)*CR4+WA3(I)*CI4
             CH(I,K,4) = WA3(I-1)*CI4-WA3(I)*CR4
          ENDDO
       ENDDO
    ENDIF
    RETURN
  END subroutine passf4

  SUBROUTINE PASSF5(IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
    !
    INTEGER I, K, IDO, L1
    real(chm_real) :: CC(IDO,5,L1),CH(IDO,L1,5), &
         WA1(*),WA2(*),WA3(*),WA4(*)
    !
    real(chm_real) :: CI2, CI3, DI2, CI4, DI3, CI5, DI4, DI5, CR2, CR3
    real(chm_real) :: DR2, CR4, DR3, CR5, DR4, DR5, TI2, TI3, TI4, TI5
    real(chm_real) :: TR2, TR3, TR4, TR5, TI11, TI12, TR11, TR12
    !
    DATA TR11,TI11,TR12,TI12 /.309016994374947d0, -.951056516295154d0, &
         -.809016994374947d0,-.587785252292473d0/
    !
    IF (IDO  ==  2) THEN
       DO K=1,L1
          TI5 = CC(2,2,K)-CC(2,5,K)
          TI2 = CC(2,2,K)+CC(2,5,K)
          TI4 = CC(2,3,K)-CC(2,4,K)
          TI3 = CC(2,3,K)+CC(2,4,K)
          TR5 = CC(1,2,K)-CC(1,5,K)
          TR2 = CC(1,2,K)+CC(1,5,K)
          TR4 = CC(1,3,K)-CC(1,4,K)
          TR3 = CC(1,3,K)+CC(1,4,K)
          CH(1,K,1) = CC(1,1,K)+TR2+TR3
          CH(2,K,1) = CC(2,1,K)+TI2+TI3
          CR2 = CC(1,1,K)+TR11*TR2+TR12*TR3
          CI2 = CC(2,1,K)+TR11*TI2+TR12*TI3
          CR3 = CC(1,1,K)+TR12*TR2+TR11*TR3
          CI3 = CC(2,1,K)+TR12*TI2+TR11*TI3
          CR5 = TI11*TR5+TI12*TR4
          CI5 = TI11*TI5+TI12*TI4
          CR4 = TI12*TR5-TI11*TR4
          CI4 = TI12*TI5-TI11*TI4
          CH(1,K,2) = CR2-CI5
          CH(1,K,5) = CR2+CI5
          CH(2,K,2) = CI2+CR5
          CH(2,K,3) = CI3+CR4
          CH(1,K,3) = CR3-CI4
          CH(1,K,4) = CR3+CI4
          CH(2,K,4) = CI3-CR4
          CH(2,K,5) = CI2-CR5
       ENDDO
    ELSE
       DO K=1,L1
          DO I=2,IDO,2
             TI5 = CC(I,2,K)-CC(I,5,K)
             TI2 = CC(I,2,K)+CC(I,5,K)
             TI4 = CC(I,3,K)-CC(I,4,K)
             TI3 = CC(I,3,K)+CC(I,4,K)
             TR5 = CC(I-1,2,K)-CC(I-1,5,K)
             TR2 = CC(I-1,2,K)+CC(I-1,5,K)
             TR4 = CC(I-1,3,K)-CC(I-1,4,K)
             TR3 = CC(I-1,3,K)+CC(I-1,4,K)
             CH(I-1,K,1) = CC(I-1,1,K)+TR2+TR3
             CH(I,K,1) = CC(I,1,K)+TI2+TI3
             CR2 = CC(I-1,1,K)+TR11*TR2+TR12*TR3
             CI2 = CC(I,1,K)+TR11*TI2+TR12*TI3
             CR3 = CC(I-1,1,K)+TR12*TR2+TR11*TR3
             CI3 = CC(I,1,K)+TR12*TI2+TR11*TI3
             CR5 = TI11*TR5+TI12*TR4
             CI5 = TI11*TI5+TI12*TI4
             CR4 = TI12*TR5-TI11*TR4
             CI4 = TI12*TI5-TI11*TI4
             DR3 = CR3-CI4
             DR4 = CR3+CI4
             DI3 = CI3+CR4
             DI4 = CI3-CR4
             DR5 = CR2+CI5
             DR2 = CR2-CI5
             DI5 = CI2-CR5
             DI2 = CI2+CR5
             CH(I-1,K,2) = WA1(I-1)*DR2+WA1(I)*DI2
             CH(I,K,2) = WA1(I-1)*DI2-WA1(I)*DR2
             CH(I-1,K,3) = WA2(I-1)*DR3+WA2(I)*DI3
             CH(I,K,3) = WA2(I-1)*DI3-WA2(I)*DR3
             CH(I-1,K,4) = WA3(I-1)*DR4+WA3(I)*DI4
             CH(I,K,4) = WA3(I-1)*DI4-WA3(I)*DR4
             CH(I-1,K,5) = WA4(I-1)*DR5+WA4(I)*DI5
             CH(I,K,5) = WA4(I-1)*DI5-WA4(I)*DR5
          ENDDO
       ENDDO
    ENDIF
    return
  END subroutine passf5
#else /* (cray_main)*/
  subroutine pub_fft_dummy_stub()
    RETURN
  END subroutine pub_fft_dummy_stub
#endif /* (cray_main) IFN CRAY_1DFFT*/
end module pmeutil

