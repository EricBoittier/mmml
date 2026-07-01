#ifdef BONDLIST
  ! *
  ! * Build bondlist
  ! *
  subroutine KERNEL_NAME(nbondtbl, bondtbl, ib, jb, icb, glo2loc_ind, xyzq, bondlist)
    implicit none
    ! Input / Output
    integer, intent(in) :: nbondtbl, bondtbl(:), ib(:), jb(:), icb(:), glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(bondlist_t), intent(out) :: bondlist(:)
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#else
    real(chm_real4) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#endif
    integer i, j, ib_loc, jb_loc

#ifdef DOUBLE_PREC
    hbox(1) = hboxx
    hbox(2) = hboxy
    hbox(3) = hboxz
#else
    hbox(1) = real(hboxx, kind=chm_real4)
    hbox(2) = real(hboxy, kind=chm_real4)
    hbox(3) = real(hboxz, kind=chm_real4)
#endif
    min_hbox = -hbox

    if (nbondtbl.gt.0) then
!$omp parallel do schedule(static) private(i, j, ib_loc, jb_loc, xyz_i, xyz_j)
    do i=1,nbondtbl
       j = bondtbl(i)
       ib_loc = glo2loc_ind(ib(j))+1
       jb_loc = glo2loc_ind(jb(j))+1
       bondlist(i)%i = ib_loc-1
       bondlist(i)%j = jb_loc-1
       bondlist(i)%itype = icb(j)-1
       xyz_i = (/ xyzq(ib_loc)%x, xyzq(ib_loc)%y, xyzq(ib_loc)%z /)
       xyz_j = (/ xyzq(jb_loc)%x, xyzq(jb_loc)%y, xyzq(jb_loc)%z /)
#ifdef DOUBLE_PREC
       bondlist(i)%ishift = calc_ishift_pd(xyz_i, xyz_j, hbox, min_hbox)
#else
       bondlist(i)%ishift = calc_ishift_ps(xyz_i, xyz_j, hbox, min_hbox)
#endif
    enddo
!$omp end parallel do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef ANGLELIST
  ! *
  ! * Build anglelist
  ! *
  subroutine KERNEL_NAME(nangletbl, angletbl, it, jt, kt, ict, ctb, glo2loc_ind, xyzq, &
       anglelist, q_gromacs_style, q_charmm_style)
    implicit none
    ! Input / Output
    integer, intent(in) :: nangletbl, angletbl(:), it(:), jt(:), kt(:), ict(:)
    real(chm_real), intent(in) :: ctb(:)
    integer, intent(in) :: glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(anglelist_t), intent(out) :: anglelist(:)
    logical, intent(out) :: q_gromacs_style, q_charmm_style
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3), xyz_k(3)
#else
    real(chm_real4) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3), xyz_k(3)
#endif
    integer i, j, it_loc, jt_loc, kt_loc, ic
    logical style

#ifdef DOUBLE_PREC
    hbox(1) = hboxx
    hbox(2) = hboxy
    hbox(3) = hboxz
#else
    hbox(1) = real(hboxx, kind=chm_real4)
    hbox(2) = real(hboxy, kind=chm_real4)
    hbox(3) = real(hboxz, kind=chm_real4)
#endif
    min_hbox = -hbox

    q_gromacs_style = .false.
    q_charmm_style = .false.
! fix for domains with no angle terms
    if (nangletbl==0) q_charmm_style = .true. 

    if (nangletbl.gt.0) then
!$omp parallel do schedule(static) private(i, j, it_loc, jt_loc, kt_loc, ic, xyz_i, xyz_j, xyz_k, style) &
!$omp& reduction(.or.:q_gromacs_style, q_charmm_style)
    do i=1,nangletbl
       j = angletbl(i)
       it_loc = glo2loc_ind(it(j))+1
       jt_loc = glo2loc_ind(jt(j))+1
       kt_loc = glo2loc_ind(kt(j))+1
       ic = ict(j)
       anglelist(i)%i = it_loc-1
       anglelist(i)%j = jt_loc-1
       anglelist(i)%k = kt_loc-1
       anglelist(i)%itype = ic-1
       ! ctb(ic).ge.0 --> CHARMM. .lt.0 --> GROMACS cosine.
       style = (ctb(ic) < zero)
       q_gromacs_style = q_gromacs_style .or. style
       q_charmm_style = q_charmm_style .or. .not.style
       xyz_i = (/ xyzq(it_loc)%x, xyzq(it_loc)%y, xyzq(it_loc)%z /)
       xyz_j = (/ xyzq(jt_loc)%x, xyzq(jt_loc)%y, xyzq(jt_loc)%z /)
       xyz_k = (/ xyzq(kt_loc)%x, xyzq(kt_loc)%y, xyzq(kt_loc)%z /)
#ifdef DOUBLE_PREC
       anglelist(i)%ishift1 = calc_ishift_pd(xyz_i, xyz_j, hbox, min_hbox)
       anglelist(i)%ishift2 = calc_ishift_pd(xyz_k, xyz_j, hbox, min_hbox)
#else
       anglelist(i)%ishift1 = calc_ishift_ps(xyz_i, xyz_j, hbox, min_hbox)
       anglelist(i)%ishift2 = calc_ishift_ps(xyz_k, xyz_j, hbox, min_hbox)
#endif
    enddo
!$omp end parallel do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef DIHELIST
  ! *
  ! * Build dihelist
  ! *
  subroutine KERNEL_NAME(ndihetbl, dihetbl, ip, jp, kp, lp, icp, glo2loc_ind, xyzq, dihelist)
    implicit none
    ! Input / Output
    integer, intent(in) :: ndihetbl, dihetbl(:), ip(:), jp(:), kp(:), lp(:), icp(:), glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(dihelist_t), intent(out) :: dihelist(:)
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3), xyz_k(3), xyz_l(3)
#else
    real(chm_real4) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3), xyz_k(3), xyz_l(3)
#endif
    integer i, j, ip_loc, jp_loc, kp_loc, lp_loc

#ifdef DOUBLE_PREC
    hbox(1) = hboxx
    hbox(2) = hboxy
    hbox(3) = hboxz
#else
    hbox(1) = real(hboxx, kind=chm_real4)
    hbox(2) = real(hboxy, kind=chm_real4)
    hbox(3) = real(hboxz, kind=chm_real4)
#endif
    min_hbox = -hbox

    if (ndihetbl.gt.0) then
!$omp parallel do schedule(static) private(i, j, ip_loc, jp_loc, kp_loc, lp_loc, &
!$omp&                                     xyz_i, xyz_j, xyz_k, xyz_l)
    do i=1,ndihetbl
       j = dihetbl(i)
       ip_loc = glo2loc_ind(ip(j))+1
       jp_loc = glo2loc_ind(jp(j))+1
       kp_loc = glo2loc_ind(kp(j))+1
       lp_loc = glo2loc_ind(lp(j))+1
       dihelist(i)%i = ip_loc-1
       dihelist(i)%j = jp_loc-1
       dihelist(i)%k = kp_loc-1
       dihelist(i)%l = lp_loc-1
       dihelist(i)%itype = icp(j)-1
       xyz_i = (/ xyzq(ip_loc)%x, xyzq(ip_loc)%y, xyzq(ip_loc)%z /)
       xyz_j = (/ xyzq(jp_loc)%x, xyzq(jp_loc)%y, xyzq(jp_loc)%z /)
       xyz_k = (/ xyzq(kp_loc)%x, xyzq(kp_loc)%y, xyzq(kp_loc)%z /)
       xyz_l = (/ xyzq(lp_loc)%x, xyzq(lp_loc)%y, xyzq(lp_loc)%z /)
#ifdef DOUBLE_PREC
       dihelist(i)%ishift1 = calc_ishift_pd(xyz_i, xyz_k, hbox, min_hbox)
       dihelist(i)%ishift2 = calc_ishift_pd(xyz_j, xyz_k, hbox, min_hbox)
       dihelist(i)%ishift3 = calc_ishift_pd(xyz_l, xyz_k, hbox, min_hbox)
#else
       dihelist(i)%ishift1 = calc_ishift_ps(xyz_i, xyz_k, hbox, min_hbox)
       dihelist(i)%ishift2 = calc_ishift_ps(xyz_j, xyz_k, hbox, min_hbox)
       dihelist(i)%ishift3 = calc_ishift_ps(xyz_l, xyz_k, hbox, min_hbox)
#endif
    enddo
!$omp end parallel do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef XX14LIST
  ! *
  ! * Build in14list or ex14list
  ! *
  subroutine KERNEL_NAME(nxx14tbl, xx14tbl, xx14i, xx14j, glo2loc_ind, xyzq, xx14list)
    implicit none
    ! Input / Output
    integer, intent(in) :: nxx14tbl, xx14tbl(:), xx14i(:), xx14j(:), glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(list14_t), intent(out) :: xx14list(:)
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#else
    real(chm_real4) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#endif
    integer i, j, xx14i_loc, xx14j_loc

#ifdef DOUBLE_PREC
    hbox(1) = hboxx
    hbox(2) = hboxy
    hbox(3) = hboxz
#else
    hbox(1) = real(hboxx, kind=chm_real4)
    hbox(2) = real(hboxy, kind=chm_real4)
    hbox(3) = real(hboxz, kind=chm_real4)
#endif
    min_hbox = -hbox

    if (nxx14tbl.gt.0) then
!$omp parallel do schedule(static) private(i, j, xx14i_loc, xx14j_loc, xyz_i, xyz_j)
    do i=1,nxx14tbl
       j = xx14tbl(i)
       xx14i_loc = glo2loc_ind(xx14i(j))+1
       xx14j_loc = glo2loc_ind(xx14j(j))+1
       xx14list(i)%i = xx14i_loc-1
       xx14list(i)%j = xx14j_loc-1
       xyz_i = (/ xyzq(xx14i_loc)%x, xyzq(xx14i_loc)%y, xyzq(xx14i_loc)%z /)
       xyz_j = (/ xyzq(xx14j_loc)%x, xyzq(xx14j_loc)%y, xyzq(xx14j_loc)%z /)
#ifdef DOUBLE_PREC
       xx14list(i)%ishift = calc_ishift_pd(xyz_i, xyz_j, hbox, min_hbox)
#else
       xx14list(i)%ishift = calc_ishift_ps(xyz_i, xyz_j, hbox, min_hbox)
#endif
    enddo
!$omp end parallel do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef EX14THOLELIST
  ! *
  ! * Build ex14tholelist
  ! *
  subroutine KERNEL_NAME(nex14tholetbl, ex14tholetbl, ex14tholei, ex14tholej, ex14thole_aa,&
       glo2loc_ind, xyzq, ex14tholelist)
    implicit none
    ! Input / Output
    integer, intent(in) :: nex14tholetbl, ex14tholetbl(:), ex14tholei(:), ex14tholej(:)
    real(chm_real), intent(in) :: ex14thole_aa(:)
    integer, intent(in) :: glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(list14thole_t), intent(out) :: ex14tholelist(:)
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#else
    real(chm_real4) hbox(3), min_hbox(3), xyz_i(3), xyz_j(3)
#endif
    integer i, j, i_glo, j_glo, i_loc, j_loc, ithole_loc, jthole_loc

#ifdef DOUBLE_PREC
    hbox(1) = hboxx
    hbox(2) = hboxy
    hbox(3) = hboxz
#else
    hbox(1) = real(hboxx, kind=chm_real4)
    hbox(2) = real(hboxy, kind=chm_real4)
    hbox(3) = real(hboxz, kind=chm_real4)
#endif
    min_hbox = -hbox

    if (nex14tholetbl.gt.0) then
!$omp parallel do schedule(static) private(i, j, i_glo, j_glo, i_loc, j_loc, &
!$omp&                                     ithole_loc, jthole_loc, xyz_i, xyz_j)
    do i=1,nex14tholetbl
       j = ex14tholetbl(i)
       i_glo = ex14tholei(j)
       j_glo = ex14tholej(j)
       i_loc = glo2loc_ind(i_glo)+1
       j_loc = glo2loc_ind(j_glo)+1
       ithole_loc = glo2loc_ind(i_glo + 1)+1   ! Thole atoms are +1 in global index from the
       jthole_loc = glo2loc_ind(j_glo + 1)+1   ! host atoms.
       ex14tholelist(i)%i = i_loc-1
       ex14tholelist(i)%j = j_loc-1
       ex14tholelist(i)%ith = ithole_loc-1
       ex14tholelist(i)%jth = jthole_loc-1
       ex14tholelist(i)%aa = ex14thole_aa(j)
       ! NOTE: Since the thole atoms are in the same groups as the host atoms, we only
       !       need a single shift
       xyz_i = (/ xyzq(i_loc)%x, xyzq(i_loc)%y, xyzq(i_loc)%z /)
       xyz_j = (/ xyzq(j_loc)%x, xyzq(j_loc)%y, xyzq(j_loc)%z /)
#ifdef DOUBLE_PREC
       ex14tholelist(i)%ishift = calc_ishift_pd(xyz_i, xyz_j, hbox, min_hbox)
#else
       ex14tholelist(i)%ishift = calc_ishift_ps(xyz_i, xyz_j, hbox, min_hbox)
#endif
    enddo
!$omp end parallel do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef ISHIFT
  ! *
  ! * Calculates shift index for atoms i and j, that gives the minimum image such at
  ! * "xi + shift_x - xj" is the minimum image
  ! *
  integer function KERNEL_NAME(xyzi, xyzj, hbox, min_hbox)
    implicit none
    ! Input 
#ifdef DOUBLE_PREC
    real(chm_real), intent(in) :: xyzi(3), xyzj(3), hbox(3), min_hbox(3)
#else
    real(chm_real4), intent(in) :: xyzi(3), xyzj(3), hbox(3), min_hbox(3)
#endif
    ! Variables
#ifdef DOUBLE_PREC
    real(chm_real) dxyz(3)
#else
    real(chm_real4) dxyz(3)
#endif
    integer is(3)
    integer i

    dxyz = xyzi - xyzj
    is = 0
    
    ! is(1:3) = -1, 0, 1
    do i=1,3
       if (dxyz(i) >= hbox(i)) then
          is(i) = -1
       elseif (dxyz(i) < min_hbox(i)) then
          is(i) = 1
       endif
    enddo

    ! shift index = 0...26
#ifdef DOUBLE_PREC
    calc_ishift_pd = is(1)+1 + (is(2)+1)*3 + (is(3)+1)*9
#else
    calc_ishift_ps = is(1)+1 + (is(2)+1)*3 + (is(3)+1)*9
#endif

    return
  end function KERNEL_NAME
#endif

#ifdef MEMORY
  ! Needs:
  ! LISTNAME
  ! REALLOC_NAME
  ! ALLOC_NAME
  ! DEALLOC_NAME
  ! TYPENAME
  ! BOOLNAME
  ! NCAP

  ! *
  ! * Re-allocate with 50% extra space, but capped by NCAP
  ! *
  subroutine REALLOC_NAME(newsize_in)
    implicit none
    ! Input / Output
    integer, intent(in) :: newsize_in
    ! Variables
    integer newsize

    if (associated(LISTNAME)) then
       if (size(LISTNAME) < newsize_in) then
          call DEALLOC_NAME()
       endif
    endif

    if (.not.associated(LISTNAME)) then
       newsize = min(NCAP, int(newsize_in*1.5))
       if (newsize .gt. 0) then
          call ALLOC_NAME(newsize)
       endif
    endif

    return
  end subroutine REALLOC_NAME

  ! *
  ! * Allocates
  ! *
  subroutine ALLOC_NAME(newsize_in)
    implicit none
    ! Input / Output
    integer, intent(in) :: newsize_in

#if KEY_DOMDEC_GPU==1
    if (q_gpu .and. gpu_code_version==2) then
       BOOLNAME = .true.
       call alloc_gpu(LISTNAME, newsize_in)
    else
       BOOLNAME = .false.
#endif
       allocate(LISTNAME(newsize_in))
#if KEY_DOMDEC_GPU==1
    endif
#endif

    return
  end subroutine ALLOC_NAME

  ! *
  ! * Deallocates
  ! *
  subroutine DEALLOC_NAME()
    implicit none

    if (associated(LISTNAME)) then
#if KEY_DOMDEC_GPU==1
       if (BOOLNAME) then
          call dealloc_gpu(LISTNAME)
       else
#endif
          deallocate(LISTNAME)
#if KEY_DOMDEC_GPU==1
       endif
#endif
    endif

    return
  end subroutine DEALLOC_NAME

#endif
