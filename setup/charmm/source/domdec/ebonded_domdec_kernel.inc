
#ifdef BOND_LOOP
  ! *
  ! * Bonds loop
  ! *
  subroutine KERNEL_NAME(nbondlist, bondlist, cbb, cbc, xyzq, scoordtab, &
       fscale, epot, force, sforce)
!    use domdec_bonded_types,only:bondlist_t
!    use domdec_local_types,only:xyzq_dp_t  !##dp
!    use domdec_local_types,only:xyzq_sp_t  !##sp
    implicit none
    ! Input / Output
    integer, intent(in) :: nbondlist
    type(bondlist_t), intent(in) :: bondlist(:)
    real(chm_real), intent(in) :: cbb(:), cbc(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:), fscale
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:), fscale
#endif
    real(chm_real), intent(out) :: epot
    real(chm_real), intent(inout) :: force(:), sforce(:)
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, two = 2.0_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, two = 2.0_chm_real4
#endif
    ! Variables
    integer i, ii, jj, ii3, jj3
    integer ic, is
#ifdef DOUBLE_PREC
    real(chm_real) sx, sy, sz
    real(chm_real) dx, dy, dz
    real(chm_real) r, two_rinv, db, df
#else
    real(chm_real4) sx, sy, sz
    real(chm_real4) dx, dy, dz
    real(chm_real4) r, two_rinv, db, df
#endif
    real(chm_real) depot, fx, fy, fz

    epot = zero

    if (nbondlist.gt.0) then
!$omp barrier
!$omp do schedule(static)
    do i=1,nbondlist
       ii = bondlist(i)%i+1
       jj = bondlist(i)%j+1
       ic = bondlist(i)%itype+1
       is = bondlist(i)%ishift*3+1

       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)

       if (ic == 0) cycle
       if (cbc(ic) == 0.0_chm_real) cycle
       
       dx = xyzq(ii)%x + sx - xyzq(jj)%x
       dy = xyzq(ii)%y + sy - xyzq(jj)%y
       dz = xyzq(ii)%z + sz - xyzq(jj)%z

       r = sqrt(dx*dx + dy*dy + dz*dz)

       if (r == zero) cycle
       two_rinv = two/r
#ifdef DOUBLE_PREC
       db = r - cbb(ic)
       df = cbc(ic)*db
       depot = df*db
       df = df*two_rinv
#else
       db = r - real(cbb(ic),kind=chm_real4)
       df = real(cbc(ic),kind=chm_real4)*db
       depot = real(df*db,kind=chm_real)
       df = df*two_rinv
#endif

#if KEY_BLOCK==1
       df = df*fscale
#endif
       epot = epot + depot

#ifdef DOUBLE_PREC
       fx = dx*df
       fy = dy*df
       fz = dz*df
#else
       fx = real(dx*df,kind=chm_real)
       fy = real(dy*df,kind=chm_real)
       fz = real(dz*df,kind=chm_real)
#endif
       
       ii3 = 3*ii - 2
       jj3 = 3*jj - 2       

       ! Store forces
       force(ii3)   = force(ii3)   + fx
       force(ii3+1) = force(ii3+1) + fy
       force(ii3+2) = force(ii3+2) + fz
       
       force(jj3)   = force(jj3)   - fx
       force(jj3+1) = force(jj3+1) - fy
       force(jj3+2) = force(jj3+2) - fz
       
       ! Store shifted forces
       sforce(is)   = sforce(is)   + fx
       sforce(is+1) = sforce(is+1) + fy
       sforce(is+2) = sforce(is+2) + fz

    enddo
!$omp end do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef BOND_SOFT_LOOP
  ! *
  ! * Bonds loop
  ! *
  subroutine KERNEL_NAME(nbondlist, bondlist, cbb, cbc, xyzq, scoordtab, &
       fscale, soboalpha, epot, efsoft, force, sforce)
!    use domdec_bonded_types,only:bondlist_t
!    use domdec_local_types,only:xyzq_dp_t  !##dp
!    use domdec_local_types,only:xyzq_sp_t  !##sp
    implicit none
    ! Input / Output
    integer, intent(in) :: nbondlist
    type(bondlist_t), intent(in) :: bondlist(:)
    real(chm_real), intent(in) :: cbb(:), cbc(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:), fscale, soboalpha
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:), fscale, soboalpha
#endif
    real(chm_real), intent(out) :: epot, efsoft
    real(chm_real), intent(inout) :: force(:), sforce(:)
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, two = 2.0_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, two = 2.0_chm_real4
#endif
    ! Variables
    integer i, ii, jj, ii3, jj3
    integer ic, is
#ifdef DOUBLE_PREC
    real(chm_real) sx, sy, sz
    real(chm_real) dx, dy, dz
    real(chm_real) r, two_rinv, db, df
    real(chm_real) soft, kdb
#else
    real(chm_real4) sx, sy, sz
    real(chm_real4) dx, dy, dz
    real(chm_real4) r, two_rinv, db, df
    real(chm_real4) soft, kdb
#endif
    real(chm_real) depot, defsoft, fx, fy, fz

    epot = zero
    efsoft = zero

    if (nbondlist.gt.0) then
!$omp barrier
!$omp do schedule(static)
    do i=1,nbondlist
       ii = bondlist(i)%i+1
       jj = bondlist(i)%j+1
       ic = bondlist(i)%itype+1
       is = bondlist(i)%ishift*3+1

       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)

       if (ic == 0) cycle
       if (cbc(ic) == 0.0_chm_real) cycle
       
       dx = xyzq(ii)%x + sx - xyzq(jj)%x
       dy = xyzq(ii)%y + sy - xyzq(jj)%y
       dz = xyzq(ii)%z + sz - xyzq(jj)%z

       r = sqrt(dx*dx + dy*dy + dz*dz)

       if (r == zero) cycle
       two_rinv = two/r
#ifdef DOUBLE_PREC
       db = r - cbb(ic)
       soft = 1 / (1 + (1-fscale)*soboalpha*db*db)
       kdb = cbc(ic)*db
       depot = kdb*db*soft
       defsoft = fscale*depot*soft*soboalpha*db*db
       df = (kdb-depot*(1-fscale)*soboalpha*db)*soft*two_rinv
#else
       db = r - real(cbb(ic),kind=chm_real4)
       soft = 1 / (1 + (1-fscale)*soboalpha*db*db)
       kdb = real(cbc(ic),kind=chm_real4)*db
       depot = real(kdb*db*soft,kind=chm_real)
       defsoft = real(fscale*depot*soft*soboalpha*db*db,kind=chm_real)
       df = (kdb-depot*(1-fscale)*soboalpha*db)*soft*two_rinv
#endif

#if KEY_BLOCK==1
       df = df*fscale
#endif
       epot = epot + depot
       efsoft = efsoft + defsoft

#ifdef DOUBLE_PREC
       fx = dx*df
       fy = dy*df
       fz = dz*df
#else
       fx = real(dx*df,kind=chm_real)
       fy = real(dy*df,kind=chm_real)
       fz = real(dz*df,kind=chm_real)
#endif
       
       ii3 = 3*ii - 2
       jj3 = 3*jj - 2       

       ! Store forces
       force(ii3)   = force(ii3)   + fx
       force(ii3+1) = force(ii3+1) + fy
       force(ii3+2) = force(ii3+2) + fz
       
       force(jj3)   = force(jj3)   - fx
       force(jj3+1) = force(jj3+1) - fy
       force(jj3+2) = force(jj3+2) - fz
       
       ! Store shifted forces
       sforce(is)   = sforce(is)   + fx
       sforce(is+1) = sforce(is+1) + fy
       sforce(is+2) = sforce(is+2) + fz

    enddo
!$omp end do
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef ANGLE_LOOP
  ! *
  ! * Angles loop
  ! *
  subroutine KERNEL_NAME(nanglelist, anglelist, ctb, ctc, xyzq, scoordtab, &
       fscale, epot, force, sforce)
!    use domdec_bonded_types,only:anglelist_t
!    use domdec_local_types,only:xyzq_dp_t  !##dp
!    use domdec_local_types,only:xyzq_sp_t  !##sp
!    use consta,only:pi
    implicit none
    ! Input / Output
    integer, intent(in) :: nanglelist
    type(anglelist_t), intent(in) :: anglelist(:)
    real(chm_real), intent(in) :: ctb(:), ctc(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:), fscale
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:), fscale
#endif
    real(chm_real), intent(out) :: epot
    real(chm_real), intent(inout) :: force(:), sforce(:)
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real, two = 2.0_chm_real
    real(chm_real), parameter :: mintwo = -2.0_chm_real, sixth = 1.0_chm_real/6.0_chm_real
    real(chm_real), parameter :: pt001 = 1.0e-3_chm_real, rpreci = 2.22045d-16
    real(chm_real), parameter :: pi_val = pi, pt99 = 0.999999_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4, two = 2.0_chm_real4
    real(chm_real4), parameter :: mintwo = -2.0_chm_real4, sixth = 1.0_chm_real4/6.0_chm_real4
    real(chm_real4), parameter :: pt001 = 1.0e-3_chm_real4, rpreci = 1.1921E-7
    real(chm_real4), parameter :: pi_val = real(pi,kind=chm_real4), pt99 = 0.999999_chm_real4
#endif
    ! Variables
    integer i
    integer ii, jj, kk, ic, is, ks
    integer ii3, jj3, kk3
#ifdef DOUBLE_PREC
    real(chm_real) sxi, syi, szi
    real(chm_real) sxk, syk, szk
    real(chm_real) dxij, dyij, dzij
    real(chm_real) dxkj, dykj, dzkj
    real(chm_real) rij, rkj, rij_inv, rkj_inv
    real(chm_real) dxijr, dyijr, dzijr
    real(chm_real) dxkjr, dykjr, dzkjr
    real(chm_real) cst, at, da, df, st2r, str
    real(chm_real) ctb_val, ctc_val
    real(chm_real) dtxi, dtyi, dtzi
    real(chm_real) dtxj, dtyj, dtzj    
#else
    real(chm_real4) sxi, syi, szi
    real(chm_real4) sxk, syk, szk
    real(chm_real4) dxij, dyij, dzij
    real(chm_real4) dxkj, dykj, dzkj
    real(chm_real4) rij, rkj, rij_inv, rkj_inv
    real(chm_real4) dxijr, dyijr, dzijr
    real(chm_real4) dxkjr, dykjr, dzkjr
    real(chm_real4) cst, at, da, df, st2r, str
    real(chm_real4) ctb_val, ctc_val
    real(chm_real4) dtxi, dtyi, dtzi
    real(chm_real4) dtxj, dtyj, dtzj
#endif
    real(chm_real) dti, dtj, depot

    epot = zero

    if (nanglelist.gt.0) then
!$omp barrier
!$omp do schedule(static)
    do i=1,nanglelist
       ii = anglelist(i)%i+1
       jj = anglelist(i)%j+1
       kk = anglelist(i)%k+1
       ic = anglelist(i)%itype+1
       is = anglelist(i)%ishift1*3+1
       ks = anglelist(i)%ishift2*3+1

       sxi = scoordtab(is)
       syi = scoordtab(is+1)
       szi = scoordtab(is+2)
       sxk = scoordtab(ks)
       syk = scoordtab(ks+1)
       szk = scoordtab(ks+2)

       if (ic == 0) cycle

       dxij = xyzq(ii)%x + sxi - xyzq(jj)%x
       dyij = xyzq(ii)%y + syi - xyzq(jj)%y
       dzij = xyzq(ii)%z + szi - xyzq(jj)%z

       dxkj = xyzq(kk)%x + sxk - xyzq(jj)%x
       dykj = xyzq(kk)%y + syk - xyzq(jj)%y
       dzkj = xyzq(kk)%z + szk - xyzq(jj)%z

       rij = sqrt(dxij*dxij + dyij*dyij + dzij*dzij)
       if (rij == zero) cycle
       rkj = sqrt(dxkj*dxkj + dykj*dykj + dzkj*dzkj)
       if (rkj == zero) cycle

       rij_inv = one/rij
       rkj_inv = one/rkj
       dxijr = dxij*rij_inv
       dyijr = dyij*rij_inv
       dzijr = dzij*rij_inv
       dxkjr = dxkj*rkj_inv
       dykjr = dykj*rkj_inv
       dzkjr = dzkj*rkj_inv
       cst = dxijr*dxkjr + dyijr*dykjr + dzijr*dzkjr

#ifdef DOUBLE_PREC
       ctb_val = ctb(ic)
       ctc_val = ctc(ic)
#else
       ctb_val = real(ctb(ic),kind=chm_real4)
       ctc_val = real(ctc(ic),kind=chm_real4)
#endif

       ! The CHARMM angle potential has issues related to dividing by
       ! zero when you take the derivative w.r.t. cos(theta). The
       ! GROMACS cosine angle potential does not. So, the "flat angle"
       ! code isn't needed for the GROMACS cosine angle potential.
       ! ctb(ic).ge.0 --> CHARMM. .lt.0 --> GROMACS cosine.
       if((abs(cst) >= pt99) &
#ifdef GROMACS_STYLE
            .and. (ctb(ic).ge.0) &
#endif
            ) then
#ifdef CHARMM_STYLE
          if(abs(cst) > one) cst=sign(one,cst)
          at = acos(cst)
          da = at - ctb_val
          df = ctc_val*da
          !     
          depot = real(df*da,kind=chm_real)
          st2r = one/(one - cst*cst + rpreci)
          str = sqrt(st2r)
          if(ctb_val < pt001) then
             df = mintwo*ctc_val*(one + da*da*sixth)
          else if(pi_val-ctb_val < pt001) then
             df = two*ctc_val*(one + da*da*sixth)
          else
             df = mintwo*df*str
          endif
#endif
       else
          at = acos(cst)
#if defined(CHARMM_STYLE) && defined(GROMACS_STYLE)
          if(ctb(ic).ge.0) then
#endif
#ifdef CHARMM_STYLE
             da = at - ctb_val
             df = ctc_val*da
             depot = real(df*da,kind=chm_real)
             
             st2r = one/(one - cst*cst)
             str = sqrt(st2r)
             df = mintwo*df*str
#endif
#if defined(CHARMM_STYLE) && defined(GROMACS_STYLE)
          else
#endif
#ifdef GROMACS_STYLE
             da = cst-(ctb_val+one) ! cos(theta) - cos(theta_0)
             df = ctc_val*da
             depot = real(df*da/two,kind=chm_real)
#endif
#if defined(CHARMM_STYLE) && defined(GROMACS_STYLE)
          endif
#endif
       endif

#if KEY_BLOCK==1
       df = df*fscale
#endif

       epot = epot + depot

       dtxi = rij_inv*(dxkjr - cst*dxijr)
       dtxj = rkj_inv*(dxijr - cst*dxkjr)
       dtyi = rij_inv*(dykjr - cst*dyijr)
       dtyj = rkj_inv*(dyijr - cst*dykjr)
       dtzi = rij_inv*(dzkjr - cst*dzijr)
       dtzj = rkj_inv*(dzijr - cst*dzkjr)

       ! Store forces
       ii3 = 3*ii - 2
       jj3 = 3*jj - 2
       kk3 = 3*kk - 2

#ifdef DOUBLE_PREC
       dti = df*dtxi
       dtj = df*dtxj
#else
       dti = real(df*dtxi,kind=chm_real)
       dtj = real(df*dtxj,kind=chm_real)
#endif
       force(ii3) = force(ii3) + dti
       force(kk3) = force(kk3) + dtj
       force(jj3) = force(jj3) - dti - dtj
       sforce(is) = sforce(is) + dti
       sforce(ks) = sforce(ks) + dtj

#ifdef DOUBLE_PREC
       dti = df*dtyi
       dtj = df*dtyj
#else
       dti = real(df*dtyi,kind=chm_real)
       dtj = real(df*dtyj,kind=chm_real)
#endif
       force(ii3+1) = force(ii3+1) + dti
       force(kk3+1) = force(kk3+1) + dtj
       force(jj3+1) = force(jj3+1) - dti - dtj
       sforce(is+1) = sforce(is+1) + dti
       sforce(ks+1) = sforce(ks+1) + dtj

#ifdef DOUBLE_PREC
       dti = df*dtzi
       dtj = df*dtzj
#else
       dti = real(df*dtzi,kind=chm_real)
       dtj = real(df*dtzj,kind=chm_real)
#endif
       force(ii3+2) = force(ii3+2) + dti
       force(kk3+2) = force(kk3+2) + dtj
       force(jj3+2) = force(jj3+2) - dti - dtj
       sforce(is+2) = sforce(is+2) + dti
       sforce(ks+2) = sforce(ks+2) + dtj
    enddo
!$omp end do
    endif
       
    return
  end subroutine KERNEL_NAME
#endif

#ifdef DIHE_LOOP

#ifdef DOUBLE_PREC
#define SUB_KERNEL_NAME dihe_pot_pd
#else
#define SUB_KERNEL_NAME dihe_pot_ps
#endif

  ! *
  ! * Dihedrals loop
  ! *
  subroutine KERNEL_NAME(ndihelist, dihelist, cpd, cpc, cpsin, cpcos, xyzq, scoordtab, &
       fscale, epot, force, sforce, nlinear)
!    use domdec_bonded_types,only:dihelist_t
!    use domdec_local_types,only:xyzq_dp_t  !##dp
!    use domdec_local_types,only:xyzq_sp_t  !##sp
    implicit none
    ! Input / Output
    integer, intent(in) :: ndihelist
    type(dihelist_t), intent(in) :: dihelist(:)
    integer, intent(in) :: cpd(:)
    real(chm_real), intent(in) :: cpc(:), cpsin(:), cpcos(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:), fscale
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:), fscale
#endif
    real(chm_real), intent(out) :: epot
    real(chm_real), intent(inout) :: force(:), sforce(:)
    integer, intent(inout) :: nlinear
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real
    real(chm_real), parameter :: rxmin=0.005_chm_real, rxmin2=0.000025_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4
    real(chm_real4), parameter :: rxmin=0.005_chm_real4, rxmin2=0.000025_chm_real4
#endif
    ! Variables
    integer i
    integer ii, jj, kk, ll, ic, is, js, ls
    integer ii3, jj3, kk3, ll3
#ifdef DOUBLE_PREC
    real(chm_real) sxi, syi, szi
    real(chm_real) sxj, syj, szj
    real(chm_real) sxl, syl, szl
    real(chm_real) fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(chm_real) ax, ay, az, bx, by, bz
    real(chm_real) ra2, rb2, rg, rgr, ra2r, rb2r, rabr, ct, st
    real(chm_real) fg, hg, fga, hgb, gaa, gbb
    real(chm_real) e, df
#else
    real(chm_real4) sxi, syi, szi
    real(chm_real4) sxj, syj, szj
    real(chm_real4) sxl, syl, szl
    real(chm_real4) fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(chm_real4) ax, ay, az, bx, by, bz
    real(chm_real4) ra2, rb2, rg, rgr, ra2r, rb2r, rabr, ct, st
    real(chm_real4) fg, hg, fga, hgb, gaa, gbb
    real(chm_real4) e, df
#endif
    real(chm_real) dfx, dfy, dfz, dgx, dgy, dgz, dhx, dhy, dhz

    epot = zero

    if (ndihelist.gt.0) then
!$omp barrier
!$omp do schedule(static)
    do i=1,ndihelist
       ii = dihelist(i)%i+1
       jj = dihelist(i)%j+1
       kk = dihelist(i)%k+1
       ll = dihelist(i)%l+1
       ic = dihelist(i)%itype+1
       is = dihelist(i)%ishift1*3+1
       js = dihelist(i)%ishift2*3+1
       ls = dihelist(i)%ishift3*3+1

       sxi = scoordtab(is)
       syi = scoordtab(is+1)
       szi = scoordtab(is+2)
       sxj = scoordtab(js)
       syj = scoordtab(js+1)
       szj = scoordtab(js+2)
       sxl = scoordtab(ls)
       syl = scoordtab(ls+1)
       szl = scoordtab(ls+2)

       fx = (xyzq(ii)%x + sxi) - (xyzq(jj)%x + sxj)
       fy = (xyzq(ii)%y + syi) - (xyzq(jj)%y + syj)
       fz = (xyzq(ii)%z + szi) - (xyzq(jj)%z + szj)

       gx = xyzq(jj)%x + sxj - xyzq(kk)%x
       gy = xyzq(jj)%y + syj - xyzq(kk)%y
       gz = xyzq(jj)%z + szj - xyzq(kk)%z

       hx = xyzq(ll)%x + sxl - xyzq(kk)%x
       hy = xyzq(ll)%y + syl - xyzq(kk)%y
       hz = xyzq(ll)%z + szl - xyzq(kk)%z

       ! A=F^G, B=H^G.
       ax = fy*gz - fz*gy
       ay = fz*gx - fx*gz
       az = fx*gy - fy*gx
       bx = hy*gz - hz*gy
       by = hz*gx - hx*gz
       bz = hx*gy - hy*gx

       ra2 = ax*ax + ay*ay + az*az
       rb2 = bx*bx + by*by + bz*bz
       rg = sqrt(gx*gx + gy*gy + gz*gz)

       if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
          nlinear = nlinear + 1
       endif

       rgr = one / rg
       ra2r = one / ra2
       rb2r = one / rb2
       rabr = sqrt(ra2r*rb2r)
       ! ct=cos(phi)
       ct = (ax*bx + ay*by + az*bz)*rabr
       !
       ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
       ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
       st = rg*rabr*(ax*hx + ay*hy + az*hz)
       !
       !     Energy and derivative contributions.

       call SUB_KERNEL_NAME(cpd, cpc, cpcos, cpsin, ic, st, ct, e, df)

#if KEY_BLOCK==1
       df = df*fscale
#endif

#ifdef DOUBLE_PREC
       epot = epot + e
#else
       epot = epot + real(e,kind=chm_real)
#endif

       !
       !     Compute derivatives wrt catesian coordinates.
       !
       ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
       !  FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
       fg = fx*gx + fy*gy + fz*gz
       hg = hx*gx + hy*gy + hz*gz
       ra2r = df*ra2r
       rb2r = df*rb2r
       fga = fg*ra2r*rgr
       hgb = hg*rb2r*rgr
       gaa = ra2r*rg
       gbb = rb2r*rg
       ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
#ifdef DOUBLE_PREC
       dfx = -gaa*ax
       dfy = -gaa*ay
       dfz = -gaa*az
       dgx = fga*ax - hgb*bx
       dgy = fga*ay - hgb*by
       dgz = fga*az - hgb*bz
       dhx = gbb*bx
       dhy = gbb*by
       dhz = gbb*bz
#else
       dfx = real(-gaa*ax,kind=chm_real)
       dfy = real(-gaa*ay,kind=chm_real)
       dfz = real(-gaa*az,kind=chm_real)
       dgx = real(fga*ax - hgb*bx,kind=chm_real)
       dgy = real(fga*ay - hgb*by,kind=chm_real)
       dgz = real(fga*az - hgb*bz,kind=chm_real)
       dhx = real(gbb*bx,kind=chm_real)
       dhy = real(gbb*by,kind=chm_real)
       dhz = real(gbb*bz,kind=chm_real)
#endif

       ! Store forces
       ii3 = 3*ii - 2
       jj3 = 3*jj - 2
       kk3 = 3*kk - 2
       ll3 = 3*ll - 2

       force(ii3)   = force(ii3)   + dfx
       force(ii3+1) = force(ii3+1) + dfy
       force(ii3+2) = force(ii3+2) + dfz
       sforce(is)   = sforce(is)   + dfx
       sforce(is+1) = sforce(is+1) + dfy
       sforce(is+2) = sforce(is+2) + dfz

       force(jj3)   = force(jj3)   - dfx + dgx
       force(jj3+1) = force(jj3+1) - dfy + dgy
       force(jj3+2) = force(jj3+2) - dfz + dgz
       sforce(js)   = sforce(js)   - dfx + dgx
       sforce(js+1) = sforce(js+1) - dfy + dgy
       sforce(js+2) = sforce(js+2) - dfz + dgz

       force(kk3)   = force(kk3)   - dhx - dgx
       force(kk3+1) = force(kk3+1) - dhy - dgy
       force(kk3+2) = force(kk3+2) - dhz - dgz

       force(ll3)   = force(ll3)   + dhx
       force(ll3+1) = force(ll3+1) + dhy
       force(ll3+2) = force(ll3+2) + dhz
       sforce(ls)   = sforce(ls)   + dhx
       sforce(ls+1) = sforce(ls+1) + dhy
       sforce(ls+2) = sforce(ls+2) + dhz
    enddo
!$omp end do
    endif

    return
  end subroutine KERNEL_NAME

  ! *
  ! * Dihedral potential
  ! *
  subroutine SUB_KERNEL_NAME(cpd, cpc, cpcos, cpsin, ic_in, st, ct, e, df)
    implicit none
    ! Input / Output
    integer, intent(in) :: cpd(:)
    real(chm_real), intent(in) :: cpc(:), cpcos(:), cpsin(:)
    integer, intent(in) :: ic_in
#ifdef DOUBLE_PREC
    real(chm_real), intent(in) :: st, ct
    real(chm_real), intent(out) :: e, df
#else
    real(chm_real4), intent(in) :: st, ct
    real(chm_real4), intent(out) :: e, df
#endif
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4
#endif
    ! Variables
    integer ic, iper, nper
    logical lrep
#ifdef DOUBLE_PREC
    real(chm_real) e1, df1, ddf1, cpcos_val, cpsin_val, arg
#else
    real(chm_real4) e1, df1, ddf1, cpcos_val, cpsin_val, arg
#endif

    e = zero
    df = zero
    ic = ic_in
30  continue
    iper = cpd(ic)
    if (iper > 0) then
       lrep = .false.
    else
       lrep = .true.
       iper = -iper
    endif
    !
    e1 = one
    df1 = zero
    ddf1 = zero
    !Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
    do nper=1,iper
       ddf1 = e1*ct - df1*st
       df1 = e1*st + df1*ct
       e1 = ddf1
    enddo
#ifdef DOUBLE_PREC
    cpcos_val = cpcos(ic)
    cpsin_val = cpsin(ic)
#else
    cpcos_val = real(cpcos(ic),kind=chm_real4)
    cpsin_val = real(cpsin(ic),kind=chm_real4)
#endif
    e1 = e1*cpcos_val + df1*cpsin_val
    df1 = df1*cpcos_val - ddf1*cpsin_val
    df1 = -iper*df1
    e1 = one + e1
    !brb...03-Jul-2004 Zero-period dihedral bugfix
    if(iper == 0) e1 = one
    !
    arg = cpc(ic)
    e = e + arg*e1
    df = df + arg*df1
    !
    if(lrep) then
       ic = ic+1
       goto 30
    endif

    return
  end subroutine SUB_KERNEL_NAME
#undef SUB_KERNEL_NAME

#endif

#ifdef IMDIHE_LOOP

#ifdef DOUBLE_PREC
#define SUB_KERNEL_NAME imdihe_pot_pd
#else
#define SUB_KERNEL_NAME imdihe_pot_ps
#endif

  ! *
  ! * Improper dihedrals loop
  ! *
  subroutine KERNEL_NAME(ndihelist, dihelist, cid, cic, cisin, cicos, xyzq, &
       scoordtab, fscale, epot, force, sforce, nlinear, nbent)
!    use domdec_bonded_types,only:dihelist_t
!    use domdec_local_types,only:xyzq_dp_t  !##dp
!    use domdec_local_types,only:xyzq_sp_t  !##sp
    implicit none
    ! Input / Output
    integer, intent(in) :: ndihelist
    type(dihelist_t), intent(in) :: dihelist(:)
    integer, intent(in) :: cid(:)
    real(chm_real), intent(in) :: cic(:), cisin(:), cicos(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:), fscale
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:), fscale
#endif
    real(chm_real), intent(out) :: epot
    real(chm_real), intent(inout) :: force(:), sforce(:)
    integer, intent(inout) :: nlinear, nbent
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real
    real(chm_real), parameter :: rxmin=0.005_chm_real, rxmin2=0.000025_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4
    real(chm_real4), parameter :: rxmin=0.005_chm_real4, rxmin2=0.000025_chm_real4
#endif
    ! Variables
    integer i
    integer ii, jj, kk, ll, ic, is, js, ls
    integer ii3, jj3, kk3, ll3
#ifdef DOUBLE_PREC
    real(chm_real) sxi, syi, szi
    real(chm_real) sxj, syj, szj
    real(chm_real) sxl, syl, szl
    real(chm_real) fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(chm_real) ax, ay, az, bx, by, bz
    real(chm_real) ra2, rb2, rg, rgr, ra2r, rb2r, rabr, ct, st
    real(chm_real) fg, hg, fga, hgb, gaa, gbb
    real(chm_real) e, df
#else
    real(chm_real4) sxi, syi, szi
    real(chm_real4) sxj, syj, szj
    real(chm_real4) sxl, syl, szl
    real(chm_real4) fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(chm_real4) ax, ay, az, bx, by, bz
    real(chm_real4) ra2, rb2, rg, rgr, ra2r, rb2r, rabr, ct, st
    real(chm_real4) fg, hg, fga, hgb, gaa, gbb
    real(chm_real4) e, df
#endif
    real(chm_real) dfx, dfy, dfz, dgx, dgy, dgz, dhx, dhy, dhz

    epot = zero

    if (ndihelist.gt.0) then
!$omp barrier
!$omp do schedule(static)
    do i=1,ndihelist
       ii = dihelist(i)%i+1
       jj = dihelist(i)%j+1
       kk = dihelist(i)%k+1
       ll = dihelist(i)%l+1
       ic = dihelist(i)%itype+1
       is = dihelist(i)%ishift1*3+1
       js = dihelist(i)%ishift2*3+1
       ls = dihelist(i)%ishift3*3+1
       
       sxi = scoordtab(is)
       syi = scoordtab(is+1)
       szi = scoordtab(is+2)
       sxj = scoordtab(js)
       syj = scoordtab(js+1)
       szj = scoordtab(js+2)
       sxl = scoordtab(ls)
       syl = scoordtab(ls+1)
       szl = scoordtab(ls+2)

       if (ic.eq.0) cycle

#if KEY_OPLS==0
       if (cid(ic).ne.0) then
          call wrndie(-3,'<eiphifs>', &
               'bad periodicity value for improper dihedral angles.')
          cycle
       endif
#endif

       ! f=ri-rj, g=rj-rk, h-rl-rk.
       fx = (xyzq(ii)%x + sxi) - (xyzq(jj)%x + sxj)
       fy = (xyzq(ii)%y + syi) - (xyzq(jj)%y + syj)
       fz = (xyzq(ii)%z + szi) - (xyzq(jj)%z + szj)

       gx = xyzq(jj)%x + sxj - xyzq(kk)%x
       gy = xyzq(jj)%y + syj - xyzq(kk)%y
       gz = xyzq(jj)%z + szj - xyzq(kk)%z

       hx = xyzq(ll)%x + sxl - xyzq(kk)%x
       hy = xyzq(ll)%y + syl - xyzq(kk)%y
       hz = xyzq(ll)%z + szl - xyzq(kk)%z

       ! a=f^g, b=h^g.
       ax = fy*gz - fz*gy
       ay = fz*gx - fx*gz
       az = fx*gy - fy*gx
       bx = hy*gz - hz*gy
       by = hz*gx - hx*gz
       bz = hx*gy - hy*gx

       ra2 = ax*ax + ay*ay + az*az
       rb2 = bx*bx + by*by + bz*bz
       rg = sqrt(gx*gx + gy*gy + gz*gz)

       if((ra2 <= rxmin2) .or. (rb2 <= rxmin2) .or. (rg <= rxmin)) then
          nlinear = nlinear + 1
       endif

       rgr = one/rg
       ra2r = one/ra2
       rb2r = one/rb2
       rabr = sqrt(ra2r*rb2r)
       ! ct=cos(phi)
       ct = (ax*bx + ay*by + az*bz)*rabr
       !
       ! st=sin(phi), note that sin(phi).g/|g|=b^a/(|a|.|b|)
       ! which can be simplify to sin(phi)=|g|h.a/(|a|.|b|)
       st = rg*rabr*(ax*hx + ay*hy + az*hz)

       call SUB_KERNEL_NAME(cid, cic, cicos, cisin, ic, st, ct, e, df, nbent)

#if KEY_BLOCK==1
       df = df*fscale
#endif

#ifdef DOUBLE_PREC
       epot = epot + e
#else
       epot = epot + real(e,kind=chm_real)
#endif

       !
       !     compute derivatives wrt catesian coordinates.
       !
       ! gaa=de/dphi.|g|/a^2, gbb=de/dphi.|g|/b^2, fg=f.g, hg=h.g
       !  fga=de/dphi*f.g/(|g|a^2), hgb=de/dphi*h.g/(|g|b^2)
       fg = fx*gx + fy*gy + fz*gz
       hg = hx*gx + hy*gy + hz*gz
       ra2r = df*ra2r
       rb2r = df*rb2r
       fga = fg*ra2r*rgr
       hgb = hg*rb2r*rgr
       gaa = ra2r*rg
       gbb = rb2r*rg

       ! dfi=de/dfi, dgi=de/dgi, dhi=de/dhi.
#ifdef DOUBLE_PREC
       dfx = -gaa*ax
       dfy = -gaa*ay
       dfz = -gaa*az
       dgx = fga*ax - hgb*bx
       dgy = fga*ay - hgb*by
       dgz = fga*az - hgb*bz
       dhx = gbb*bx
       dhy = gbb*by
       dhz = gbb*bz
#else
       dfx = real(-gaa*ax,kind=chm_real)
       dfy = real(-gaa*ay,kind=chm_real)
       dfz = real(-gaa*az,kind=chm_real)
       dgx = real(fga*ax - hgb*bx,kind=chm_real)
       dgy = real(fga*ay - hgb*by,kind=chm_real)
       dgz = real(fga*az - hgb*bz,kind=chm_real)
       dhx = real(gbb*bx,kind=chm_real)
       dhy = real(gbb*by,kind=chm_real)
       dhz = real(gbb*bz,kind=chm_real)
#endif

       ! Store forces
       ii3 = 3*ii - 2
       jj3 = 3*jj - 2
       kk3 = 3*kk - 2
       ll3 = 3*ll - 2

       force(ii3)   = force(ii3)   + dfx
       force(ii3+1) = force(ii3+1) + dfy
       force(ii3+2) = force(ii3+2) + dfz
       sforce(is)   = sforce(is)   + dfx
       sforce(is+1) = sforce(is+1) + dfy
       sforce(is+2) = sforce(is+2) + dfz

       force(jj3)   = force(jj3)   - dfx + dgx
       force(jj3+1) = force(jj3+1) - dfy + dgy
       force(jj3+2) = force(jj3+2) - dfz + dgz
       sforce(js)   = sforce(js)   - dfx + dgx
       sforce(js+1) = sforce(js+1) - dfy + dgy
       sforce(js+2) = sforce(js+2) - dfz + dgz

       force(kk3)   = force(kk3)   - dhx - dgx
       force(kk3+1) = force(kk3+1) - dhy - dgy
       force(kk3+2) = force(kk3+2) - dhz - dgz

       force(ll3)   = force(ll3)   + dhx
       force(ll3+1) = force(ll3+1) + dhy
       force(ll3+2) = force(ll3+2) + dhz
       sforce(ls)   = sforce(ls)   + dhx
       sforce(ls+1) = sforce(ls+1) + dhy
       sforce(ls+2) = sforce(ls+2) + dhz
    enddo
!$omp end do
    endif

    return
  end subroutine KERNEL_NAME

  ! *
  ! * Improper dihedral potential
  ! *
  subroutine SUB_KERNEL_NAME(cid, cic, cicos, cisin, ic_in, st, ct, e, df, nbent)
    implicit none
    ! Input / Output
    integer, intent(in) :: cid(:)
    real(chm_real), intent(in) :: cic(:), cicos(:), cisin(:)
    integer, intent(in) :: ic_in
#ifdef DOUBLE_PREC
    real(chm_real), intent(in) :: st, ct
    real(chm_real), intent(out) :: e, df
#else
    real(chm_real4), intent(in) :: st, ct
    real(chm_real4), intent(out) :: e, df
#endif
    integer, intent(inout) :: nbent
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real, ptone  = 0.1_chm_real
    real(chm_real), parameter :: minone = -1.0_chm_real, two = 2.0_chm_real
#else
    real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4, ptone  = 0.1_chm_real4
    real(chm_real4), parameter :: minone = -1.0_chm_real4, two = 2.0_chm_real4
#endif
    ! Variables
    integer ic, iper, nper
    logical lrep
#ifdef DOUBLE_PREC
    real(chm_real) e1, df1, ddf1, arg
    real(chm_real) ca, sa, ap
    real(chm_real) cicos_val, cisin_val
#else
    real(chm_real4) e1, df1, ddf1, arg
    real(chm_real4) ca, sa, ap
    real(chm_real4) cicos_val, cisin_val
#endif

    ic = ic_in

#if KEY_OPLS==1
    if (cid(ic) /= 0) then

       e = zero
       df = zero
35     continue
       iper = cid(ic)
       if (iper >= 0) then
          lrep = .false.
       else
          lrep = .true.
          iper = -iper
       endif
       !
       e1 = one
       df1 = zero
       !calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
       do nper=1,iper
          ddf1 = e1*ct - df1*st
          df1 = e1*st + df1*ct
          e1 = ddf1
       enddo
#ifdef DOUBLE_PREC
       cicos_val = cicos(ic)
       cisin_val = cisin(ic)
#else
       cicos_val = real(cicos(ic),kind=chm_real4)
       cisin_val = real(cisin(ic),kind=chm_real4)
#endif
       e1 = e1*cicos_val + df1*cisin_val
       df1 = df1*cicos_val - ddf1*cisin_val
       df1 = -iper*df1
       e1 = one + e1
       !
#ifdef DOUBLE_PREC
       arg = cic(ic)
#else
       arg = real(cic(ic),kind=chm_real4)
#endif
       e = e + arg*e1
       df = df + arg*df1
       !
       if(lrep) then
          ic = ic + 1
          goto 35
       endif
       !
       
       ! use harmonic potential
       !
    else
       !
#endif
       !
       !calcul of cos(phi-phi0),sin(phi-phi0) and (phi-phi0).
#ifdef DOUBLE_PREC
       cicos_val = cicos(ic)
       cisin_val = cisin(ic)
#else
       cicos_val = real(cicos(ic),kind=chm_real4)
       cisin_val = real(cisin(ic),kind=chm_real4)
#endif
       ca = ct*cicos_val + st*cisin_val
       sa = st*cicos_val - ct*cisin_val
       if (ca > ptone ) then
          ap = asin(sa)
       else
          ap = sign(acos(max(ca,minone)),sa)
          ! warning is now triggered at deltaphi=84.26...deg (used to be 90).
          nbent = nbent + 1
       endif
       !
#ifdef DOUBLE_PREC
       df = cic(ic)*ap
#else
       df = real(cic(ic),kind=chm_real4)*ap
#endif
       e = df*ap
       df = two*df
    endif

    return
  end subroutine SUB_KERNEL_NAME

#undef SUB_KERNEL_NAME
#endif
