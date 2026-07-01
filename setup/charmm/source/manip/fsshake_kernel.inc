#define PASTE(a) a
#define ADD_SUFFIX(name) PASTE(name)PASTE(_)FSSHAKE_SUFFIX

! this code relies on the C Preprocessor
! # define FSSHAKE_SUFFIX to be either
! D0 and undefine FSSHAKE_DD or
! D1 and define FSSHAKE_DD

  ! *
  ! * Kernel for settle_water
  ! *
  subroutine ADD_SUFFIX(settle_water_kernel)(imol_start, imol_end, &
#ifdef FSSHAKE_DD
       shakewater_ind, &
#else /**/
       nstwat, shkapr, &
#endif 
       x0, y0, z0, x1, y1, z1, &
       mO_div_mH2O, mH_div_mH2O, ra, rc, rb, ra_inv, rc2)
    use number,only:one
    implicit none
    ! Input / Output
    integer, intent(in) :: imol_start, imol_end
#ifdef FSSHAKE_DD
    integer, intent(in) :: shakewater_ind(:)
#else /**/
    integer, intent(in) :: nstwat, shkapr(:,:)
#endif 
    real(chm_real), intent(in) :: x0(*), y0(*), z0(*)
    real(chm_real), intent(inout) :: x1(*), y1(*), z1(*)
    real(chm_real), intent(in) :: mO_div_mH2O, mH_div_mH2O, ra, rc, rb, ra_inv, rc2
    ! Variables
    real(chm_real) xcm, ycm, zcm
    real(chm_real) xa1, ya1, za1, xb1, yb1, zb1, xc1, yc1, zc1
    real(chm_real) xb0, yb0, zb0, xc0, yc0, zc0
    real(chm_real) xaksxd, yaksxd, zaksxd, xaksyd, yaksyd, zaksyd, xakszd, yakszd, zakszd
    real(chm_real) axlng_inv, aylng_inv, azlng_inv
    real(chm_real) trans11, trans21, trans31, trans12, trans22, trans32, trans13, trans23, trans33
    real(chm_real) xb0p, yb0p, xc0p, yc0p, za1p, xb1p, yb1p, zb1p, xc1p, yc1p, zc1p
    real(chm_real) sinphi, cosphi, sinpsi, cospsi
    real(chm_real) ya2p, xb2p, yb2p, yc2p
    real(chm_real) alpha, beta, gamma, alpha_beta, sintheta, costheta
    real(chm_real) xa3p, ya3p, za3p, xb3p, yb3p, zb3p, xc3p, yc3p, zc3p
    integer imol, i, j, k
#ifndef FSSHAKE_DD
    integer ip
#endif 

#ifdef FSSHAKE_DD
!dec$ simd assert
#endif 
    do imol=imol_start,imol_end
#ifdef FSSHAKE_DD
       i = shakewater_ind(imol*3-2) ! O
       j = shakewater_ind(imol*3-1) ! H
       k = shakewater_ind(imol*3)   ! H
#else /**/
       ip = 3*(imol-1)+nstwat
       i = shkapr(1,ip)        ! O
       j = shkapr(2,ip)        ! H
       k = shkapr(2,ip+1)      ! H
#endif 

       !
       ! Convert to primed coordinates
       !

       ! Calculate center of mass for (x1, y1, z1)
       xcm = x1(i)*mO_div_mH2O + (x1(j) + x1(k))*mH_div_mH2O
       ycm = y1(i)*mO_div_mH2O + (y1(j) + y1(k))*mH_div_mH2O
       zcm = z1(i)*mO_div_mH2O + (z1(j) + z1(k))*mH_div_mH2O

       ! Calculate (x1, y1, z1) with center of mass at origin
       xa1 = x1(i) - xcm
       ya1 = y1(i) - ycm
       za1 = z1(i) - zcm
       xb1 = x1(j) - xcm
       yb1 = y1(j) - ycm
       zb1 = z1(j) - zcm
       xc1 = x1(k) - xcm
       yc1 = y1(k) - ycm
       zc1 = z1(k) - zcm

       xb0 = x0(j) - x0(i)
       yb0 = y0(j) - y0(i)
       zb0 = z0(j) - z0(i)
       xc0 = x0(k) - x0(i)
       yc0 = y0(k) - y0(i)
       zc0 = z0(k) - z0(i)

       ! (xb0, yb0, zb0), (xc0, yc0, zc0), and (xa1, ya1, za1) define the primed coordinate set:
       ! * X'Y' plane is parallel to the plane defined by (xb0, yb0, zb0) and (xc0, yc0, zc0)
       ! * Y'Z' plane contains (xa1, ya1, za1)

       xakszd = yb0 * zc0 - zb0 * yc0
       yakszd = zb0 * xc0 - xb0 * zc0
       zakszd = xb0 * yc0 - yb0 * xc0
       xaksxd = ya1 * zakszd - za1 * yakszd
       yaksxd = za1 * xakszd - xa1 * zakszd
       zaksxd = xa1 * yakszd - ya1 * xakszd
       xaksyd = yakszd * zaksxd - zakszd * yaksxd
       yaksyd = zakszd * xaksxd - xakszd * zaksxd
       zaksyd = xakszd * yaksxd - yakszd * xaksxd

       axlng_inv = one / sqrt(xaksxd * xaksxd + yaksxd * yaksxd + zaksxd * zaksxd)
       aylng_inv = one / sqrt(xaksyd * xaksyd + yaksyd * yaksyd + zaksyd * zaksyd)
       azlng_inv = one / sqrt(xakszd * xakszd + yakszd * yakszd + zakszd * zakszd)

       trans11 = xaksxd * axlng_inv
       trans21 = yaksxd * axlng_inv
       trans31 = zaksxd * axlng_inv
       trans12 = xaksyd * aylng_inv
       trans22 = yaksyd * aylng_inv
       trans32 = zaksyd * aylng_inv
       trans13 = xakszd * azlng_inv
       trans23 = yakszd * azlng_inv
       trans33 = zakszd * azlng_inv

       ! Calculate necessary primed coordinates
       xb0p = trans11 * xb0 + trans21 * yb0 + trans31 * zb0
       yb0p = trans12 * xb0 + trans22 * yb0 + trans32 * zb0
       xc0p = trans11 * xc0 + trans21 * yc0 + trans31 * zc0
       yc0p = trans12 * xc0 + trans22 * yc0 + trans32 * zc0
       za1p = trans13 * xa1 + trans23 * ya1 + trans33 * za1
       xb1p = trans11 * xb1 + trans21 * yb1 + trans31 * zb1
       yb1p = trans12 * xb1 + trans22 * yb1 + trans32 * zb1
       zb1p = trans13 * xb1 + trans23 * yb1 + trans33 * zb1
       xc1p = trans11 * xc1 + trans21 * yc1 + trans31 * zc1
       yc1p = trans12 * xc1 + trans22 * yc1 + trans32 * zc1
       zc1p = trans13 * xc1 + trans23 * yc1 + trans33 * zc1

       !
       ! Calculate rotation angles (phi, psi, theta)
       !

       sinphi = za1p * ra_inv
       cosphi = sqrt(one - sinphi*sinphi)
       sinpsi = (zb1p - zc1p) / (rc2 * cosphi)
       cospsi = sqrt(one - sinpsi*sinpsi)

       ya2p =   ra * cosphi
       xb2p = - rc * cospsi
       yb2p = - rb * cosphi - rc *sinpsi * sinphi
       yc2p = - rb * cosphi + rc *sinpsi * sinphi

       !          xb2p =  -half * sqrt(hhhh - (yb2p-yc2p) * (yb2p-yc2p) - (zb1p-zc1p) * (zb1p-zc1p))

       alpha = (xb2p * (xb0p-xc0p) + yb0p * yb2p + yc0p * yc2p)
       beta  = (xb2p * (yc0p-yb0p) + xb0p * yb2p + xc0p * yc2p)
       gamma = xb0p * yb1p - xb1p * yb0p + xc0p * yc1p - xc1p * yc0p

       alpha_beta = alpha * alpha + beta * beta
       sintheta = (alpha*gamma - beta * sqrt(alpha_beta - gamma*gamma)) / alpha_beta

       costheta = sqrt(one - sintheta*sintheta)
       xa3p = -ya2p * sintheta
       ya3p =  ya2p * costheta
       za3p =  za1p
       xb3p =  xb2p * costheta - yb2p * sintheta
       yb3p =  xb2p * sintheta + yb2p * costheta
       zb3p =  zb1p
       xc3p = -xb2p * costheta - yc2p * sintheta
       yc3p = -xb2p * sintheta + yc2p * costheta
       zc3p =  zc1p

       x1(i) = xcm + trans11 * xa3p + trans12 * ya3p + trans13 * za3p
       y1(i) = ycm + trans21 * xa3p + trans22 * ya3p + trans23 * za3p
       z1(i) = zcm + trans31 * xa3p + trans32 * ya3p + trans33 * za3p
       x1(j) = xcm + trans11 * xb3p + trans12 * yb3p + trans13 * zb3p
       y1(j) = ycm + trans21 * xb3p + trans22 * yb3p + trans23 * zb3p
       z1(j) = zcm + trans31 * xb3p + trans32 * yb3p + trans33 * zb3p
       x1(k) = xcm + trans11 * xc3p + trans12 * yc3p + trans13 * zc3p
       y1(k) = ycm + trans21 * xc3p + trans22 * yc3p + trans23 * zc3p
       z1(k) = zcm + trans31 * xc3p + trans32 * yc3p + trans33 * zc3p

    enddo

    return
  end subroutine ADD_SUFFIX(settle_water_kernel)

  ! *
  ! * Calculates fast shake for a pair where one of the atoms is hydrogen
  ! *
  subroutine ADD_SUFFIX(fsshakph_kernel2)(istart, iend, &
#ifdef FSSHAKE_DD
       shakepair_ind, shakepair_constr, shakepair_mass, &
#else /**/
       shake_list, shkapr, constr, hmassi, hmassj, &
#endif 
       xref, yref, zref, x, y, z)
    use number,only:two
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
#ifdef FSSHAKE_DD
    integer, intent(in) :: shakepair_ind(:)
    real(chm_real), intent(in) :: shakepair_constr(:), shakepair_mass(:)
#else /**/
    integer, intent(in) :: shake_list(:), shkapr(:,:)
    real(chm_real), intent(in) :: constr(:), hmassi(*), hmassj(*)
#endif 
    real(chm_real), intent(in) :: xref(*), yref(*), zref(*)
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    ! Variables
    real(chm_real) xpij, ypij, zpij
    real(chm_real) rijsq, diff
    real(chm_real) xrij, yrij, zrij
    real(chm_real) rrijsq, rijrijp, lambda
    real(chm_real) massi, massj
    integer ii, iconst, i, j

    do ii = istart,iend
#ifdef FSSHAKE_DD
       i = shakepair_ind(ii*2-1)
       j = shakepair_ind(ii*2)
#else /**/
       iconst = shake_list(ii)
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
#endif 
       xpij = x(i) - x(j)
       ypij = y(i) - y(j)
       zpij = z(i) - z(j)
       rijsq = xpij*xpij + ypij*ypij + zpij*zpij
#ifdef FSSHAKE_DD
       diff = shakepair_constr(ii) - rijsq
#else /**/
       diff = constr(iconst) - rijsq
#endif 
       xrij = xref(i) - xref(j)
       yrij = yref(i) - yref(j)
       zrij = zref(i) - zref(j)
       rrijsq= xrij*xrij + yrij*yrij + zrij*zrij
       rijrijp   = xrij*xpij  + yrij*ypij  + zrij*zpij
       lambda=two*(-rijrijp + &
            sqrt(rijrijp*rijrijp+rrijsq*diff))/(rrijsq)

#ifdef FSSHAKE_DD
       massi = shakepair_mass(ii*2-1)
       massj = shakepair_mass(ii*2)
#else /**/
       massi = hmassi(iconst)
       massj = hmassj(iconst)
#endif 

       x(i) = x(i) + massi*lambda*xrij
       y(i) = y(i) + massi*lambda*yrij
       z(i) = z(i) + massi*lambda*zrij
       x(j) = x(j) - massj*lambda*xrij
       y(j) = y(j) - massj*lambda*yrij
       z(j) = z(j) - massj*lambda*zrij
    end do

    return
  end subroutine ADD_SUFFIX(fsshakph_kernel2)

  ! *
  ! * Calculates fast shake for a triplet of atoms where two of the atoms are hydrogens
  ! *
  subroutine ADD_SUFFIX(fsshakph_kernel3)(istart, iend, &
#ifdef FSSHAKE_DD
       shaketrip_ind, shaketrip_constr, shaketrip_mass, &
#else /**/
       shake_list, shkapr, constr, ammi, i2start, &
#endif 
       a12tol, a13tol, &
       xref, yref, zref, x, y, z, max_niter, niter)
    use number,only:zero, half, two
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
#ifdef FSSHAKE_DD
    integer, intent(in) :: shaketrip_ind(:)
    real(chm_real), intent(in) :: shaketrip_constr(:), shaketrip_mass(:)
#else /**/
    integer, intent(in) :: shake_list(:), shkapr(:,:)
    real(chm_real), intent(in) :: constr(:), ammi(:)
    integer, intent(in) :: i2start
#endif 
    real(chm_real), intent(in) :: a12tol, a13tol
    real(chm_real), intent(in) :: xref(*), yref(*), zref(*)
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    integer, intent(in) :: max_niter
    integer, intent(out) :: niter
    ! Variables
    real(chm_real) xpij, ypij, zpij, xpik, ypik, zpik
    real(chm_real) xrij, yrij, zrij, xrik, yrik, zrik
    real(chm_real) mmi, mmj, mmk, mij, mik
    real(chm_real) rijsq, riksq, dij, dik
    real(chm_real) rijrijp, rijrikp
    real(chm_real) rikrijp, rikrikp
    real(chm_real) a12, a13
    real(chm_real) acorr1, acorr2, acorr3, acorr4, acorr5, acorr6
    real(chm_real) rrijsq, rriksq, rijrik
    real(chm_real) a12corr, a13corr
    real(chm_real) dinv
    real(chm_real) a120, a130
    integer ii, iconst, i, j, k, i2
    integer aniter

#ifndef FSSHAKE_DD
    i2 = i2start
#endif 

    niter = 0
    do ii = istart,iend
       a120 = zero
       a130 = zero
#ifdef FSSHAKE_DD
       i = shaketrip_ind(ii*3-2)
       j = shaketrip_ind(ii*3-1)
       k = shaketrip_ind(ii*3)
#else /**/
       iconst = shake_list(ii)
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       k = shkapr(2,iconst+1)
#endif 
       xrij = xref(i) - xref(j)
       yrij = yref(i) - yref(j)
       zrij = zref(i) - zref(j)
       xrik = xref(i) - xref(k)
       yrik = yref(i) - yref(k)
       zrik = zref(i) - zref(k)
#ifndef FSSHAKE_DD
       i2 = i2 + 5
#endif 

       rrijsq = xrij*xrij + yrij*yrij + zrij*zrij
       rriksq = xrik*xrik + yrik*yrik + zrik*zrik
       rijrik = xrij*xrik + yrij*yrik + zrij*zrik

#ifdef FSSHAKE_DD
       mmi = shaketrip_mass(ii*5-4)
       mmj = shaketrip_mass(ii*5-3)
       mmk = shaketrip_mass(ii*5-2)
       mij = shaketrip_mass(ii*5-1)
       mik = shaketrip_mass(ii*5)
#else /**/
       mmi = ammi(i2)
       mmj = ammi(i2+1)
       mmk = ammi(i2+2)
       mij = ammi(i2+3)
       mik = ammi(i2+4)
#endif 

       acorr1 = mij*mij*rrijsq
       acorr2 = mij*mmi*two*rijrik
       acorr3 = mmi*mmi*rriksq
       acorr4 = mmi*mmi*rrijsq
       acorr5 = mik*mmi*two*rijrik
       acorr6 = mik*mik*rriksq

       xpij = x(i)-x(j)
       ypij = y(i)-y(j)
       zpij = z(i)-z(j)
       xpik = x(i)-x(k)
       ypik = y(i)-y(k)
       zpik = z(i)-z(k)
       rijsq = xpij*xpij + ypij*ypij + zpij*zpij
       riksq = xpik*xpik + ypik*ypik + zpik*zpik
#ifdef FSSHAKE_DD
       dij = shaketrip_constr(ii*2-1) - rijsq
       dik = shaketrip_constr(ii*2)   - riksq
#else /**/
       dij = constr(iconst)   - rijsq
       dik = constr(iconst+1) - riksq
#endif 
       rijrijp = xrij*xpij + yrij*ypij + zrij*zpij
       rijrikp = xrij*xpik + yrij*ypik + zrij*zpik
       rikrijp = xpij*xrik + ypij*yrik + zpij*zrik
       rikrikp = xrik*xpik + yrik*ypik + zrik*zpik
       dinv=half/(rijrijp*rikrikp*mij*mik &
            - rijrikp*rikrijp*mmi*mmi)

       a12=dinv*( rikrikp*mik*(dij) &
            - rikrijp*mmi*(dik))
       a13=dinv*(-mmi*rijrikp*(dij) &
            + rijrijp*mij*(dik))
       aniter = 0

       do while((abs(a120-a12) > a12tol .or. &
            abs(a130-a13) > a13tol) .and. aniter < max_niter)
          aniter = aniter + 1
          a120 = a12
          a130 = a13
          a12corr=acorr1*a12*a12+acorr2*a12*a13+acorr3*a13*a13
          a13corr=acorr4*a12*a12+acorr5*a12*a13+acorr6*a13*a13
          a12=dinv*( rikrikp*mik*(dij-a12corr) &
               - rikrijp*mmi*(dik-a13corr))
          a13=dinv*(-mmi*rijrikp*(dij-a12corr) &
               + rijrijp*mij*(dik-a13corr))
       enddo
       x(i)  = x(i) + mmi*(xrij*a12+xrik*a13)
       x(j)  = x(j) - mmj*(xrij*a12)
       x(k)  = x(k) - mmk*(xrik*a13)
       y(i)  = y(i) + mmi*(yrij*a12+yrik*a13)
       y(j)  = y(j) - mmj*(yrij*a12)
       y(k)  = y(k) - mmk*(yrik*a13)
       z(i)  = z(i) + mmi*(zrij*a12+zrik*a13)
       z(j)  = z(j) - mmj*(zrij*a12)
       z(k)  = z(k) - mmk*(zrik*a13)

       niter = max(niter,aniter)
    end do

    return
  end subroutine ADD_SUFFIX(fsshakph_kernel3)

  ! *
  ! * Calculates fast shake for a group of 4 atoms where three of the atoms are hydrogens
  ! *
  subroutine ADD_SUFFIX(fsshakph_kernel4)(istart, iend, &
#ifdef FSSHAKE_DD
       shakequad_ind, shakequad_constr, shakequad_mass, &
#else /**/
       shake_list, shkapr, constr, ammi, i2start, &
#endif 
       a12tol, a13tol, a14tol, &
       xref, yref, zref, x, y, z, d2tol2, max_niter, niter)
    use number,only:zero, half, two
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
#ifdef FSSHAKE_DD
    integer, intent(in) :: shakequad_ind(:)
    real(chm_real), intent(in) :: shakequad_constr(:), shakequad_mass(:)
#else /**/
    integer, intent(in) :: shake_list(:), shkapr(:,:)
    real(chm_real), intent(in) :: constr(:), ammi(:)
    integer, intent(in) :: i2start
#endif 
    real(chm_real), intent(in) :: a12tol, a13tol, a14tol
    real(chm_real), intent(in) :: xref(*), yref(*), zref(*)
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    real(chm_real), intent(in) :: d2tol2(*)
    integer, intent(in) :: max_niter
    integer, intent(out) :: niter
    ! Variables
    real(chm_real) a120, a130, a140
    real(chm_real) xrij, yrij, zrij, xrik, yrik, zrik, xril, yril, zril
    real(chm_real) rrijsq, rriksq, rrilsq, rijrik, rijril, rikril
    real(chm_real) mmi, mmj, mmk, mml, mij, mik, mil
    real(chm_real) acorr1, acorr2, acorr3, acorr4, acorr5
    real(chm_real) acorr6, acorr7, acorr8, acorr9, acorr10
    real(chm_real) acorr11, acorr12, acorr13, acorr14, acorr15
    real(chm_real) xpij, ypij, zpij, xpik, ypik, zpik, xpil, ypil, zpil
    real(chm_real) rijsq, riksq, rilsq
    real(chm_real) dij, dik, dil
    real(chm_real) rijrijp, rijrikp, rijrilp
    real(chm_real) rikrijp, rikrikp, rikrilp
    real(chm_real) rilrijp, rilrikp, rilrilp
    real(chm_real) dinv, d1, d2, d3, d4, d5, d6, d7, d8, d9
    real(chm_real) a12, a13, a14
    real(chm_real) a12corr, a13corr, a14corr
    real(chm_real) constr1, constr2, constr3
    logical donep
    integer ii, iconst, i, j, k, l, i2, aniter

#ifndef FSSHAKE_DD
    i2 = i2start
#endif 

    niter = 0
    do ii =  istart, iend
       a120 = zero
       a130 = zero
       a140 = zero
       donep = .false.
#ifdef FSSHAKE_DD
       i = shakequad_ind(ii*4-3)
       j = shakequad_ind(ii*4-2)
       k = shakequad_ind(ii*4-1)
       l = shakequad_ind(ii*4)
#else /**/
       iconst = shake_list(ii)
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       k = shkapr(2,iconst+1)
       l = shkapr(2,iconst+2)
#endif 
       xrij = xref(i) - xref(j)
       yrij = yref(i) - yref(j)
       zrij = zref(i) - zref(j)
       xrik = xref(i) - xref(k)
       yrik = yref(i) - yref(k)
       zrik = zref(i) - zref(k)       
       xril = xref(i) - xref(l)
       yril = yref(i) - yref(l)
       zril = zref(i) - zref(l)
#ifndef FSSHAKE_DD
       i2 = i2 + 7
#endif 
       rrijsq = xrij*xrij + yrij*yrij + zrij*zrij
       rriksq = xrik*xrik + yrik*yrik + zrik*zrik
       rrilsq = xril*xril + yril*yril + zril*zril
       rijrik = xrij*xrik + yrij*yrik + zrij*zrik
       rijril = xrij*xril + yrij*yril + zrij*zril
       rikril = xrik*xril + yrik*yril + zrik*zril

#ifdef FSSHAKE_DD
       mmi = shakequad_mass(ii*7-6)
       mmj = shakequad_mass(ii*7-5)
       mmk = shakequad_mass(ii*7-4)
       mml = shakequad_mass(ii*7-3)
       mij = shakequad_mass(ii*7-2)
       mik = shakequad_mass(ii*7-1)
       mil = shakequad_mass(ii*7)
#else /**/
       mmi=ammi(i2)
       mmj=ammi(i2+1)
       mmk=ammi(i2+2)
       mml=ammi(i2+3)
       mij = ammi(i2+4)
       mik = ammi(i2+5)
       mil = ammi(i2+6)
#endif 

       acorr1 =    mij*mij*rrijsq
       acorr2 =two*mij*mmi*rijrik
       acorr3 =    mmi*mmi*rriksq
       acorr4 =    mmi*mmi*rrijsq
       acorr5 =two*mik*mmi*rijrik
       acorr6 =    mik*mik*rriksq
       acorr7 =two*mij*mmi*rijril
       acorr8 =two*mmi*mmi*rikril
       acorr9 =    mmi*mmi*rrilsq
       acorr10=two*mmi*mmi*rijril
       acorr11=two*mmi*mik*rikril
       acorr12=two*mmi*mmi*rijrik
       acorr13=two*mmi*mil*rijril
       acorr14=two*mmi*mil*rikril
       acorr15=    mil*mil*rrilsq

       xpij = x(i) - x(j)
       ypij = y(i) - y(j)
       zpij = z(i) - z(j)
       xpik = x(i) - x(k)
       ypik = y(i) - y(k)
       zpik = z(i) - z(k)
       xpil = x(i) - x(l)
       ypil = y(i) - y(l)
       zpil = z(i) - z(l)
       rijsq = xpij*xpij + ypij*ypij + zpij*zpij
       riksq = xpik*xpik + ypik*ypik + zpik*zpik
       rilsq = xpil*xpil + ypil*ypil + zpil*zpil

#ifdef FSSHAKE_DD
       constr1 = shakequad_constr(ii*3-2)
       constr2 = shakequad_constr(ii*3-1)
       constr3 = shakequad_constr(ii*3)
#else /**/
       constr1 = constr(iconst)
       constr2 = constr(iconst+1)
       constr3 = constr(iconst+2)
#endif 

       dij = constr1 - rijsq
       dik = constr2 - riksq
       dil = constr3 - rilsq
       rijrijp = xrij*xpij + yrij*ypij + zrij*zpij
       rijrikp = xrij*xpik + yrij*ypik + zrij*zpik
       rijrilp = xrij*xpil + yrij*ypil + zrij*zpil
       rikrijp = xrik*xpij + yrik*ypij + zrik*zpij
       rikrikp = xrik*xpik + yrik*ypik + zrik*zpik
       rikrilp = xrik*xpil + yrik*ypil + zrik*zpil
       rilrijp = xril*xpij + yril*ypij + zril*zpij
       rilrikp = xril*xpik + yril*ypik + zril*zpik
       rilrilp = xril*xpil + yril*ypil + zril*zpil
       d1=mik*mil*rikrikp*rilrilp - mmi*mmi*rikrilp*rilrikp
       d2=mmi*mil*rikrijp*rilrilp - mmi*mmi*rikrilp*rilrijp
       d3=mmi*mmi*rikrijp*rilrikp - mik*mmi*rikrikp*rilrijp
       d4=mmi*mil*rijrikp*rilrilp - mmi*mmi*rijrilp*rilrikp
       d5=mij*mil*rijrijp*rilrilp - mmi*mmi*rijrilp*rilrijp
       d6=mij*mmi*rijrijp*rilrikp - mmi*mmi*rijrikp*rilrijp
       d7=mmi*mmi*rijrikp*rikrilp - mmi*mik*rijrilp*rikrikp
       d8=mij*mmi*rijrijp*rikrilp - mmi*mmi*rijrilp*rikrijp
       d9=mij*mik*rijrijp*rikrikp - mmi*mmi*rijrikp*rikrijp
       dinv=half/(rijrijp*mij*d1 - mmi*rijrikp*d2 &
            + mmi*rijrilp*d3)

       a12=dinv*( d1*dij - d2*dik + d3*dil)
       a13=dinv*(-d4*dij + d5*dik - d6*dil)
       a14=dinv*( d7*dij - d8*dik + d9*dil)
       aniter=0
       do while(( &
            abs(a120-a12) > a12tol .or. &
            abs(a130-a13) > a13tol .or. &
            abs(a140-a14) > a14tol) &
            .and. &
            (aniter  <  max_niter))
          aniter=aniter+1
          a120=a12
          a130=a13
          a140=a14
          a12corr=acorr1*a12*a12+acorr2*a12*a13+acorr3*a13*a13 &
               +acorr7*a12*a14+acorr8*a13*a14+acorr9*a14*a14
          a13corr=acorr4*a12*a12+acorr5*a12*a13+acorr6*a13*a13 &
               +acorr10*a12*a14+acorr11*a13*a14+acorr9*a14*a14
          a14corr=acorr4*a12*a12+acorr12*a12*a13+acorr3*a13*a13 &
               +acorr13*a12*a14+acorr14*a13*a14+acorr15*a14*a14
          !
          a12=dinv*( d1*(dij-a12corr) - d2*(dik-a13corr) &
               + d3*(dil-a14corr))
          a13=dinv*( -d4*(dij-a12corr) + d5*(dik-a13corr) &
               - d6*(dil-a14corr))
          a14=dinv*( d7*(dij-a12corr) - d8*(dik-a13corr) &
               + d9*(dil-a14corr))
       enddo
       !
       x(i)  = x(i) + mmi*(xrij*a12+xrik*a13+xril*a14)
       y(i)  = y(i) + mmi*(yrij*a12+yrik*a13+yril*a14)
       z(i)  = z(i) + mmi*(zrij*a12+zrik*a13+zril*a14)
       x(j)  = x(j) - mmj*(xrij*a12)
       y(j)  = y(j) - mmj*(yrij*a12)
       z(j)  = z(j) - mmj*(zrij*a12)
       x(k)  = x(k) - mmk*(xrik*a13)
       y(k)  = y(k) - mmk*(yrik*a13)
       z(k)  = z(k) - mmk*(zrik*a13)
       x(l)  = x(l) - mml*(xril*a14)
       y(l)  = y(l) - mml*(yril*a14)
       z(l)  = z(l) - mml*(zril*a14)
#if KEY_CKSHKTOL==1
       xpij = x(i) - x(j)
       ypij = y(i) - y(j)
       zpij = z(i) - z(j)
       xpik = x(i) - x(k)
       ypik = y(i) - y(k)
       zpik = z(i) - z(k)
       xpil = x(i) - x(l)
       ypil = y(i) - y(l)
       zpil = z(i) - z(l)
       rijsq = xpij*xpij + ypij*ypij + zpij*zpij
       riksq = xpik*xpik + ypik*ypik + zpik*zpik
       rilsq = xpil*xpil + ypil*ypil + zpil*zpil
       dij = constr1 - rijsq
       dik = constr2 - riksq
       dil = constr3 - rilsq
       if(abs(dij) > d2tol2(iconst))then
          write(outu,*) "OUTSIDE TOLERANCE",ii,iconst,i,j,k,l
          write(outu,*) "TOLERANCE",d2tol2(iconst), &
               "  diff  ",dij
       endif
       if(abs(dik) > d2tol2(iconst+1))then
          write(OUTU,*)"OUTSIDE TOLERANCE",ii,iconst,i,j,k,l
          write(OUTU,*)"TOLERANCE",d2tol2(iconst+1), &
               "  diff  ",dik
       endif
       if(abs(dil) > d2tol2(iconst+2))then
          write(OUTU,*)"OUTSIDE TOLERANCE",ii,iconst,i,j,k,l
          write(OUTU,*)"TOLERANCE",d2tol2(iconst+2), &
               "  diff  ",dil
       endif
#endif 
       niter = max(niter, aniter)
    enddo

    return
  end subroutine ADD_SUFFIX(fsshakph_kernel4)
