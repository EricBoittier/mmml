! Explicit electrostatics: energy + analytic forces (conservative).
! Same expression as the JAX version so residual targets are consistent.
! Compile for f2py OR for a ctypes .so (see build commands at bottom of file).

subroutine elec_ef(n, p, R, q, pi, pj, ke, r_on, r_off, use_switch, energy, F)
  implicit none
  ! ---- args -------------------------------------------------------------
  integer,          intent(in)  :: n, p          ! n atoms, p pairs
  double precision, intent(in)  :: R(3, n)       ! column-major: R(:,atom)
  double precision, intent(in)  :: q(n)
  integer,          intent(in)  :: pi(p), pj(p)  ! 1-based atom indices
  double precision, intent(in)  :: ke, r_on, r_off
  integer,          intent(in)  :: use_switch
  double precision, intent(out) :: energy
  double precision, intent(out) :: F(3, n)
  ! ---- locals -----------------------------------------------------------
  integer          :: k, a, b, d
  double precision :: dx(3), dist2, dist, rinv, e_pair, qq
  double precision :: s, dsdr, ron2, roff2, x, denom, de_dr, fmag, dir(3)

  energy = 0.d0
  F = 0.d0
  ron2  = r_on  * r_on
  roff2 = r_off * r_off
  denom = roff2 - ron2

  !$omp parallel do private(k,a,b,d,dx,dist2,dist,rinv,qq,e_pair,s,dsdr,x,de_dr,fmag,dir) &
  !$omp   reduction(+:energy,F)
  do k = 1, p
     a = pi(k); b = pj(k)
     do d = 1, 3
        dx(d) = R(d, a) - R(d, b)
     end do
     dist2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3)
     dist  = sqrt(dist2)
     rinv = 1.d0 / dist
     qq = ke * q(a) * q(b)
     e_pair = qq * rinv                       ! bare Coulomb energy

     if (use_switch /= 0) then
        if (dist >= r_off) then
           cycle                              ! zero contribution
        else if (dist <= r_on) then
           s = 1.d0; dsdr = 0.d0
        else
           x    = (roff2 - dist2) / denom
           s    = x*x*(3.d0 - 2.d0*x)
           ! ds/dr = ds/dx * dx/dr,  dx/dr = -2r/denom
           dsdr = (6.d0*x - 6.d0*x*x) * (-2.d0*dist/denom)
        end if
     else
        s = 1.d0; dsdr = 0.d0
     end if

     energy = energy + e_pair * s

     ! E = e_pair(r) * s(r);  dE/dr = (-qq/r^2)*s + e_pair*dsdr
     de_dr = (-qq * rinv * rinv) * s + e_pair * dsdr
     fmag  = -de_dr                           ! scalar force along +r_hat on atom a
     do d = 1, 3
        dir(d) = dx(d) * rinv
        F(d, a) = F(d, a) + fmag * dir(d)
        F(d, b) = F(d, b) - fmag * dir(d)
     end do
  end do
  !$omp end parallel do
end subroutine elec_ef


! ----- ctypes entry point: same math, C-friendly signature (all by pointer) -----
subroutine elec_ef_c(n, p, R, q, pi, pj, ke, r_on, r_off, use_switch, energy, F) &
     bind(C, name="elec_ef_c")
  use iso_c_binding, only: c_int, c_double
  implicit none
  integer(c_int),   intent(in), value :: n, p, use_switch
  real(c_double),   intent(in)  :: R(3, n), q(n)
  integer(c_int),   intent(in)  :: pi(p), pj(p)
  real(c_double),   intent(in), value :: ke, r_on, r_off
  real(c_double),   intent(out) :: energy
  real(c_double),   intent(out) :: F(3, n)
  call elec_ef(n, p, R, q, pi, pj, ke, r_on, r_off, use_switch, energy, F)
end subroutine elec_ef_c

! Build (f2py module):
!   f2py -c -m elec_f2py elec_kernel.f90 --f90flags="-fopenmp" -lgomp
! Build (ctypes shared lib):
!   gfortran -O3 -fopenmp -shared -fPIC -o libelec.so elec_kernel.f90
