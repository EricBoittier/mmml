#     SUBROUTINE VNORM
#     =================
#
#     Copyright 1991 by Peter Blaudeck
# *********************************************************************
#
#     PROGRAM CHARACTERISTICS
#     -----------------------
#
# Scaling of velocities, total impulse = 0
#
# PARAMETERS:
# nn        i number of atoms 
# x         r coordinates, a.u.
# xold      r      "      deltat earlier
# t         r temperature, K
# deltat    r time stepwidth, a.u.
# xm        r atom masses, nucleon masses
# izp       i sort numbers
#
# *********************************************************************
#
subroutine vnorm(nn,x,xold,t,deltat,xm,izp)
implicit REAL*8 (A-H,O-Z)
integer izp(*)
real*8 x(3,*), xold(3,*), xm(*)
parameter (r032 = 1.5 * 3.16679e-6)  # 3/2*k in H/K
gm = 0.0
#
# total mass
#
do j = 1,nn {
  gm = gm +xm(izp(j))
}
rcdt = 1.0/deltat
rcgm = 1.0/gm
#
# set total impuls = 0
#
do i = 1,3 {
  puls = 0.0
  do j = 1,nn {
    v = (x(i,j) - xold(i,j))*rcdt
    puls = puls+xm(izp(j))*v
  }
  puls = puls*deltat*rcgm
  do j = 1,nn {
    xold(i,j) = xold(i,j) + puls
  }
}
#
# rescale velocities according to temperature
#
ekin = 0.0
do i = 1,nn {
  do j = 1,3 {
    v = (x(j,i) - xold(j,i))*rcdt
    ekin = ekin+0.5*xm(izp(i))*v**2
  }
}
if (ekin > 0.0) fak = sqrt(nn*r032*t/(1822.887428*ekin))
else fak = 0.0
do i = 1,3 {
  do j = 1,nn {
    xold(i,j) = (xold(i,j) - x(i,j))*fak + x(i,j)
  }
}
end

