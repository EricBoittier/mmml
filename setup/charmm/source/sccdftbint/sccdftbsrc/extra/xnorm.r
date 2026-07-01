#     SUBROUTINE XNORM
#     =================
#
#     Copyright 1991 by Peter Blaudeck
# *********************************************************************
#
#     PROGRAM CHARACTERISTICS
#     -----------------------
#
# XNORM moves the mass centre to the origin
#
# PARAMETERS:
# nn       i number of atoms
# xx       r coordinates
# xold     r      "      deltat earlier
# xm       r atom masses
# izp      i sort numbers
#
# *********************************************************************
#
subroutine xnorm(nn,xx,xold,xm,izp)
implicit REAL*8 (A-H,O-Z)
integer izp(*)
real*8 xx(3,*), xold(3,*), xm(*)
gmrc = 0.0
do i = 1,nn {
  gmrc = gmrc+xm(izp(i))
}
gmrc = 1.0/gmrc
do i = 1,3 {
  sum = 0.0
  do j = 1,nn {
    sum = sum+xx(i,j)*xm(izp(j))
  }
  sum = sum*gmrc
  do j = 1,nn {
    xx(i,j) = xx(i,j)-sum
    xold(i,j) = xold(i,j)-sum
  }
}
end

