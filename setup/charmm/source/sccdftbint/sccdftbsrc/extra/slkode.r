#
# Calculates Hamilton or Overlap matrix (depending on iovpar)
# for Atoms i and j, the result is stored in em
#
SUBROUTINE slkparam(i,j,xat,em,iovpar)
implicit none
include 'maxima.h'
integer i,j, izp(NNDIM), nbeweg, nlat,l1,l2,nu,nv,nw
logical period
real*8 xat(3,*),nel,em(LDIM,LDIM),em_tmp(LDIM,LDIM),boxsiz,xinvbox,xnullvec
common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(3)
# QC: izp-izpc
common /izpc/ izp, nel, nbeweg
external iovpar

integer izpi,izpj
real*8 dif(3)

if(period) {
 DO l1 = 1,LDIM {
  DO l2 = 1,LDIM {
    em(l1,l2) = 0.0d0
  }
 }
 izpi=izp(i)
 izpj=izp(j)
 
 DO nu =-nlat(1),nlat(1) {
  DO nv = -nlat(2),nlat(2) {
   DO nw = -nlat(3),nlat(3) {
    dif(1) = xat(1,j)-(xat(1,i)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,1))
    dif(2) = xat(2,j)-(xat(2,i)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,2))
    dif(3) = xat(3,j)-(xat(3,i)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,3)) 
 
    call SLKODE(dif,izpi,izpj,em_tmp,iovpar)
    DO l1 = 1,LDIM {
     DO l2 = 1,LDIM {
      em(l1,l2) = em(l1,l2) + em_tmp(l1,l2)
     }
    }
   }
  }
 }
}
else {
 dif(1)=xat(1,j)-xat(1,i)
 dif(2)=xat(2,j)-xat(2,i)
 dif(3)=xat(3,j)-xat(3,i)

 izpi=izp(i)
 izpj=izp(j)
 call SLKODE(dif,izpi,izpj,em,iovpar)
}

END

#
# Calculates Hamilton and Overlap matrix
# for Atoms i and j, the results are stored in ham,over
#
SUBROUTINE slkmatrices(i,j,xat,ham,over)
implicit none
include 'maxima.h'
integer i,j, izp(NNDIM), nbeweg, nlat, l1,l2,nu,nv,nw
logical period
real*8 xat(3,*),boxsiz,xinvbox,xnullvec, nel
real*8 ham(LDIM,LDIM),over(LDIM,LDIM),ham_tmp(LDIM,LDIM),over_tmp(LDIM,LDIM)
common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(3)
# QC: izp --> izpc
common /izpc/ izp, nel, nbeweg
external skspar,skhpar

integer izpi,izpj
real*8 dif(3)

if(period) {
 DO l1 = 1,LDIM {
  DO l2 = 1,LDIM {
    ham(l1,l2)  = 0.0d0
    over(l1,l2) = 0.0d0
  }
 }
 izpi=izp(i)
 izpj=izp(j)
 
 DO nu =-nlat(1),nlat(1) {
  DO nv = -nlat(2),nlat(2) {
   DO nw = -nlat(3),nlat(3) {
    dif(1) = xat(1,j)-(xat(1,i)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,1))
    dif(2) = xat(2,j)-(xat(2,i)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,2))
    dif(3) = xat(3,j)-(xat(3,i)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,3)) 
 
    call SLKODE(dif,izpi,izpj,ham_tmp,skhpar)
    call SLKODE(dif,izpi,izpj,over_tmp,skspar)
    DO l1 = 1,LDIM {
     DO l2 = 1,LDIM {
      ham(l1,l2)  = ham(l1,l2)  + ham_tmp(l1,l2)
      over(l1,l2) = over(l1,l2) + over_tmp(l1,l2)
     }
    }
   }
  }
 }
}
else {
 dif(1)=xat(1,j)-xat(1,i)
 dif(2)=xat(2,j)-xat(2,i)
 dif(3)=xat(3,j)-xat(3,i)

 izpi=izp(i)
 izpj=izp(j)

 call SLKODE(dif,izpi,izpj,ham,skhpar)
 call SLKODE(dif,izpi,izpj,over,skspar)
}

END


SUBROUTINE SLKODE(DUM,I,J,EM,iovpar)
implicit REAL*8 (A-H,O-Z)
include 'maxima.h'
real*8 DUM(3),EM(LDIM,LDIM)
real*8 X(6),X2(6),dummy(LDIM,LDIM)
external iovpar
common /lmax/ lmax(MAXTYP)
R2=0.0
DO L=1,3 {
 X(L)=DUM(L)
 X2(L)=X(L)*X(L)
 R2=R2+X2(L)
}
IF(R2 >= 1.0E-8) {
 R2I=1.0/R2
 RI=SQRT(R2I)
 DO L=1,3 {
  X(L)=X(L)*RI
  X(L+3)=X(L)
  X2(L)=X2(L)*R2I
  X2(L+3)=X2(L)
 }
 maxmax = max(lmax(i),lmax(j))
 minmax = min(lmax(i),lmax(j))
#
# s Interaction
#
 call skss(x,x2,i,j,r2,iovpar,em,LDIM)
 if (maxmax <= 1) return
#
# p Interaction
#
 if (minmax >= 2) {
  call skpp(x,x2,i,j,r2,iovpar,em(2,2),LDIM)
  call sksp(x,x2,i,j,r2,iovpar,em(1,2),em(2,1),LDIM)
  if (i != j)
   call sksp(x,x2,j,i,r2,iovpar,dummy,em(2,1),LDIM)
 }
 else if (lmax(j) >= 2)
  call sksp(x,x2,i,j,r2,iovpar,em(1,2),em(2,1),LDIM)
 else
  call sksp(x,x2,j,i,r2,iovpar,dummy,em(2,1),LDIM)
 if (maxmax <= 2) return
#
# d Interaction
#
 if (minmax == 3) {
  call skdd(x,x2,i,j,r2,iovpar,em(5,5),LDIM)
  call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),LDIM)
  call skpd(x,x2,i,j,r2,iovpar,em(2,5),em(5,2),LDIM)
  if (i != j) {
   call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),LDIM)
   call skpd(x,x2,j,i,r2,iovpar,dummy,em(5,2),LDIM)
  }
 }
 else if (lmax(i) == 1)
  call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),LDIM)
 else if (lmax(i) == 2) {
  call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),LDIM)
  call skpd(x,x2,i,j,r2,iovpar,em(2,5),em(5,2),LDIM)
 }
 else if (lmax(j) == 1)
  call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),LDIM)
 else {
  call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),LDIM)
  call skpd(x,x2,j,i,r2,iovpar,dummy,em(5,2),LDIM)
 }
}
else {
 do k = 1,LDIM {
  do l = 1,LDIM {
   em(k,l) = 0.
  }
 }
 if (i!=j) return
 call selfs(i,j,r2,iovpar,em,LDIM)
 if (lmax(i) <= 1) return
 call selfp(i,j,r2,iovpar,em(2,2),LDIM)
 if (lmax(i) <= 2) return
 call selfd(i,j,r2,iovpar,em(5,5),LDIM)
}
end

