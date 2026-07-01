#
# looking for the overlap matrix parameters
#  use Newton's approximation for the calculation of elements
#  and in the end a spline
#
integer function skspar(i,j,r2,dd)
implicit REAL*8 (A-H,O-Z)
include 'maxima.h'
integer i,j
real*8 r2,dd(13)
integer dim(MAXTYP,MAXTYP),mxind
real*8 skhtab(10,MAXTAB,MAXTYP,MAXTYP),skstab(10,MAXTAB,MAXTYP,MAXTYP)
real*8 skself(3,MAXTYP),dr(MAXTYP,MAXTYP),x0,x1,x2,f0,f1,f2
real*8 xh,hl
common /sktab/ skhtab,skstab,skself,dr,dim
common /lmax/ lmax(MAXTYP)

skspar = 0
maxmax = max(lmax(i),lmax(j))
minmax = min(lmax(i),lmax(j))
if (maxmax <= 1) inu = 10
else if (maxmax <= 2) {
 if (minmax <= 1) inu = 9
 else inu = 6
}
else {
 if (minmax <= 1) inu = 8
 else if (minmax <= 2) inu = 4
 else inu = 1
}
#
# mxind = Maximaler Index bis zu dem der Spline weitergefuehrt wird
#
mxind=dim(i,j)+(0.3/dr(i,j)-1.0)
r = sqrt(r2)
ind = r/dr(i,j)+1.0
if (r2 < 1d-8) {
 do in = 1,3 {
  dd(in+10) = 1.0
 }
}
else if (ind+2 > dim(i,j)) {
#
# neu einfuegen
#
 if (ind < dim(i,j)) {
#
# free cubic spline
#
  x0=(dim(i,j)-3)*dr(i,j)
  x1=x0+dr(i,j)
  x2=x1+dr(i,j)
  xh=r-x1
  hl=x2-x1
  do in = inu,10 {
   f0 = skstab(in,dim(i,j)-2,i,j)
   f1 = skstab(in,dim(i,j)-1,i,j)
   f2 = skstab(in,dim(i,j),i,j)
   dd(in)=cubicspline(f0,f1,f2,x0,x1,xh,hl,dr(i,j))
  }
 }
 else if ( (ind >= dim(i,j)) && (ind < mxind) ) {
#
# 5th degree spline
#
  x0=(dim(i,j)-3)*dr(i,j)
  x1=x0+dr(i,j)
  x2=x1+dr(i,j)
  xh=r-(mxind-1)*dr(i,j)
  do in = inu,10 {
   f0 = skstab(in,dim(i,j)-2,i,j)
   f1 = skstab(in,dim(i,j)-1,i,j)
   f2 = skstab(in,dim(i,j),i,j)
   dd(in)=spline5th(f0,f1,f2,x0,x1,x2,xh,dr(i,j),mxind)
  }
 }
 else {
#
# zero
#
  do in = inu,10 {
   dd(in) = 0.0
  }
 }
#
# rem Ende
#
}
else {
 grdr = (r-(ind-1.)*dr(i,j))/dr(i,j)
 do in = inu,10 {
  f0 = skstab(in,ind,i,j)
  f1 = skstab(in,ind+1,i,j)
  f2 = skstab(in,ind+2,i,j)
  dd(in) = f0 + (f1-f0)*grdr + (f2+f0-2.0*f1)*grdr*(grdr-1.0)/2.0
 }
}
end


#
# looking for the hailton matrix parameters
#  use Newton's approximation for the calculation of elements
#  and in the end a spline (same routine as for ovelapp)
#
integer function skhpar(i,j,r2,dd)
implicit REAL*8 (A-H,O-Z)
include 'maxima.h' 
integer i,j
real*8 r2,dd(13)
integer dim(MAXTYP,MAXTYP),mxind
real*8 skhtab(10,MAXTAB,MAXTYP,MAXTYP),skstab(10,MAXTAB,MAXTYP,MAXTYP)
real*8 skself(3,MAXTYP),dr(MAXTYP,MAXTYP),x0,x1,x2,f0,f1,f2
real*8 xh,hl
common /sktab/ skhtab,skstab,skself,dr,dim
common /lmax/ lmax(MAXTYP)

skhpar = 0
maxmax = max(lmax(i),lmax(j))
minmax = min(lmax(i),lmax(j))
if (maxmax <= 1) inu = 10
else if (maxmax <= 2) {
 if (minmax <= 1) inu = 9
 else inu = 6
}
else {
 if (minmax <= 1) inu = 8
 else if (minmax <= 2) inu = 4
 else inu = 1
}
#
# mxind = Maximaler Index bis zu dem der Spline weitergefuehrt wird
#
mxind=dim(i,j)+(0.3/dr(i,j)-1.0)
r = sqrt(r2)
ind = r/dr(i,j)+1.0
if (r2 < 1d-8) {
 do in = 1,3 {
  dd(in+10) = skself(in,i)
 }
}
else if (ind+2 > dim(i,j)) {
#
# neu einfuegen
#
 if (ind < dim(i,j)) {
#
# free cubic spline
#
  x0=(dim(i,j)-3)*dr(i,j)
  x1=x0+dr(i,j)
  x2=x1+dr(i,j)
  xh=r-x1
  hl=x2-x1
  do in = inu,10 {
   f0 = skhtab(in,dim(i,j)-2,i,j)
   f1 = skhtab(in,dim(i,j)-1,i,j)
   f2 = skhtab(in,dim(i,j),i,j)
   dd(in)=cubicspline(f0,f1,f2,x0,x1,xh,hl,dr(i,j))
  }
 }
 else if ( (ind >= dim(i,j)) && (ind < mxind) ) {
#
# 5th degree spline
#
    x0=(dim(i,j)-3)*dr(i,j)
  x1=x0+dr(i,j)
  x2=x1+dr(i,j)
  xh=r-(mxind-1)*dr(i,j)
  do in = inu,10 {
   f0 = skhtab(in,dim(i,j)-2,i,j)
   f1 = skhtab(in,dim(i,j)-1,i,j)
   f2 = skhtab(in,dim(i,j),i,j)
   dd(in)=spline5th(f0,f1,f2,x0,x1,x2,xh,dr(i,j),mxind)
  }
 }
 else {
#
# zero
#
  do in = inu,10 {
   dd(in) = 0.0
  }
 }
#
# rem Ende
#
}
else {
 grdr = (r-(ind-1.0)*dr(i,j))/dr(i,j)
 do in = inu,10 {
  f0 = skhtab(in,ind,i,j)
  f1 = skhtab(in,ind+1,i,j)
  f2 = skhtab(in,ind+2,i,j)
  dd(in) = f0 + (f1-f0)*grdr + (f2+f0-2.0*f1)*grdr*(grdr-1.0)/2.0
 }
}
end

real*8 function cubicspline(f0,f1,f2,x0,x1,xh,hl,dr)
   implicit none
   real*8 f0,f1,f2,x0,x1,xh,hl,dr
   real*8 f1abl,f2abl,a,b,c,d
   
   f2abl=(f2+f0-2.0*f1)/(dr*dr)
   f1abl=(f1-f0)/dr+0.5*f2abl*(x1-x0)
   a=f1
   b=f1abl
   c=f2abl/2.0
   d=(f2-a)/(hl*hl*hl)-b/(hl*hl)-c/hl
  
   cubicspline=a+b*xh+c*xh*xh+d*xh*xh*xh
end

real*8 function spline5th(f0,f1,f2,x0,x1,x2,xh,dr,mxind)
   implicit none
   real*8 f0,f1,f2,x0,x1,x2,xh,dr
   integer mxind
   real*8 hl,f1abl,f2abl,a,b,c,d,hsp,isp,jsp

   f2abl=(f2+f0-2.0*f1)/(dr*dr)
   f1abl=(f1-f0)/dr+0.5*f2abl*(x1-x0)
   a=f1
   b=f1abl
   c=f2abl/2.0
   hl=x2-x1
   d=(f2-a)/(hl*hl*hl)-b/(hl*hl)-c/hl

   f1abl=b+2.0*c*hl+3.0*d*hl*hl
   f2abl=2.0*c+6.0*d*hl
   
   hl=x2-(mxind-1)*dr
   hsp=10.0*f2/(hl*hl*hl)-4.0*f1abl/(hl*hl)+f2abl/(2.0*hl)
   isp=-15.0*f2/(hl*hl*hl*hl)+7.0*f1abl/(hl*hl*hl)-f2abl/(hl*hl)
   jsp=6.0*f2/(hl*hl*hl*hl*hl)-3.0*f1abl/(hl*hl*hl*hl)+f2abl/(2.0*hl*hl*hl)
   
   hl=xh*xh*xh  
   spline5th=(hsp+isp*xh+jsp*xh*xh)*hl
end

