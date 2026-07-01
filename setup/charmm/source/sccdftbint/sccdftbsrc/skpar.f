! Output from Public domain Ratfor, version 1.0
      integer function skspar(i,j,r2,dd)
      use sccdftbsrc, dr=>sr

      implicit real*8 (a-h,o-z)
!     integer nndim
!     parameter( nndim= 650)
!     integer maxint
!     parameter( maxint= 160)
!     integer mdim
!     parameter( mdim= 1650)
!     integer maxtyp
!     parameter( maxtyp= 6)
!     integer maxtab
!     parameter( maxtab= 600)
!     integer ldim
!     parameter( ldim= 9)
!     integer maxopt
!     parameter(maxopt=3*nndim)
!     integer maxsiz
!     parameter(maxsiz=mdim)
!     integer maxtp
!     parameter(maxtp=maxtyp)

      integer i,j
      real*8 r2,dd(13)

!     integer dim(maxtyp,maxtyp),mxind
      integer                    mxind
!     real*8 skhtab(10,maxtab,maxtyp,maxtyp),skstab(10,maxtab,maxtyp,max
!    *typ)
!     real*8 skself(3,maxtyp),dr(maxtyp,maxtyp),x0,x1,x2,f0,f1,f2
      real*8                                    x0,x1,x2,f0,f1,f2
      real*8 xh,hl
!     common /sktab/ skhtab,skstab,skself,dr,dim
!     common /lmax/ lmax(maxtyp)

      skspar = 0
      maxmax = max(lmax(i),lmax(j))
      minmax = min(lmax(i),lmax(j))
      if(maxmax .le. 1)then
      inu = 10
      else
      if(maxmax .le. 2)then
      if(minmax .le. 1)then
      inu = 9
      else
      inu = 6
      endif
      else
      if(minmax .le. 1)then
      inu = 8
      else
      if(minmax .le. 2)then
      inu = 4
      else
      inu = 1
      endif
      endif
      endif
      endif
      mxind=dim(i,j)+(0.3/dr(i,j)-1.0)
      r = sqrt(r2)
      ind = r/dr(i,j)+1.0
      if(r2 .lt. 1d-8)then
      do23012 in = 1,3 
      dd(in+10) = 1.0
23012 continue
23013 continue
      else
      if(ind+2 .gt. dim(i,j))then
      if(ind .lt. dim(i,j))then
      x0=(dim(i,j)-3)*dr(i,j)
      x1=x0+dr(i,j)
      x2=x1+dr(i,j)
      xh=r-x1
      hl=x2-x1
      do23018 in = inu,10 
      f0 = skstab(in,dim(i,j)-2,i,j)
      f1 = skstab(in,dim(i,j)-1,i,j)
      f2 = skstab(in,dim(i,j),i,j)
      dd(in)=cubicspline(f0,f1,f2,x0,x1,xh,hl,dr(i,j))
23018 continue
23019 continue
      else
      if( (ind .ge. dim(i,j)) .and. (ind .lt. mxind) )then
      x0=(dim(i,j)-3)*dr(i,j)
      x1=x0+dr(i,j)
      x2=x1+dr(i,j)
      xh=r-(mxind-1)*dr(i,j)
      do23022 in = inu,10 
      f0 = skstab(in,dim(i,j)-2,i,j)
      f1 = skstab(in,dim(i,j)-1,i,j)
      f2 = skstab(in,dim(i,j),i,j)
      dd(in)=spline5th(f0,f1,f2,x0,x1,x2,xh,dr(i,j),mxind)
23022 continue
23023 continue
      else
      do23024 in = inu,10 
      dd(in) = 0.0
23024 continue
23025 continue
      endif
      endif
      else
      grdr = (r-(ind-1.)*dr(i,j))/dr(i,j)
      do23026 in = inu,10 
      f0 = skstab(in,ind,i,j)
      f1 = skstab(in,ind+1,i,j)
      f2 = skstab(in,ind+2,i,j)
      dd(in) = f0 + (f1-f0)*grdr + (f2+f0-2.0*f1)*grdr*(grdr-1.0)/2.0
23026 continue
23027 continue
      endif
      endif
      end
      integer function skhpar(i,j,r2,dd)
      use sccdftbsrc, dr=>sr

      implicit real*8 (a-h,o-z)
!     integer nndim
!     parameter( nndim= 650)
!     integer maxint
!     parameter( maxint= 160)
!     integer mdim
!     parameter( mdim= 1650)
!     integer maxtyp
!     parameter( maxtyp= 6)
!     integer maxtab
!     parameter( maxtab= 600)
!     integer ldim
!     parameter( ldim= 9)
!     integer maxopt
!     parameter(maxopt=3*nndim)
!     integer maxsiz
!     parameter(maxsiz=mdim)
!     integer maxtp
!     parameter(maxtp=maxtyp)

      integer i,j
      real*8 r2,dd(13)

!     integer dim(maxtyp,maxtyp),mxind
      integer                    mxind
!     real*8 skhtab(10,maxtab,maxtyp,maxtyp),skstab(10,maxtab,maxtyp,max
!    *typ)
!     real*8 skself(3,maxtyp),dr(maxtyp,maxtyp),x0,x1,x2,f0,f1,f2
      real*8                                    x0,x1,x2,f0,f1,f2
      real*8 xh,hl
!     common /sktab/ skhtab,skstab,skself,dr,dim
!     common /lmax/ lmax(maxtyp)

      skhpar = 0
      maxmax = max(lmax(i),lmax(j))
      minmax = min(lmax(i),lmax(j))
      if(maxmax .le. 1)then
      inu = 10
      else
      if(maxmax .le. 2)then
      if(minmax .le. 1)then
      inu = 9
      else
      inu = 6
      endif
      else
      if(minmax .le. 1)then
      inu = 8
      else
      if(minmax .le. 2)then
      inu = 4
      else
      inu = 1
      endif
      endif
      endif
      endif
      mxind=dim(i,j)+(0.3/dr(i,j)-1.0)
      r = sqrt(r2)
      ind = r/dr(i,j)+1.0
      if(r2 .lt. 1d-8)then
      do23040 in = 1,3 
      dd(in+10) = skself(in,i)
23040 continue
23041 continue
      else
      if(ind+2 .gt. dim(i,j))then
      if(ind .lt. dim(i,j))then
      x0=(dim(i,j)-3)*dr(i,j)
      x1=x0+dr(i,j)
      x2=x1+dr(i,j)
      xh=r-x1
      hl=x2-x1
      do23046 in = inu,10 
      f0 = skhtab(in,dim(i,j)-2,i,j)
      f1 = skhtab(in,dim(i,j)-1,i,j)
      f2 = skhtab(in,dim(i,j),i,j)
      dd(in)=cubicspline(f0,f1,f2,x0,x1,xh,hl,dr(i,j))
23046 continue
23047 continue
      else
      if( (ind .ge. dim(i,j)) .and. (ind .lt. mxind) )then
      x0=(dim(i,j)-3)*dr(i,j)
      x1=x0+dr(i,j)
      x2=x1+dr(i,j)
      xh=r-(mxind-1)*dr(i,j)
      do23050 in = inu,10 
      f0 = skhtab(in,dim(i,j)-2,i,j)
      f1 = skhtab(in,dim(i,j)-1,i,j)
      f2 = skhtab(in,dim(i,j),i,j)
      dd(in)=spline5th(f0,f1,f2,x0,x1,x2,xh,dr(i,j),mxind)
23050 continue
23051 continue
      else
      do23052 in = inu,10 
      dd(in) = 0.0
23052 continue
23053 continue
      endif
      endif
      else
      grdr = (r-(ind-1.0)*dr(i,j))/dr(i,j)
      do23054 in = inu,10 
      f0 = skhtab(in,ind,i,j)
      f1 = skhtab(in,ind+1,i,j)
      f2 = skhtab(in,ind+2,i,j)
      dd(in) = f0 + (f1-f0)*grdr + (f2+f0-2.0*f1)*grdr*(grdr-1.0)/2.0
23054 continue
23055 continue
      endif
      endif
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
      jsp=6.0*f2/(hl*hl*hl*hl*hl)-3.0*f1abl/(hl*hl*hl*hl)+f2abl/(2.0*hl*
     *hl*hl)
      hl=xh*xh*xh
      spline5th=(hsp+isp*xh+jsp*xh*xh)*hl
      end

