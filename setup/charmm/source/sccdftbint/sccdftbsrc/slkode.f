! Output from Public domain Ratfor, version 1.0
      subroutine slkparam(i,j,xat,em,iovpar)
      use sccdftbsrc, only: ldim, izp=>izp2, nel, nbeweg
      use sccdftb, only: boxsiz,xinvbox,xnullvec,period,nlat
      implicit none
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
!     integer i,j, izp(nndim), nbeweg, nlat,l1,l2,nu,nv,nw
      integer i,j,                          l1,l2,nu,nv,nw
!     logical period
      real*8 xat(3,*),    em(ldim,ldim),em_tmp(ldim,ldim)
!     real*8 xat(3,*),nel,em(ldim,ldim),em_tmp(ldim,ldim),boxsiz,xinvbox
!    *,xnullvec

!     common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(
!    *3)
!     common /izpc/ izp, nel, nbeweg

      external iovpar
      integer izpi,izpj
      real*8 dif(3)
      if(period)then
      do23002 l1 = 1,ldim 
      do23004 l2 = 1,ldim 
      em(l1,l2) = 0.0d0
23004 continue
23005 continue
23002 continue
23003 continue
      izpi=izp(i)
      izpj=izp(j)
      do23006 nu =-nlat(1),nlat(1) 
      do23008 nv = -nlat(2),nlat(2) 
      do23010 nw = -nlat(3),nlat(3) 
      dif(1) = xat(1,j)-(xat(1,i)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsi
     *z(3,1))
      dif(2) = xat(2,j)-(xat(2,i)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsi
     *z(3,2))
      dif(3) = xat(3,j)-(xat(3,i)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsi
     *z(3,3))
      call slkode(dif,izpi,izpj,em_tmp,iovpar)
      do23012 l1 = 1,ldim 
      do23014 l2 = 1,ldim 
      em(l1,l2) = em(l1,l2) + em_tmp(l1,l2)
23014 continue
23015 continue
23012 continue
23013 continue
23010 continue
23011 continue
23008 continue
23009 continue
23006 continue
23007 continue
      else
      dif(1)=xat(1,j)-xat(1,i)
      dif(2)=xat(2,j)-xat(2,i)
      dif(3)=xat(3,j)-xat(3,i)
      izpi=izp(i)
      izpj=izp(j)
      call slkode(dif,izpi,izpj,em,iovpar)
      endif
      end
      subroutine slkmatrices(i,j,xat,ham,over)
      use sccdftbsrc, only: nndim,ldim, izp=>izp2, nel, nbeweg
      use sccdftb, only: boxsiz,xinvbox,xnullvec,period,nlat
      implicit none
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

!     integer i,j, izp(nndim), nbeweg, nlat, l1,l2,nu,nv,nw
      integer i,j,                           l1,l2,nu,nv,nw
!     logical period
!     real*8 xat(3,*),boxsiz,xinvbox,xnullvec, nel
      real*8 xat(3,*)
      real*8 ham(ldim,ldim),over(ldim,ldim),ham_tmp(ldim,ldim),over_tmp(
     *ldim,ldim)
!     common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(
!    *3)
!     common /izpc/ izp, nel, nbeweg

      external skspar,skhpar
      integer izpi,izpj
      real*8 dif(3)
      if(period)then
      do23018 l1 = 1,ldim 
      do23020 l2 = 1,ldim 
      ham(l1,l2) = 0.0d0
      over(l1,l2) = 0.0d0
23020 continue
23021 continue
23018 continue
23019 continue
      izpi=izp(i)
      izpj=izp(j)
      do23022 nu =-nlat(1),nlat(1) 
      do23024 nv = -nlat(2),nlat(2) 
      do23026 nw = -nlat(3),nlat(3) 
      dif(1) = xat(1,j)-(xat(1,i)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsi
     *z(3,1))
      dif(2) = xat(2,j)-(xat(2,i)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsi
     *z(3,2))
      dif(3) = xat(3,j)-(xat(3,i)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsi
     *z(3,3))
      call slkode(dif,izpi,izpj,ham_tmp,skhpar)
      call slkode(dif,izpi,izpj,over_tmp,skspar)
      do23028 l1 = 1,ldim 
      do23030 l2 = 1,ldim 
      ham(l1,l2) = ham(l1,l2) + ham_tmp(l1,l2)
      over(l1,l2) = over(l1,l2) + over_tmp(l1,l2)
23030 continue
23031 continue
23028 continue
23029 continue
23026 continue
23027 continue
23024 continue
23025 continue
23022 continue
23023 continue
      else
      dif(1)=xat(1,j)-xat(1,i)
      dif(2)=xat(2,j)-xat(2,i)
      dif(3)=xat(3,j)-xat(3,i)
      izpi=izp(i)
      izpj=izp(j)
      call slkode(dif,izpi,izpj,ham,skhpar)
      call slkode(dif,izpi,izpj,over,skspar)
      endif
      end
      subroutine slkode(dum,i,j,em,iovpar)
      use sccdftbsrc, only: ldim, lmax

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

      real*8 dum(3),em(ldim,ldim)
      real*8 x(6),x2(6),dummy(ldim,ldim)
      external iovpar
!     common /lmax/ lmax(maxtyp)
      r2=0.0
      do23032 l=1,3 
      x(l)=dum(l)
      x2(l)=x(l)*x(l)
      r2=r2+x2(l)
23032 continue
23033 continue
      if(r2 .ge. 1.0e-8)then
      r2i=1.0/r2
      ri=sqrt(r2i)
      do23036 l=1,3 
      x(l)=x(l)*ri
      x(l+3)=x(l)
      x2(l)=x2(l)*r2i
      x2(l+3)=x2(l)
23036 continue
23037 continue
      maxmax = max(lmax(i),lmax(j))
      minmax = min(lmax(i),lmax(j))
      call skss(x,x2,i,j,r2,iovpar,em,ldim)
      if(maxmax .le. 1)then
      return
      endif
      if(minmax .ge. 2)then
      call skpp(x,x2,i,j,r2,iovpar,em(2,2),ldim)
      call sksp(x,x2,i,j,r2,iovpar,em(1,2),em(2,1),ldim)
      if(i .ne. j)then
      call sksp(x,x2,j,i,r2,iovpar,dummy,em(2,1),ldim)
      endif
      else
      if(lmax(j) .ge. 2)then
      call sksp(x,x2,i,j,r2,iovpar,em(1,2),em(2,1),ldim)
      else
      call sksp(x,x2,j,i,r2,iovpar,dummy,em(2,1),ldim)
      endif
      endif
      if(maxmax .le. 2)then
      return
      endif
      if(minmax .eq. 3)then
      call skdd(x,x2,i,j,r2,iovpar,em(5,5),ldim)
      call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),ldim)
      call skpd(x,x2,i,j,r2,iovpar,em(2,5),em(5,2),ldim)
      if(i .ne. j)then
      call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),ldim)
      call skpd(x,x2,j,i,r2,iovpar,dummy,em(5,2),ldim)
      endif
      else
      if(lmax(i) .eq. 1)then
      call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),ldim)
      else
      if(lmax(i) .eq. 2)then
      call sksd(x,x2,i,j,r2,iovpar,em(1,5),em(5,1),ldim)
      call skpd(x,x2,i,j,r2,iovpar,em(2,5),em(5,2),ldim)
      else
      if(lmax(j) .eq. 1)then
      call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),ldim)
      else
      call sksd(x,x2,j,i,r2,iovpar,dummy,em(5,1),ldim)
      call skpd(x,x2,j,i,r2,iovpar,dummy,em(5,2),ldim)
      endif
      endif
      endif
      endif
      else
      do23058 k = 1,ldim 
      do23060 l = 1,ldim 
      em(k,l) = 0.
23060 continue
23061 continue
23058 continue
23059 continue
      if(i.ne.j)then
      return
      endif
      call selfs(i,j,r2,iovpar,em,ldim)
      if(lmax(i) .le. 1)then
      return
      endif
      call selfp(i,j,r2,iovpar,em(2,2),ldim)
      if(lmax(i) .le. 2)then
      return
      endif
      call selfd(i,j,r2,iovpar,em(5,5),ldim)
      endif
      end

