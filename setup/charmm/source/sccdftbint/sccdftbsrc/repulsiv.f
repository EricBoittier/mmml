! Output from Public domain Ratfor, version 1.0
      subroutine repulsive(nn,izp,period,x,boxsiz,xinvbox,nlat,erep)
      use sccdftbsrc, only: nndim
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
!     parameter( maxtab= 650)
!     integer ldim
!     parameter( ldim= 9)
!     integer maxopt
!     parameter(maxopt=3*nndim)
!     integer maxsiz
!     parameter(maxsiz=mdim)
!     integer maxtp
!     parameter(maxtp=maxtyp)

      real*8 x(3,*),boxsiz(3,3),xinvbox(3,3),erep
      integer nn,izp(nndim), nlat(3),nu,nv,nw
      logical period
      integer i,j,k,izpj,izpk
      real*8 dif(3),r,r2
!
! include the GHO common block ... PJ 07/2004
!
      logical qlink
      integer natqm,nqmlnk,iqlink,jqlink,kqlink,maxqmlink,mxqm16
      parameter (maxqmlink=5,mxqm16=16*maxqmlink)
      common/qlinki/natqm,nqmlnk,iqlink(maxqmlink),jqlink(3,maxqmlink),
     &              kqlink(maxqmlink)
      common/qlinkl/qlink
      integer iglnk,kglnk
      common/qlinkq/ iglnk(maxqmlink),kglnk(maxqmlink)
!
! empirical repulsion term between A-B ... PJ 07/2004 
!
      real*8 c1,c2,c4
      integer ibt 
      logical isqb
!
      if(period)then
      do23002 j = 1,nn 
      izpj = izp(j)
      do23004 k = j+1,nn 
      izpk = izp(k)
      do23006 nu =-nlat(1),nlat(1) 
      do23008 nv = -nlat(2),nlat(2) 
      do23010 nw = -nlat(3),nlat(3) 
      dif(1) = x(1,j)-(x(1,k)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,
     *1))
      dif(2) = x(2,j)-(x(2,k)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,
     *2))
      dif(3) = x(3,j)-(x(3,k)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,
     *3))
      r = sqrt(dif(1)**2 + dif(2)**2 + dif(3)**2)
      erep = erep + repen(r,izpj,izpk)
23010 continue
23011 continue
23008 continue
23009 continue
23006 continue
23007 continue
23004 continue
23005 continue
23002 continue
23003 continue
      else
      do23012 j = 1,nn 
      izpj = izp(j)
      do23014 k = j+1,nn 
      izpk = izp(k)
      r2 = 0.0
      do23016 i = 1,3 
      dif(i) = x(i,k) - x(i,j)
23016 continue
23017 continue
      r2 = dif(1)**2 + dif(2)**2 + dif(3)**2
      r = sqrt(r2)
      erep = erep + repen(r,izpj,izpk)
!
! an empirical correction for GHO ... PJ 7/2004
      if (qlink) then
          isqb = .false.
          do ibt = 1, nqmlnk
             if ((j.eq.iglnk(ibt) .and. k.eq.kglnk(ibt)) .or.
     &           (k.eq.iglnk(ibt) .and. j.eq.kglnk(ibt)) ) then
                 isqb = .true.
             end if
          end do
          if (isqb) then
             call gtcoef(izp,j,k,c1,c2,c4)
             erep = erep + qbrep(r,c1,c2,c4)
          end if
      end if
!
23014 continue
23015 continue
23012 continue
23013 continue
      endif
      end
      subroutine repulsivegrd(nn,nbeweg,izp,period,x,boxsiz,xinvbox,nlat
     *,grd)
      use sccdftbsrc, only: nndim
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
!     parameter( maxtab= 650)
!     integer ldim
!     parameter( ldim= 9)
!     integer maxopt
!     parameter(maxopt=3*nndim)
!     integer maxsiz
!     parameter(maxsiz=mdim)
!     integer maxtp
!     parameter(maxtp=maxtyp)

      real*8 x(3,*),boxsiz(3,3),xinvbox(3,3),grd(3,*)
      integer nn,nbeweg,izp(nndim),nlat(3),nu,nv,nw
      logical period
      integer i,j,k,izpj,izpk
      real*8 dif(3),dgr,grdr,r,r2
!
! include the GHO common block ... PJ 7/2004
!
      logical qlink
      integer natqm,nqmlnk,iqlink,jqlink,kqlink,maxqmlink,mxqm16
      parameter (maxqmlink=5,mxqm16=16*maxqmlink)
      common/qlinki/natqm,nqmlnk,iqlink(maxqmlink),jqlink(3,maxqmlink),
     &              kqlink(maxqmlink)
      common/qlinkl/qlink
      integer iglnk,kglnk
      common/qlinkq/ iglnk(maxqmlink),kglnk(maxqmlink)
!
! empirical repulsion term between A-B ... PJ 7/2004
!
      real*8 c1, c2, c4
      integer ibt
      logical isqb
!
      do23018 j = 1,nn 
      do23020 i = 1,3 
      grd(i,j) = 0.0
23020 continue
23021 continue
23018 continue
23019 continue
      if(period)then
      do23024 j = 1,nn 
      izpj = izp(j)
      do23026 k = 1,nn 
      izpk = izp(k)
      if(k.le.nbeweg)then
      do23030 nu =-nlat(1),nlat(1) 
      do23032 nv = -nlat(2),nlat(2) 
      do23034 nw = -nlat(3),nlat(3) 
      dif(1) = x(1,k)-(x(1,j)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,
     *1))
      dif(2) = x(2,k)-(x(2,j)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,
     *2))
      dif(3) = x(3,k)-(x(3,j)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,
     *3))
      r = sqrt(dif(1)**2 + dif(2)**2 + dif(3)**2)
      if(r .ge. 1.0d-2)then
      grdr = grdrep(r,izpj,izpk)/r
      do23038 i = 1,3 
      dgr = dif(i)*grdr
      grd(i,k) = grd(i,k) + dgr
23038 continue
23039 continue
      endif
23034 continue
23035 continue
23032 continue
23033 continue
23030 continue
23031 continue
      endif
23026 continue
23027 continue
23024 continue
23025 continue
      else
      do23040 j = 1,nn 
      izpj = izp(j)
      do23042 k = 1,nn 
      izpk = izp(k)
      r2 = 0.0
      do23044 i = 1,3 
      dif(i) = x(i,k) - x(i,j)
23044 continue
23045 continue
      r2 = dif(1)**2 + dif(2)**2 + dif(3)**2
      r = sqrt(r2)
      if(k .le. nbeweg .and. r .ge. 1.0d-2)then
      grdr = grdrep(r,izpj,izpk)/r
      do23048 i = 1,3 
      dgr = dif(i)*grdr
      grd(i,k) = grd(i,k) + dgr
!
! gradient for the GHO empirical correction term ... PJ 7/2004
!
      if (qlink) then
          isqb = .false.
          do ibt = 1, nqmlnk
             if ((j.eq.iglnk(ibt) .and. k.eq.kglnk(ibt)) .or.
     &          (k.eq.iglnk(ibt) .and. j.eq.kglnk(ibt)) ) then
                isqb = .true.
             end if
          end do
          if (isqb) then
             call gtcoef(izp,j,k,c1,c2,c4)
             grd(i,k) = grd(i,k)
     &                  + qbgrd(r,c1,c2,c4)*dif(i)/r
          end if
      end if
!
23048 continue
23049 continue
      endif
23042 continue
23043 continue
23040 continue
23041 continue
      endif
      end
      real*8 function repen(r,izpj,izpk)
      use sccdftbsrc

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

      real*8 r
      integer izpj,izpk
      real*8 fhelp,xh,xv1
      integer i,j

!     real*8 coeff(6,maxint,maxtyp,maxtyp),xr(2,maxint,maxtyp,maxtyp)
!     real*8 efkt(3,maxtyp,maxtyp),cutoff(maxtyp,maxtyp)
!     integer numint(maxtyp,maxtyp)
!     QC haibo UW_06: Add SRP Gaussian
!     real*8 gbump(3,maxtyp,maxtyp),ronsw(maxtyp,maxtyp),ron,roff,sw
!     real*8 roff0,deltar(maxtyp,maxtyp)
!     logical lgausw
!
!     common /spltab/ coeff,xr,efkt,cutoff,numint,gbump,ronsw,deltar,
!    &       lgausw

      if(r.lt.1.0d-2)then
      fhelp=0.0
      else
      if(r.lt.xr(1,1,izpj,izpk))then
      fhelp=exp(-efkt(1,izpj,izpk)*r+  efkt(2,izpj,izpk))+efkt(3,izpj,iz
     *pk)
      else
      if(r.gt.cutoff(izpj,izpk))then
      fhelp=0.0
      else
      do23056 i=1,numint(izpj,izpk) 
      if(r.ge.xr(1,i,izpj,izpk) .and. r.le.xr(2,i,izpj,izpk))then
      goto 23057
      endif
23056 continue
23057 continue
      xv1=r-xr(1,i,izpj,izpk)
      fhelp=coeff(1,i,izpj,izpk)
      xh=xv1
      if(i.lt.numint(izpj,izpk))then
      do23062 j=2,4 
      fhelp=fhelp+coeff(j,i,izpj,izpk)*xh
      xh=xh*xv1
23062 continue
23063 continue
      else
      do23064 j=2,6 
      fhelp=fhelp+coeff(j,i,izpj,izpk)*xh
      xh=xh*xv1
23064 continue
23065 continue
      endif
      endif
      endif
      endif
!     QC_Haibo UW_06 Adds Gaussian for SRP
      if (lgausw) then
        ron=ronsw(izpj,izpk)
        roff=ron+deltar(izpj,izpk)
        roff0=ron-deltar(izpj,izpk)

        if (r .le. roff0) then
          sw = 0  
        else if (r .gt. roff0 .and. r .le. ron) then
          sw = (roff0*roff0-r*r)**2*(roff0*roff0+2*r*r-3*ron*ron)
     $           /(roff0*roff0-ron*ron)**3
        else if (r .gt. ron .and. r .le. roff) then
          sw = (roff*roff-r*r)**2*(roff*roff+2*r*r-3*ron*ron)
     $           /(roff*roff-ron*ron)**3      
        else    
          sw = 0  
        endif   
        fhelp=fhelp+sw*gbump(1,izpj,izpk)*
     $    exp(-(r-gbump(2,izpj,izpk))**2/gbump(3,izpj,izpk))
      endif   

      repen=fhelp
      end
      real*8 function grdrep(r,izpj,izpk)
      use sccdftbsrc
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

      real*8 r
      integer izpj,izpk
      real*8 grdr,xv1,xh
      integer i,j

!     real*8 coeff(6,maxint,maxtyp,maxtyp),xr(2,maxint,maxtyp,maxtyp)
!     real*8 efkt(3,maxtyp,maxtyp),cutoff(maxtyp,maxtyp)
!     integer numint(maxtyp,maxtyp)
!     QC haibo UW_06: Add SRP Gaussian
!     real*8 gbump(3,maxtyp,maxtyp),ronsw(maxtyp,maxtyp),ron,roff,sw
!     real*8 roff0,deltar(maxtyp,maxtyp)
!     logical lgausw
!
!     common /spltab/ coeff,xr,efkt,cutoff,numint,gbump,ronsw,deltar,
!    &       lgausw

      grdr = 0.0
      if(r.lt.1.0d-2)then
      grdr=0.0
      else
      if(r.lt.xr(1,1,izpj,izpk))then
      grdr=-efkt(1,izpj,izpk)* exp(-efkt(1,izpj,izpk)*r+efkt(2,izpj,izpk
     *))
      else
      if(r.gt.cutoff(izpj,izpk))then
      grdr=0.0
      else
      do23072 i=1,numint(izpj,izpk) 
      if(r.ge.xr(1,i,izpj,izpk) .and. r.le.xr(2,i,izpj,izpk))then
      goto 23073
      endif
23072 continue
23073 continue
      xv1=r-xr(1,i,izpj,izpk)
      xh=1
      if(i.lt.numint(izpj,izpk))then
      do23078 j=2,4 
      grdr=grdr+(j-1)*coeff(j,i,izpj,izpk)*xh
      xh=xh*xv1
23078 continue
23079 continue
      else
      do23080 j=2,6 
      grdr=grdr+(j-1)*coeff(j,i,izpj,izpk)*xh
      xh=xh*xv1
23080 continue
23081 continue
      endif
      endif
      endif
      endif

!     QC_Haibo UW_06 Adds SRP Gaussian
      if (lgausw) then

      ron=ronsw(izpj,izpk)
      roff=ron+deltar(izpj,izpk)
      roff0=ron-deltar(izpj,izpk)

      gau=gbump(1,izpj,izpk)*
     $    exp(-(r-gbump(2,izpj,izpk))**2/gbump(3,izpj,izpk))
! MG_QC_UW1205: bugfix
      gaugrad=-2*gbump(1,izpj,izpk)*(r-gbump(2,izpj,izpk))
     $    /gbump(3,izpj,izpk)*
     $    exp(-(r-gbump(2,izpj,izpk))**2/gbump(3,izpj,izpk))
      if (r .le. roff0) then
          sw = 0
          swgrad = 0
      else if (r .gt. roff0 .and. r .le. ron) then
          sw = (roff0*roff0-r*r)**2*(roff0*roff0+2*r*r-3*ron*ron)
     $           /(roff0*roff0-ron*ron)**3
          swgrad=-4*r*(roff0*roff0-r*r)*(roff0*roff0+2*r*r-3*ron*ron)
     $          /(roff0*roff0-ron*ron)**3
     $           +4*r*(roff0*roff0-r*r)**2/(roff0*roff0-ron*ron)**3
      else if (r .gt. ron .and. r .le. roff) then
          sw = (roff*roff-r*r)**2*(roff*roff+2*r*r-3*ron*ron)
     $           /(roff*roff-ron*ron)**3
          swgrad=-4*r*(roff*roff-r*r)*(roff*roff+2*r*r-3*ron*ron)
     $          /(roff*roff-ron*ron)**3
     $           +4*r*(roff*roff-r*r)**2/(roff*roff-ron*ron)**3
      else
          sw=0
          swgrad=0
      endif
!
      grdr=grdr+gaugrad*sw+gau*swgrad
      endif

      grdrep=grdr
      end


