! Output from Public domain Ratfor, version 1.0
! MG_UW1210,MG_QC_UW1207: arguments added
      subroutine usualgrd(nn,nbeweg,ndim,izp,lmax,ind,period,doscf, 
     *x,ev,a,b,occ,shift,shift3,shift3A,shift3B,shiftE,shiftE2,
     *dacc,boxsiz,xinvbox,grad,lldep,lcolspin,spinfac,spinshift)
      use sccdftbsrc, only: mdim, ldim, lscc3rd, nndim
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
      real*8 deltax
      parameter ( deltax = 0.0001 )
      logical period,doscf,lcolspin,lldep ! MG_QC_UW1207,MG_UW1210
      integer nn,nbeweg,ndim,izp(*),lmax(*),ind(*)
      real*8 x(3,*),ev(*),a(mdim,mdim),b(mdim,mdim),occ(*),grad(3,*)
      real*8 dacc,boxsiz(3,3),xinvbox(3,3),shift(nndim,3) !MG_UW1210
      real*8 shift3(nndim,3),shift3A(nndim,3),shift3b(nndim),dgr3 ! MG+Guanhua_QC_UW1205,MG_UW1210
      real*8 shiftE(*),shiftE2(*) ! MG+Guanhua_QC_UW1206
      real*8 spinfac,spinshift(nndim,3) ! MG_QC_UW1207
!      logical lscc3rd
!      common /scc3/ lscc3rd
! end
      integer m,n,i,j,k,lj,lk,mj,mk
      integer mu,nu,izpj,izpk,indj,indk
      real*8 au(ldim,ldim),bu(ldim,ldim),ocmcc,dgrh,rcdx,xhelp,dgrs,dgr
      real*8 auh(ldim,ldim),buh(ldim,ldim),dgrspin ! MG_QC_UW1207
!
! include the GHO switch ... PJ 7/2004
      logical qlink
      common/qlinkl/qlink
!
      rcdx=1.0/deltax
      do23000 i=1,nn 
      do23002 k=1,3 
      grad(k,i) = 0.0
23002 continue
23003 continue
23000 continue
23001 continue

! MG+Guanhua_QC_UW1205
      dgr=0.0d0
      dgr3=0.0d0
! end

      do23004 n = 1,ndim 
      do23006 m = 1,ndim 
      b(m,n) = 0.0
23006 continue
23007 continue
23004 continue
23005 continue

      do23008 i = 1,ndim 
!
! for GHO, we have to loop over all orbitals, becuase
! auxiliary orbitals are also occupied ... PJ 7/2004
!
      if (.not. qlink) then
         if(occ(i) .lt. dacc)then
            goto 23009
         endif
      endif
!
      do23012 m = 1,ndim 
      do23014 n = 1,m-1 
      ocmcc = occ(i)*a(m,i)*a(n,i)
      b(n,m) = b(n,m) + ocmcc*ev(i)
      b(m,n) = b(m,n) + ocmcc
23014 continue
23015 continue
23012 continue
23013 continue
23008 continue
23009 continue
      do23016 n = 1,ndim 
      do23018 m = 1,ndim 
      if(abs(b(m,n)) .lt. dacc)then
      b(m,n) = 0.0
      endif
23018 continue
23019 continue
23016 continue
23017 continue

      do23022 j = 1,nbeweg 
      indj=ind(j)
      izpj=izp(j)
      do23024 k = 1,nn 
      if(k.ne.j)then
      indk=ind(k)
      izpk=izp(k)
      do23028 i = 1,3 
      xhelp = x(i,j)
      x(i,j) = xhelp + deltax
      call slkmatrices(k,j,x,au,bu)
      x(i,j) = xhelp - deltax
      call slkmatrices(k,j,x,auh,buh)
      x(i,j) = xhelp
      do23030 lj = 1,lmax(izpj) 
      do23032 mj=1,2*lj-1 
      n = (lj-1)**2 + mj
      nu = n + indj
      do23034 lk = 1,lmax(izpk) 
      do23036 mk = 1,2*lk-1 
      m = (lk-1)**2 + mk
      mu = m + indk
      dgrh = (au(m,n)-auh(m,n))*rcdx
      dgrs = -(bu(m,n)-buh(m,n))*rcdx
      if(lldep) then
      if(doscf)then
      dgr = -0.5d0 * dgrs * (shift(k,lk) + shift(j,lj)
     &     -shiftE(k)-shiftE(j)-shiftE2(k)-shiftE2(j)) ! MG+Guanhua_QC_UW1206: KO
      endif
      if(lscc3rd)then
      dgr3=-0.5d0*dgrs/3.0d0* (shift3(k,lk)+shift3a(k,lk)+shift3(j,lj)+s
     *hift3a(j,lj)+shift3b(k)+shift3b(j))
      endif
      else
      if(doscf)then
      dgr = -0.5d0*dgrs*(shift(k,1)+shift(j,1)
     &     -shiftE(k)-shiftE(j)-shiftE2(k)-shiftE2(j)) ! MG+Guanhua_QC_UW1206: KO
      endif
! MG+Guanhua_QC_UW1205, MG_UW1210
      if (lscc3rd) then
        dgr3=-0.5d0*dgrs/3.0d0*(2.0d0*shift3(k,1)+shift3A(k,1)
     &       +2.0d0*shift3(j,1)+shift3A(j,1))
! end
      endif
      endif
! MG_QC_UW1207
      if (lcolspin) then
        dgrspin = -0.5d0*dgrs*(spinshift(k,lk)+spinshift(j,lj))
      endif
! end 
      if(mu.gt.nu)then
      dgrh = dgrh * b(mu,nu)
      dgrs = dgrs * b(nu,mu)
      if(doscf)then
      dgr = dgr * b(mu,nu)
      endif
! Guanhua_QC_UW1101
      if (lscc3rd) then
        dgr3= dgr3* b(mu,nu)
      endif
! end
! MG_QC_UW1207
      if (lcolspin) then   
        dgrspin= dgrspin*b(mu,nu)
      endif
! end
      else
      dgrh = dgrh * b(nu,mu)
      dgrs = dgrs * b(mu,nu)
      if(doscf)then
      dgr = dgr * b(nu,mu)
      endif
! MG+Guanhua_QC_UW1205
      if (lscc3rd) then
        dgr3=dgr3*b(nu,mu)
      endif
! end
! MG_QC_UW1207
      if (lcolspin) then   
        dgrspin= dgrspin*b(nu,mu)
      endif
! end
      endif
      grad(i,j) = grad(i,j) + dgrh + dgrs
      if(doscf)then
      grad(i,j) = grad(i,j) + dgr
      endif
! MG+Guanhua_QC_UW1205
      if (lscc3rd) then
        grad(i,j) = grad(i,j) + dgr3
      endif
! end
! MG_QC_UW1207
      if (lcolspin) then
        grad(i,j) = grad(i,j) + spinfac*dgrspin
      endif
! end

23036 continue
23037 continue
23034 continue
23035 continue
23032 continue
23033 continue
23030 continue
23031 continue
23028 continue
23029 continue
      endif
23024 continue
23025 continue
23022 continue
23023 continue
      end

