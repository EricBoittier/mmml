! Output from Public domain Ratfor, version 1.0
! MG_QC_UW1207: note change of arguments
      subroutine mulliken(nn,x,izp,qmat,qzeroscc,qmulli,ql,qtot,
     * ndim,dacc,occ,a,overl,lmax,ind,dipol,dipabs)

      use sccdftbsrc,only: maxtyp, nndim, mdim
      use sccdftb, only:  qmulik,qmuli2,lmulik 

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

      integer nn,ndim,lmax(maxtyp),izp(nndim),ind(nndim+1)
      real*8 dacc,occ(mdim),a(mdim,mdim),overl(mdim,mdim)
      real*8 x(3,*),qzeroscc(maxtyp,4),qtot,qmulli(mdim)
      real*8 qmat(nndim),dipol(3),dipabs,ql(3*nndim) !MG_QC_UW1207,MG_UW1210
      integer i,j,lj,m,n,izpj,lcount !MG_QC_UW1207
      real*8 sum,qhelp,conv
!     =======================================================
!     QC_UW04: Add mulliken population transfer here
!     real*8 qmulik(nndim)
!     integer mmdim,mxrp
!     parameter(mmdim=100)
!     parameter(mxrp=50)
!     real*8 qmuli2(mmdim,mxrp)
!     logical lmulik
!     common /sccmul/ qmulik,qmuli2,lmulik 
!     =======================================================
!
! added for GHO ... PJ 7/2004
      logical qlink
      common/qlinkl/qlink
!
      do23000 n = 1,ndim 
      qmulli(n) = 0.0
      do23002 i = 1,ndim 
!
! for GHO, we have to loop over all orbitals, becuase
! auxiliary orbitals are also occupied ... PJ 7/2004
!
      if (.not. qlink) then
         if(occ(i) .lt. dacc)then
            goto 23003
         endif
      endif
!
      sum = 0.0
      do23006 m = 1,ndim 
      sum = sum + a(m,i)*overl(m,n)
23006 continue
23007 continue
      qmulli(n) = qmulli(n) + occ(i)*sum*a(n,i)
23002 continue
23003 continue
23000 continue
23001 continue
      lcount=0  !MG_QC_UW1207
      do23008 j = 1,nn 
      qtot = 0.0
      do23010 lj = 1,lmax(izp(j)) 
      jofn = ind(j)+(lj-1)**2
      qhelp = 0.0
      do23012 mj = 1,2*lj-1 
      qhelp = qhelp + qmulli(jofn+mj)
23012 continue
23013 continue
      lcount=lcount+1   !MG_QC_UW1207
      ql(lcount)=qhelp  !MG_QC_UW1207
      qtot = qtot + qhelp
23010 continue
23011 continue
      qmat(j) = qtot
23008 continue
23009 continue
      qtot = 0.0
      do23014 j = 1,nn 
      qtot = qtot+qmat(j)
23014 continue
23015 continue
      do23016 i = 1,3 
      dipol(i) = 0.d0
      do23018 j = 1,nn 
      izpj = izp(j)
      qhelp = qzeroscc(izpj,4) - qmat(j) !MG_UW1210
      dipol(i) = dipol(i) + qhelp*x(i,j)
23018 continue
23019 continue
      dipol(i) = dipol(i)*2.541765d0
23016 continue
23017 continue
      dipabs = 0.d0
      do23020 i = 1,3 
      dipabs = dipabs + dipol(i)**2
23020 continue
23021 continue
      dipabs = sqrt(dipabs)
!     =============================================
!     QC_UW04: Add Muliken
! MG_UW1210: moved to eglcao for compatibility with lcolspin
!      IF (LMULIK) THEN
!        do j=1,nn
!         qmulik(j)=qzeroscc(izp(j),4) - qmat(j)
!!        write(*,'(1x,"MULIK>",I5,2F8.3)') j,qmulik(j),qmat(j)
!        enddo
!      ENDIF

      end

