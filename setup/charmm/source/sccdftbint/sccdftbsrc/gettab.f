! Output from Public domain Ratfor, version 1.0
!     subroutine gettab(ntype)
      subroutine gettab
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
!     real*8 skhtab(10,maxtab,maxtyp,maxtyp),skstab(10,maxtab,maxtyp,max
!    *typ)
!     real*8 espin(maxtyp)
!     real*8 skself(3,maxtyp),dr(maxtyp,maxtyp),qzeroh(3),uhubbh(3)
!MG_UW1210      real*8                                    qzeroh(3),uhubbh(3)
!     real*8 coeff(6,maxint,maxtyp,maxtyp),xr(2,maxint,maxtyp,maxtyp)
!     real*8 efkt(3,maxtyp,maxtyp),cutoff(maxtyp,maxtyp)
!     integer numint(maxtyp,maxtyp)
!     integer dim(maxtyp,maxtyp),ppp,lmax(maxtyp)
      integer                    ppp
      character*64 skfile
      character*128 chdummy
!MG_UW1210      integer lcount  ! MG_QC_UW1207
!     common /sktab/ skhtab,skstab,skself,dr,dim
!     common /spin/ espin
!     QC: UW_06 Haibo adds Gaussian bump as SRP
!     real*8 gbump(3,maxtyp,maxtyp),ronsw(maxtyp,maxtyp)
!     real*8 deltar(maxtyp,maxtyp)
!     logical lgausw

!     common /spltab/ coeff,xr,efkt,cutoff,numint,gbump,ronsw,deltar,
!    $       lgausw

!     common /mcharge/ qzeroscc(maxtyp), uhubb(maxtyp)
!     common /lmax/ lmax

!MG_UW1210      lcount=0
      do23000 i = 1,ntype 
      qzeroscc(i,4) = 0.0d0
      do k=1,3
        qzeroscc(i,k) = 0.0d0
        uhubb(i,k)    = 0.0d0
      enddo 
      do23002 j = 1,ntype 
!      print *,'enter file name for sk-data of pair ',i,j
      read (4,*) skfile
      open (3,file=skfile,status='unknown')
      rewind 3
      if(i .ne. j)then
      read (3,*) dr(i,j),dim(i,j)
      endif
      if(i .eq. j)then
      read (3,*) dr(i,j),dim(i,j),lmax(i)
      read (3,*) (skself(l,i),l = 1,3),espin(i), (uhubb(i,4-l), l=1,3),(
     *qzeroscc(i,4-l), l=1,3)
      !if(uhubb(i).eq.0.4195007d0)then
      !write(*,*) '+++modified version of gamma used of X-H+++'
      !endif
      do23010 k = 1,3 
      qzeroscc(i,4) = qzeroscc(i,4) + qzeroscc(i,k)
23010 continue
23011 continue
!MG_UW1210      ! MG_QC_UW1207: initialize l-dependent charges
!      do l=1,lmax(i)
!        lcount=lcount+1
!        qlzero(lcount)=qzeroh(l)
!      enddo
      ! end MG_QC_UW1207
      do23012 l=1,3 
      espin(i) = espin(i) + skself(l,i)*qzeroscc(i,4-l)
23012 continue
23013 continue
      endif
      do23014 k = 1,dim(i,j) 
      read (3,*) (skhtab(l,k,i,j),l = 1,10), (skstab(l,k,i,j),l = 1,10)
23014 continue
23015 continue
23016 if(.true.)then
      read(3,'(A)') chdummy
      if(chdummy.eq.'Spline')then
      goto 23017
      endif
      goto 23016
      endif
23017 continue
      read(3,*) numint(i,j),cutoff(i,j)
      if(numint(i,j).gt.maxint)then
      print *,'Too many intervalls!'
      goto 99
      endif
      read(3,*) (efkt(ppp,i,j),ppp=1,3)
      do23022 ppp=1,numint(i,j) 
      if(ppp.lt.numint(i,j))then
      read (3,*) xr(1,ppp,i,j),xr(2,ppp,i,j),coeff(1,ppp,i,j),coeff(2,pp
     *p,i,j),coeff(3,ppp,i,j),coeff(4,ppp,i,j)
      else
      read (3,*) xr(1,ppp,i,j),xr(2,ppp,i,j),coeff(1,ppp,i,j),coeff(2,pp
     *p,i,j),coeff(3,ppp,i,j),coeff(4,ppp,i,j),coeff(5,ppp,i,j),coeff(6,
     *ppp,i,j)
      endif
23022 continue
23023 continue
      if(xr(2,numint(i,j),i,j).ne.cutoff(i,j))then
      print *,'Error in Datafile'
      goto 99
      endif
!     QC_Haibo: UW_06 Adds SRP 
! get the parameters for the gaussian bump
!     Haibo Yu: If read in additional switched gaussian for repulsion
      if (lgausw) then
        read (3,*) gbump(1,i,j)
        read (3,*) gbump(2,i,j)
        read (3,*) gbump(3,i,j)
        read (3,*) ronsw(i,j),deltar(i,j)
      endif   

      print *,'skfile for pair ',i,j,' :',skfile
      close (3)
23002 continue
23003 continue
23000 continue
23001 continue
99    continue
      end

