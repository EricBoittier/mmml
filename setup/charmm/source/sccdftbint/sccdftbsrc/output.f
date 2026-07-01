! Output from Public domain Ratfor, version 1.0
      subroutine outeigenvectors(a,ev,occ,ind,nn)
      use sccdftbsrc, only: nndim, mdim, izp=>izp2, nel, nbeweg

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

!     integer ind(nndim+1), izp(nndim), nn
      integer ind(nndim+1), nn
      integer i,j,k,l
      integer neig,filled,hfilled
      real*8 a(mdim,mdim), ev(mdim), occ(mdim)
      character*4 orbital(9)
!     common /izpc/ izp, nel, nbeweg
      data orbital/'S   ', 'Px  ', 'Py  ', 'Pz  ', 'Dxx ', 'Dxy ', 'Dyy 
     *', 'Dyz ', 'Dzz '/
      neig = ind(nn+1)
      filled = 0
      hfilled = 0
      do23000 i = 1,neig 
      if(occ(i) .gt. 1)then
      filled = filled+1
      else
      if(occ(i) .gt. 0)then
      hfilled = hfilled+1
      endif
      endif
23000 continue
23001 continue
      open (1,file='EVC.DAT',status='unknown')
      rewind 1
      write (1,'(20x,a)') 'THE ONE-ELECTRON EIGENVALUES AND EIGENVECTORS
     *'
      write (1,*)
      write (1,*)
      write (1,*) 'NUMBER OF FILLED LEVELS: ',filled
      if(hfilled .gt. 0)then
      write (1,*) 'NUMBER OF HALF OCCUPIED STATES: ',hfilled
      endif
      write (1,*)
      write (1,*)
      do23008 i=1,neig
      write(1,13) 'THE ',i,'. EIGENVALUE:',ev(i),' H',ev(i)*27.2116,' eV
     *'
      write(1,*) 'Atom No.   Atom Type'
      do23010 j=1,nn
      write(1,*) '  ',j,'         ',izp(j)
      k=0
      do23012 l=ind(j)+1, ind(j+1)
      k=k+1
      write(1,'(a,f8.5)') orbital(k),a(l,i)
23012 continue
23013 continue
23010 continue
23011 continue
      write(1,*)
23008 continue
23009 continue
      endfile 1
      close (1)
13    format(1x,a,i4,a,f15.8,a,4x,f12.5,a)
      end

! MG_QC_UW1207: subroutine introduced for spin-polarization (lcolspin)
      subroutine outspineigenvectors(a,ev,occ,adown,evdown,occdown,
     &                                ind,nn)
      use sccdftbsrc, only: nndim, mdim, izp=>izp2, nel, nbeweg

      implicit none
      integer ind(nndim+1), nn
      integer i,j,k,l
      integer neig,filledup,filleddown
      real*8 a(mdim,mdim), ev(mdim), occ(mdim)
      real*8 adown(mdim,mdim), evdown(mdim), occdown(mdim)
      character*4 orbital(9)
      data orbital/'S   ', 'Px  ', 'Py  ', 'Pz  ', 'Dxx ', 'Dxy ', 'Dyy 
     *', 'Dyz ', 'Dzz '/
      neig = ind(nn+1)
      filledup = 0
      filleddown = 0
      do i = 1,neig 
        if(occ(i) .gt. 0)     filledup   = filledup+1
        if(occdown(i) .gt. 0) filleddown = filleddown+1
      enddo
      open (1,file='EVC.DAT',status='unknown')
      rewind 1
      write (1,'(20x,a)') 'THE ONE-ELECTRON EIGENVALUES AND EIGENVECTORS
     *'
      write (1,*)
      write (1,*) 
      write (1,*) 'NUMBER OF FILLED LEVELS (spin up): ',filledup
      write (1,*) 'NUMBER OF FILLED LEVELS (spin down): ',filleddown
      write (1,*)
      write (1,*)
      do i=1,neig
        write(1,16) 'THE ',i,'. EIGENVALUE (spin up/spin down):',
     *      ev(i),' H',ev(i)*27.2116,' eV / ',
     *      evdown(i),' H',evdown(i)*27.2116,' eV '
        write(1,*) 'Atom No.   Atom Type'
        do j=1,nn
          write(1,*) '  ',j,'         ',izp(j)
          k=0
          do l=ind(j)+1, ind(j+1)
            k=k+1
            write(1,'(a,f8.5,a,f8.5)') orbital(k),a(l,i),' / ',
     *                                            adown(l,i)
          enddo
        enddo
        write(1,*)
      enddo
      endfile 1
      close (1)
16    format(1x,a,i4,a,f15.8,a,4x,f12.5,a,f15.8,a,4x,f12.5,a)
      end

! MG_QC_UW1207: additional arguments for spin-polarization formalism
      subroutine outspec(nn,nmov,ndim,ind,lmax,izp,ev,occ,efermi,  qmat,
     *qmulli,qtot,dipol,dipabs,
     &                lcolspin,evdown,occdown,qmatup,qmatdown,qmulliup,
     &                qmullidown,qtotup,qtotdown,efermiup,efermidown)

      use sccdftbsrc, only: nndim, mdim, maxtyp
      implicit real*8 (a-h,o-z)
!      integer nndim
!      parameter( nndim= 650)
!      integer maxint
!      parameter( maxint= 160)
!      integer mdim
!      parameter( mdim= 1650)
!      integer maxtyp
!      parameter( maxtyp= 6)
!      integer maxtab
!      parameter( maxtab= 600)
!      integer ldim
!      parameter( ldim= 9)
!      integer maxopt
!      parameter(maxopt=3*nndim)
!      integer maxsiz
!      parameter(maxsiz=mdim)
!      integer maxtp
!      parameter(maxtp=maxtyp)
      integer nn,ndim,ind(nndim+1),lmax(maxtyp),izp(nndim)
      real*8 ev(mdim),occ(mdim),efermi,qmat(nndim),qmulli(mdim)
      real*8 qtot,dipol(3),dipabs
      integer i
      ! MG_QC_UW1207
      logical lcolspin
      real*8 evdown(MDIM),occdown(MDIM),qtotup,qtotdown,qmulliup(MDIM)
      real*8 qmullidown(MDIM),qmatup(NNDIM),qmatdown(NNDIM)

      open (1,file='SPE.DAT',status='unknown')
      rewind 1
      ! MG_QC_UW1207
      if (lcolspin) then
        write(1,*) 'spin up'
        write (1,'(f20.12)') efermiup
      else
        write (1,'(f20.12)') efermi
      endif
      write (1,'(2(1x,f20.12))') (ev(i),occ(i),i = 1,ndim)
      if (lcolspin) then
        write(1,*) 'spin down'
        write (1,'(f20.12)') efermidown
        write (1,'(2(1x,f20.12))') (evdown(i),occdown(i),i = 1,ndim)
      endif
      !write (1dd,'(f20.12)') efermi
      !write (1,'(2(1x,f20.12))') (ev(i),occ(i),i = 1,ndim)
      !end MG_QC_UW1207
      endfile 1
      close (1)
      open (1,file='CHR.DAT',status='unknown')
      rewind 1
      write (1,'(a)') '#MULLIKEN CHARGE'
      write (1,'(a)') '#CHARGE PER ATOM / PER ATOMIC STATES'
      write (1,'(a)') '#ATOM No.'
      write (1,'(a,2x,i4)') '#',nmov
      write (1,'(a,2x,i4)') '#',nn
      ! MG_QC_UW1207
      if (lcolspin) then
        write(1,*) 'spin up'
        do n = 1,nn 
          ind1 = ind(n)+1; ind2 = ind1+lmax(izp(n))**2-1
          write (1,98) n,qmatup(n),(qmulliup(i), i= ind1,ind2)
        enddo
        write(1,*) 'spin down'
        do n = 1,nn 
          ind1 = ind(n)+1; ind2 = ind1+lmax(izp(n))**2-1
          write (1,98) n,qmatdown(n),(qmullidown(i), i= ind1,ind2)
        enddo
      else
      ! end MG_QC_UW1207
      do23014 n = 1,nn 
      ind1 = ind(n)+1
      ind2 = ind1+lmax(izp(n))**2-1
      write (1,98) n,qmat(n),(qmulli(i), i= ind1,ind2)
23014 continue
23015 continue
      endif ! MG_QC_UW1207
      endfile 1
      close (1)
      do23016 n=1,nn 
      write(6,198) n,qmat(n)
23016 continue
23017 continue
      open (1,file='REST.DAT',status='unknown')
      rewind 1
      ! MG_QC_UW1207
      if (lcolspin) then
        write (1,'(a)') 'Total charge, total spin-up charge, total spin-
     &down charge'
        write (1,'(a)') '============'
        write (1,'(3(1x,f20.12))') qtot, qtotup, qtotdown
      else 
      ! end MG_QC_UW1207
      write (1,'(a)') 'Total charge'
      write (1,'(a)') '============'
      write (1,'(1x,f20.12)') qtot
      endif ! MG_QC_UW1207
      write (1,'(a)') ' '
      write (1,'(a)') 'Mulliken dipole moment [Debye]'
      write (1,'(a)') '=============================='
      write (1,'(3(1x,f20.12))') (dipol(i), i= 1,3)
      write (1,'(a)') ' '
      write (1,'(a)') 'Norm of dipole moment [Debye]'
      write (1,'(a)') '============================='
      write (1,'(1x,f20.12)') dipabs
      write (1,'(a)')
      endfile 1
      close (1)
98    format(1x,i4,11(1x,f7.4))
198   format(1x,'Charge: ',i4,f7.4)
      end
      subroutine outcoord(name, nn, period, ntype, izp, x, boxsiz, atnam
     *es)
      use sccdftbsrc, only: nndim
      implicit real*8 (a-h,o-z)
!      integer nndim
!      parameter( nndim= 650)
!      integer maxint
!      parameter( maxint= 160)
!      integer mdim
!      parameter( mdim= 1650)
!      integer maxtyp
!      parameter( maxtyp= 6)
!      integer maxtab
!      parameter( maxtab= 600)
!      integer ldim
!      parameter( ldim= 9)
!      integer maxopt
!      parameter(maxopt=3*nndim)
!      integer maxsiz
!      parameter(maxsiz=mdim)
!      integer maxtp
!      parameter(maxtp=maxtyp)
      character*65 name
      character*70 atnames
      character tmpchr
      logical period
      integer nn, ntype, izp(nndim)
      real*8 x(3,nndim),boxsiz(3,3)
      integer i,j
      real*8 xhlp(3)
      open (1,file=name,status='unknown')
      rewind 1
      if(period)then
      tmpchr='S' 
      else
      tmpchr='C'
      endif
      write(1,'(I4,1X,A1)') nn,tmpchr
      write(1,'(A)') atnames
      do23020 i=1,nn 
      do23022 j=1,3 
      xhlp(j)=x(j,i)*0.529177
23022 continue
23023 continue
      write(1,14) i,izp(i),(xhlp(j),j=1,3)
23020 continue
23021 continue
      if(period)then
      write(1,15) 0.0,0.0,0.0
      do23026 i=1,3 
      do23028 j=1,3 
      xhlp(j)=boxsiz(i,j)*0.529177
23028 continue
23029 continue
      write(1,15)(xhlp(j),j=1,3)
23026 continue
23027 continue
      endif
14    format(i4,1x,i3,3(1x,f12.6))
15    format(3(1x,f12.6))
      endfile 1
      close (1)
      end
      subroutine outforces(nbeweg,izp,gr,xm)
      implicit real*8 (a-h,o-z)
      integer nbeweg,izp(*)
      real*8 gr(*),xm(*)
      integer i,j
      open (2,file='FRC.DAT',status='unknown')
      rewind 2
      write(2,'(A)') '# INTERATOMIC FORCES AT ATOM N0. I'
      write(2,'(A)') '# ATOM N0.'
      write(2,'(A)') '# FORCE_X, FORCE_Y, FORCE_Z'
      write(2,'(A)') '#'
      write(2,'(A)') '#'
      do23030 i = 1,nbeweg 
      write (2,'(i5,3(1x,e19.10))') i,(-gr(3*i+j-3), j = 1,3)
23030 continue
23031 continue
      write(2,'(A)') 'rms/d:   0.0000   0.0000'
      close (2)
      open (2,file='MAS.DAT',status='unknown')
      rewind 2
      write (2,'(1x,i5)') nbeweg
      do23032 i = 1,nbeweg 
      write (2,'(1x,f24.15)') xm(izp(i))
23032 continue
23033 continue
      close (2)
      end
      subroutine outeglcao(nn,nmov,ndim,ind,lmax,izp,ev,occ,efermi,  qma
     *t,qmulli,qtot)
      use sccdftbsrc, only: nndim, mdim, maxtyp
      implicit real*8 (a-h,o-z)
!      integer nndim
!      parameter( nndim= 650)
!      integer maxint
!      parameter( maxint= 160)
!      integer mdim
!      parameter( mdim= 1650)
!      integer maxtyp
!      parameter( maxtyp= 6)
!      integer maxtab
!      parameter( maxtab= 600)
!      integer ldim
!      parameter( ldim= 9)
!      integer maxopt
!      parameter(maxopt=3*nndim)
!      integer maxsiz
!      parameter(maxsiz=mdim)
!      integer maxtp
!      parameter(maxtp=maxtyp)
      integer nn,ndim,ind(nndim+1),lmax(maxtyp),izp(nndim)
      real*8 ev(mdim),occ(mdim),efermi,qmat(nndim),qmulli(mdim)
      real*8 qtot
      integer i
      open (1,file='eglcao.out',status='unknown')
      rewind 1
      write (1,'(3(1x,f20.12))') (ev(i),occ(i),occ(i),i = 1,ndim)
      write (1,'(f20.12)') efermi
      write (1,'(a)') ' '
      write (1,'(a)') 'Mulliken charges'
      write (1,'(a)') '================'
      do23034 n = 1,nn 
      ind1 = ind(n)+1
      ind2 = ind1+lmax(izp(n))**2-1
      write (1,72) 'Atom ',n,': ',qmat(n)
      write (1,73) (qmulli(i), i= ind1,ind2)
23034 continue
23035 continue
      write (1,'(a)') ' '
      write (1,'(a)') 'Total charge'
      write (1,'(a)') '============'
      write (1,'(1x,f20.12)') qtot
      write (1,'(a)') ' '
      endfile 1
      close (1)
72    format(a,i4,a,f20.12)
73    format(3(1x,f20.12))
      end

!     Haibo/QC_UW0609 Added output for Dipole Moment
      subroutine outputmullcoor(fortnumber,nn,x,qmulik)
      use sccdftbsrc, only: nndim
      implicit none
!      integer nndim
!      parameter(nndim= 650)
      integer i,j
      integer nn
      integer fortnumber
      real*8 xhlp(3),qmulik(nndim)
      real*8 x(3,nndim)

      write(fortnumber,'(a)') 'START'
      do 920 i=1,nn
        do 921 j=1,3
           xhlp(j)=x(j,i)*0.529177
  921   continue
        write(fortnumber,922) i,(xhlp(j),j=1,3),qmulik(i)
  920 continue
      write(fortnumber,'(a)') 'END'
  922 format(i4,4(1x,f10.5))
      end


