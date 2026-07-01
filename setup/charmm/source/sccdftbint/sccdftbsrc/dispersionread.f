      subroutine dispersionread(nn,x)
CQC1  subroutine dispersionread(nn,ntype,izp,x)
      use sccdftbsrc, A=> Adis, B=>Bdis, C=>Cdis,
     $    r0=> r0dis, rv=>rvdis, izp=>izp2
      implicit real*8(A-H,O-Z) 
CQC1  include 'maxima.inc'
!     QC: DONOT FORGET RESIZE izp
CQC1  integer nei(NNDIM),izp(3*NNDIM)
      integer nei(NNDIM)
CQC1  real*8 Ni0(NTYPE),Ni(NNDIM)
      real*8 Ni0(MAXTYP),Ni(NNDIM)
      real*8 x(3,NNDIM),dif(3),r2
CQC1  real*8 h1(MAXTYP,4),h2(MAXTYP,4),Edis
      real*8 h1(MAXTYP,4),h2(MAXTYP,4)
CQC1  real*8 R0vdw(MAXTYP,4),C6(NNDIM,NNDIM)
      real*8 R0vdw(MAXTYP,4)
CQC1  real*8 Rvdw(NNDIM,NNDIM)
      real*8 hh1(NNDIM),hh2(NNDIM) 
CQC1  real*8 C,A,B,r0,rv,scale
      real*8 scale
CQC1  logical lread,dispers
      logical lread
CQC1  common /disper/ Edis, dispers
CQC1  common /dispertmp/ A,B,C,r0,rv,C6,Rvdw
!     QC: set a crap character to allow things to be read in
      character*2 crap
      lread = .false.
      open(54,file="DISPERSION.INP",status="unknown")
!      read(54,*) A,B,C,rv,r0,scale
!     parameters as in Elstner et al., JCP 2000
      A=7
      B=4
      C=-3.0
      rv=0.0
      r0=0.0
      scale=1.0
!
      if (scale .le. 0.0) then
      write(*,*) 'London type dispersion, pol and dis in eV,scale='
     c ,scale
      else
       write(*,*)  'Slater-Kirkwood dispersion switched on:'
      endif

      do i=1,ntype 
      if ( scale .ge.0.0 ) then
!      read(54,*,end=10) (h1(i,j),j=1,4),(h2(i,j),j=1,4),Ni0(i)
       read(54,*,end=10) crap,(h1(i,j),j=1,4),(h2(i,j),j=1,4),Ni0(i)
       write(*,*) "Para: ",crap,(h1(i,j),j=1,4),(h2(i,j),j=1,4),ni0(i)
       else
!      read(54,*,end=10) (h1(i,j),j=1,4),(h2(i,j),j=1,4)
       read(54,*,end=10) crap,(h1(i,j),j=1,4),(h2(i,j),j=1,4)
       write(*,*) "Para: ",crap,(h1(i,j),j=1,4),(h2(i,j),j=1,4)
       endif
      enddo 
       write(*,*) 
     c '  read C6, Rvdw (eV,A) from DISPERSION.INP'
!     open(15,file='DISP.CHEK')
      write(*,*) '  careful, parameter determined from # of H atoms'

      goto 20
10    continue
       lread = .true.
       open(16,file='DISP.INP')
         write(*,*) '  DISPERSION.INP empty, read from DISP.INP' 
20    continue
! if we read from DISPERSION.INP:
!  determine hybridization from number of Hydrogens
! wir starten mit 1 Nachbarn. Dann zaehlen wir die
! Anzahl der Wasserstoffe durch: fuer jeden Wasserstoff gibt es
! einen Nachbarn mehr. Die werte fuer C und N unterscheiden sich nur in den
! Waserstoffen. N mit 2H ist anders als nur N oder N mit 1H
! C ist nur fuer 3H unterschiedlich
!
!  determine parameters for all atoms
      do i=1,nn
       if (lread)  then
         read(16,*)  hh1(i),hh2(i),Ni(i) 
       else 
         nei(i) = 1
          do j=1,nn
           r2 = 0.d0
           if ( j.ne.i) then
            do n = 1,3
             dif(n) = x(n,i) - x(n,j)
             r2 = r2 + dif(n)*dif(n)
            enddo
            r = dsqrt(r2)*0.52917d0
            if (r.le.1.2d0) then
             nei(i) = nei(i) +1
            endif
           endif
          enddo 
          write(*,*) "Atom: ",i,nei(i),izp(i)
          if (nei(i).gt.4) then
            hh1(i)=h1(izp(i),4)
            hh2(i)=h2(izp(i),4)
          else  
            hh1(i)=h1(izp(i),nei(i))
            hh2(i)=h2(izp(i),nei(i))
          endif
          Ni(i) = Ni0(izp(i))
          write(*,'(3F12.6)')  hh1(i),hh2(i),Ni(i)
       endif
   
!   check values
       if ((hh1(i).eq. 0.0).or.(hh2(i).eq.0.0).or.(Ni(i).eq.0.0)) then
        write(*,*) 'a parameter is 0.0 for atom',i,izp(i),nei(i)
        stop
       endif
!
      enddo !i=1,nn
! set up mixed coefficients 
! mixing from Halgren JACS 1992 114 p7827
       if (.not. lread)  then
       write(*,*) ' --------------'
       write(*,*) ' I J  typeI typeJ C6 R NeiI NeiJ'
       endif
      do i=1,nn
       do j=1,i
        if (scale .le. 0.0) then
         C6(i,j) = -scale*1.5*hh1(i)*hh1(j)*
     c   hh2(i)*hh2(j)/
     c   ( hh2(i)+hh2(j) )
        else
!ccc  17.532 conversion from eV in a.u. for polarizability
!   0.5975 conversion from [H au**6] to [eV A**6]
!  total 17.532*0.5975 = 10.476
         C6(i,j) = scale*1.5*10.476*hh1(i)*hh1(j)/
     c   ( sqrt(hh1(i)/Ni(i) ) + 
     c    sqrt( hh1(j)/Ni(j) ) ) 
        Rvdw(i,j)=(hh2(i)**3 + hh2(j)**3)/
     c    (hh2(i)**2 + hh2(j)**2 )
        Rvdw(j,i) =  Rvdw(i,j)
        endif
        C6(j,i) = C6(i,j) 
       if (.not. lread)  then
        write(*,'(4I4,2F12.6,2I4)') i,j,izp(i),izp(j),
     c   C6(i,j),Rvdw(i,j),nei(i),nei(j)
       endif
       enddo
      enddo
       write(*,*) 'end reading disper'
      end


