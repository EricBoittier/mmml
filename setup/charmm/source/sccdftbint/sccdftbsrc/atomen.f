      subroutine atomenerg(nn,ndim,ind,izp,
     & shift,shift3,x,hamil,a,occ,doscf,period,boxsiz,xinvbox,nlat)
CQC1  include 'maxima.inc'
      use sccdftbsrc 
      implicit none
      logical doscf,period
      integer nn,ndim,ind(*),izp(*),nlat(3),nv,nw
      real*8 shift(nndim,3),hamil(maxsiz,maxsiz),shift3(nndim,3) !MG_UW1210
      real*8 a(maxsiz,maxsiz), occ(*),boxsiz(3,3),xinvbox(3,3)
      real*8 x(3,*),repen
CQC1  real*8 qdiff(NNDIM)
CQC1  common /varhelp/ Qdiff
      external repen
      integer l,ls,mu,nu,i
      real*8 eat,rat,r,dif(3),facc

      facc=27.2116
      open (1,file='ATM.DAT',status='unknown')
      rewind 1
      write (1,'(a)') 
     &'#ENERGY CONTRIBUTION OF THE ATOMS(total, electronic, repulsive)'
      write (1,'(a)') '#ENERGY PER ATOM [eV]'
      write (1,'(a)') '#ATOM No.'
      write (1,'(a,2x,i4)') '#',nn
      write (1,'(a,2x,i4)') '#',nn
      do l=1,nn 

       eat=0.0
       rat=0.0
       do i=1,ndim 
        do mu=1,ind(l+1)-ind(l) 
         do ls=1,nn 
          do nu=1,ind(ls+1)-ind(ls) 
           eat = eat + occ(i) * a(ind(l)+mu,i) * a(ind(ls)+nu,i)
     &           * hamil(ind(l)+mu,ind(ls)+nu)
          end do
         end do
        end do
       end do

! MG+Guanhua_QC_UW1205
       if (doscf) then
!old     eat = eat + 0.5*qdiff(l)*shift(l)
         eat = eat + 0.5*qdiff(l)*shift(l,1)+qdiff(l)*shift3(l,1)/3.0 !MG_UW1210
       endif
       
       do ls=1,nn
        
        if(period) then
        
         DO nu =-nlat(1),nlat(1) 
          DO nv = -nlat(2),nlat(2) 
           DO nw = -nlat(3),nlat(3)
            dif(1) = x(1,l)-(x(1,ls)+nu*boxsiz(1,1)+
     &               nv*boxsiz(2,1)+nw*boxsiz(3,1))
            dif(2) = x(2,l)-(x(2,ls)+nu*boxsiz(1,2)+
     &               nv*boxsiz(2,2)+nw*boxsiz(3,2))
            dif(3) = x(3,l)-(x(3,ls)+nu*boxsiz(1,3)+
     &               nv*boxsiz(2,3)+nw*boxsiz(3,3))

            r = sqrt(dif(1)**2 + dif(2)**2 + dif(3)**2)
            rat = rat + 0.5 * repen(r,izp(l),izp(ls))
           enddo
          enddo
         enddo
        else 
         do i=1,3
          dif(i) = x(i,l) - x(i,ls)
         end do
        
         r = sqrt(dif(1)**2+dif(2)**2+dif(3)**2)
        
         rat = rat + 0.5 * repen(r,izp(l),izp(ls))
        endif
       end do
!
! eat contains electronic part, rat repulsive part of energy per atom
!
       eat=eat*facc
       rat=rat*facc
       write (1,'(2x,i4,3(x,f12.6))') l,(eat+rat),eat,rat
       
      end do
      endfile 1
      close (1)
      end

