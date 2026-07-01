   subroutine MULLIKEN(nn,x,izp,qmat,qzero,qmulli,qtot,ndim,dacc,occ,_
                       a,overl,lmax,ind,dipol,dipabs)
#
# maxima definitions
#
   implicit REAL*8 (A-H,O-Z)
   include 'maxima.h'
#
      integer nn,ndim,lmax(MAXTYP),izp(NNDIM),ind(NNDIM+1)
      real*8 dacc,occ(MDIM),a(MDIM,MDIM),overl(MDIM,MDIM)
      real*8 x(3,*),qzero(MAXTYP),qtot,qmulli(MDIM)
      real*8 qmat(NNDIM),dipol(3),dipabs

      integer i,j,lj,m,n,izpj
      real*8 sum,qhelp,conv

      do n = 1,ndim {
      qmulli(n) = 0.0
      do i = 1,ndim {
        if (occ(i) < dacc) break
        sum = 0.0
        do m = 1,ndim {
          sum = sum + a(m,i)*overl(m,n)
        }
        qmulli(n) = qmulli(n) + occ(i)*sum*a(n,i)
      }
    }
    do j = 1,nn {
      qtot = 0.0; 
      do lj = 1,lmax(izp(j)) {
        jofn = ind(j)+(lj-1)**2; qhelp = 0.0
        do mj = 1,2*lj-1 {
          qhelp = qhelp + qmulli(jofn+mj)
        }
        qtot = qtot + qhelp
      }
      qmat(j) = qtot
    }

# calculation of total charge 
#
    qtot = 0.0
    do j = 1,nn {
# QC: debug
#       write(*,'(1x,"MULLIKEN",I5,F12.6)') j,qmat(j)
        qtot = qtot+qmat(j)
    }

# QC: debug

#   write(*,'(1x,"TOTAL SCC MULLIKEN: ",F12.6)') qtot

# calculation of dipol moment
#
    do i = 1,3 {
      dipol(i) = 0.d0
      do j = 1,nn {
        izpj = izp(j)
        qhelp = qzero(izpj) - qmat(j)
	dipol(i) = dipol(i) + qhelp*x(i,j)
      }
# conversion to debye
      dipol(i) = dipol(i)*2.541765d0
    }

# norm of dipol moment
#
    dipabs = 0.d0
    do i = 1,3 {
      dipabs = dipabs + dipol(i)**2
    }
    dipabs = sqrt(dipabs)

end

