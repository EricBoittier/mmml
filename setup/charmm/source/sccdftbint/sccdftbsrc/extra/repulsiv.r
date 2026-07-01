subroutine repulsive(nn,izp,period,x,boxsiz,xinvbox,nlat,erep)
#
implicit REAL*8 (A-H,O-Z)
include 'maxima.h'
#
real*8 x(3,*),boxsiz(3,3),xinvbox(3,3),erep
integer nn,izp(NNDIM), nlat(3),nu,nv,nw
logical period

integer i,j,k,izpj,izpk
real*8 dif(3),r,r2

if(period) {
 do j = 1,nn {
  izpj = izp(j)
#
# Avoid double counting
#
  do k = j+1,nn {
   izpk = izp(k)
    DO nu =-nlat(1),nlat(1) {
     DO nv = -nlat(2),nlat(2) {
      DO nw = -nlat(3),nlat(3) {
       dif(1) = x(1,j)-(x(1,k)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,1))
       dif(2) = x(2,j)-(x(2,k)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,2))
       dif(3) = x(3,j)-(x(3,k)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,3))

       r = sqrt(dif(1)**2 + dif(2)**2 + dif(3)**2)
       erep = erep + repen(r,izpj,izpk)
      }
     }
    }
   }
  }
 
}
else {
 do j = 1,nn {
   izpj = izp(j)
#
# Avoid double counting
#
   do k = j+1,nn {
     izpk = izp(k)
 
     r2 = 0.0
     do i = 1,3 {
       dif(i) = x(i,k) - x(i,j)
     }
    
     r2 = dif(1)**2 + dif(2)**2 + dif(3)**2
     r = sqrt(r2)

     erep = erep + repen(r,izpj,izpk)
   }
 }
}
end

subroutine repulsivegrd(nn,nbeweg,izp,period,x,boxsiz,xinvbox,nlat,grd)
#
implicit REAL*8 (A-H,O-Z)
include 'maxima.h'
#
real*8 x(3,*),boxsiz(3,3),xinvbox(3,3),grd(3,*)
integer nn,nbeweg,izp(NNDIM),nlat(3),nu,nv,nw
logical period

integer i,j,k,izpj,izpk
real*8 dif(3),dgr,grdr,r,r2

do j = 1,nn {
  do i = 1,3 {
    grd(i,j) = 0.0
  }
}

if(period) { 
  do j = 1,nn {
   izpj = izp(j)
#
# Calculate the repulsive part of the force on atom k
#  F_k  = d E_rep / d R_k = sum_j d E_rep(j,k) / d R_k
#
   do k = 1,nn {
     izpk = izp(k)
     if(k<=nbeweg) {
      DO nu =-nlat(1),nlat(1) {
       DO nv = -nlat(2),nlat(2) {
        DO nw = -nlat(3),nlat(3) {
         dif(1) = x(1,k)-(x(1,j)+nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,1))
         dif(2) = x(2,k)-(x(2,j)+nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,2))
         dif(3) = x(3,k)-(x(3,j)+nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,3))    
    
         r = sqrt(dif(1)**2 + dif(2)**2 + dif(3)**2)

#
# Only for moveable atoms (sum over d E_rep(k,k)/ d R_k, E_rep(k,k)=0 )
#
        if (r >= 1.0d-2) {  
         grdr = grdrep(r,izpj,izpk)/r     
         do i = 1,3 {
           dgr = dif(i)*grdr
           grd(i,k) = grd(i,k) + dgr
         }
        }
       }
      }
     }
    }
   }
 }
}
else {
 do j = 1,nn {
   izpj = izp(j)
#
# Calculate the repulsive part of the force on atom k
#  F_k  = d E_rep / d R_k = sum_j d E_rep(j,k) / d R_k
#
   do k = 1,nn {
     izpk = izp(k)
    
     r2 = 0.0
     do i = 1,3 {
       dif(i) = x(i,k) - x(i,j)
     }
    

     r2 = dif(1)**2 + dif(2)**2 + dif(3)**2
     r = sqrt(r2)
#
# Only for moveable atoms (sum over d E_rep(k,k)/ d R_k, E_rep(k,k)=0 )
#
     if (k <= nbeweg && r >= 1.0d-2) {  
       grdr = grdrep(r,izpj,izpk)/r     
       do i = 1,3 {
         dgr = dif(i)*grdr
         grd(i,k) = grd(i,k) + dgr
       }
     }
    
   }
 }
}

end

real*8 function repen(r,izpj,izpk)
#
include 'maxima.h'
#
     real*8 r
     integer izpj,izpk
     real*8 fhelp,xh,xv1
     integer i,j

     real*8 coeff(6,MAXINT,MAXTYP,MAXTYP),xr(2,MAXINT,MAXTYP,MAXTYP)
     real*8 efkt(3,MAXTYP,MAXTYP),cutoff(MAXTYP,MAXTYP)
     integer numint(MAXTYP,MAXTYP)
common /spltab/ coeff,xr,efkt,cutoff,numint
#              spline data for repulsiv force
     
     if(r<1.0d-2) fhelp=0.0
     else if(r<xr(1,1,izpj,izpk)) fhelp=exp(-efkt(1,izpj,izpk)*r+ _ 
                                        efkt(2,izpj,izpk))+efkt(3,izpj,izpk)
     else if (r>cutoff(izpj,izpk)) fhelp=0.0
     else {
       do i=1,numint(izpj,izpk) {
         if(r>=xr(1,i,izpj,izpk) && r<=xr(2,i,izpj,izpk)) break
       }
       xv1=r-xr(1,i,izpj,izpk)
       fhelp=coeff(1,i,izpj,izpk)
       xh=xv1
       if(i<numint(izpj,izpk)) {
         do j=2,4 {
           fhelp=fhelp+coeff(j,i,izpj,izpk)*xh
           xh=xh*xv1
         }
       }
       else {
         do j=2,6 {
           fhelp=fhelp+coeff(j,i,izpj,izpk)*xh
           xh=xh*xv1
         }
       }
     }
     repen=fhelp
end

real*8 function grdrep(r,izpj,izpk)
#
include 'maxima.h'
#
        real*8 r
        integer izpj,izpk
        real*8 grdr,xv1,xh
        integer i,j

        real*8 coeff(6,MAXINT,MAXTYP,MAXTYP),xr(2,MAXINT,MAXTYP,MAXTYP)
        real*8 efkt(3,MAXTYP,MAXTYP),cutoff(MAXTYP,MAXTYP)
        integer numint(MAXTYP,MAXTYP)
common /spltab/ coeff,xr,efkt,cutoff,numint
#              spline data for repulsiv force

        grdr = 0.0      
        if(r<1.0d-2) grdr=0.0
        else if(r<xr(1,1,izpj,izpk)) grdr=-efkt(1,izpj,izpk)* _
                                     exp(-efkt(1,izpj,izpk)*r+efkt(2,izpj,izpk))
        else if (r>cutoff(izpj,izpk)) grdr=0.0
        else {
          do i=1,numint(izpj,izpk) {
            if(r>=xr(1,i,izpj,izpk) && r<=xr(2,i,izpj,izpk)) break
          }
          xv1=r-xr(1,i,izpj,izpk)
          xh=1
          if(i<numint(izpj,izpk)) {
            do j=2,4 {
              grdr=grdr+(j-1)*coeff(j,i,izpj,izpk)*xh
              xh=xh*xv1
            }
          }
          else {
            do j=2,6 {
              grdr=grdr+(j-1)*coeff(j,i,izpj,izpk)*xh
              xh=xh*xv1
            }
          }
        }
        grdrep=grdr;
end

