!===================================================================
!     calculate gamma or gamma^h-function and Gamma-function for 
!     one atom pairing (gamma^h_{ij} and Gamma_{ij}) 
!     details see Gaus JCTC 2011.
!     
!     INPUT:
!     real*8  r           distance between the two atoms i and j
!     real*8  ui,uj       Hubbard parameters for atom i and j
!     real*8  udi         Hubbard derivative dU/dq for atom i
!     logical xhgammahlp  .true. if h=exp(-((ui+uj)/2^zeta*r^2)) 
!     real*8  zeta        parameter for gamma^h
!    
!     OUTPUT:
!     real*8  gval        value for gamma_{ij} or gamma^h_{ij}
!     real*8  gder        value for Gamma_{ij} = dgamma_{ij}/dq_i
!===================================================================

! MG+Guanhua_QC_UW1205:substantial changes
      subroutine gammaall(r,ui,uj,udi,xhgammahlp,zeta,gval,gder)
      implicit none
      real*8  r,ui,uj,udi,zeta,gval,gder
      logical xhgammahlp
      ! internal variables
      integer i,j
      real*8 zero
      parameter(zero=1.0d-4)
      real*8 a,b,a2,a3,a4,a6,b2,b3,b4,b6
      real*8 s,ar,g,fab,fba,dgda,dsdu,dfabda,dfbada
      real*8 h,dhdu

      a  = 3.2d0*ui
      b  = 3.2d0*uj

      if(a+b .lt.zero) then
        gval = 0.0d0
        gder = 0.0d0
      elseif (r .lt. zero) then
        !!gval = 0.5d0*( a*b/(a+b)+a*a*b*b/((a+b)**3) ) !with a=b: = 5/16*1/2(a+b)
        !gval = 0.15625d0*(a+b)
        !gder = 0.5d0
        ! MG_UW1210: bformal correct deviation becomes necessary for ldep!
        if (dabs(a-b) .lt. zero) then
          gval = 0.15625d0*(a+b)
          gder = 0.5d0
        else  ! necessary for ldep
          gval = 0.5d0*( a*b/(a+b)+a*a*b*b/((a+b)**3) ) !with a=b: gval = 5/16*a = U
          gder = 1.6d0*( b/(a+b) -a*b/(a+b)**2 +2.0d0*a*b*b/(a+b)**3 
     &                   -3.0d0*a*a*b*b/(a+b)**4 )      !with a=b: gder = 1/2 (for easier summation that is wanted!)
        endif
      else ! here 1/r-S
        if (dabs(a-b) .lt. 1.0d-5) then
          ar   = (a+b)/2.0d0*r
          g    = (48.0d0+33.0d0*ar+9.0d0*ar*ar+ar*ar*ar)/(r*48.0d0)
          s    = dexp(-ar)*g

          dgda = (33.0d0+18.0d0*ar+3.0d0*ar*ar)/48.0d0
          dsdu = 3.2d0*dexp(-ar)*(dgda-r*g)
        else  
          a2   = a*a
          a3   = a2*a
          a4   = a2*a2
          a6   = a4*a2
          b2   = b*b
          b3   = b2*b
          b4   = b2*b2     
          b6   = b4*b2     
          fab  = a*b4/(2.0d0*(a2-b2)**2)-(b6-3.0d0*a2*b4)/((a2-b2)**3*r)
          fba  = b*a4/(2.0d0*(b2-a2)**2)-(a6-3.0d0*b2*a4)/((b2-a2)**3*r)
          s    = dexp(-a*r)*fab + dexp(-b*r)*fba

          dfabda = -(b6+3.0d0*a2*b4)/(2.0d0*(a2-b2)**3)
     &                 -12.0d0*a3*b4/(r*(a2-b2)**4)
          dfbada = 2.0d0*b3*a3/((b2-a2)**3) 
     &                 +12.0*a3*b4/(r*(b2-a2)**4)
          dsdu = 3.2d0*(dexp(-a*r)*(dfabda-r*fab)+dexp(-b*r)*dfbada)
        endif
        ! check for gamma^h 
        if(xhgammahlp) then
          h    = dexp(-((a+b)*0.15625d0)**zeta*r*r)
          dhdu = -h*zeta*r*r*((a+b)*0.15625d0)**(zeta-1.0d0)*0.5d0
          gval = 1.0d0/r - s*h
          gder = -(dsdu*h+s*dhdu)
        else
          gval = 1.0d0/r - s
          gder = -dsdu
        endif
      endif ! end 1/r-S
          
      gder = gder*udi
        
      end


!===================================================================
!     calculate dgamma/dr or dgamma^h/dr and dGamma/dr for
!     one atom pairing (gamma^h_{ij} and Gamma_{ij})
!     details see Gaus JCTC 2011.
!     r=|R_j-R_i|
!
!     INPUT:
!     real*8  r           distance between the two atoms i and j
!     real*8  ui,uj       Hubbard parameters for atom i and j
!     real*8  udi         Hubbard derivative dU/dq for atom i
!     logical xhgammahlp  .true. if h=exp(-((ui+uj)/2^zeta*r^2))
!     real*8  zeta        parameter for gamma^h
!
!     OUTPUT:
!     real*8  dcdr        dgamma_{ij}/dr 
!     real*8  dcdr3       dGamma_{ij}/dr = d^2gamma_{ij}/drdq_i
!===================================================================

! MG+Guanhua_QC_UW1205:substantial changes
      subroutine gammaall1(r,ui,uj,udi,xhgammahlp,zeta,dcdr,dcdr3)
      implicit none
      logical xhgammahlp
      real*8  r,ui,uj,udi,zeta,dcdr,dcdr3
      ! internal variables
      real*8  zero
      parameter(zero=1.0d-4)
      real*8  r2,a,b,a2,a3,a4,b2,b3,b4
      real*8  z,z2,zr,ar,g,dgdr,dsdr
      real*8  fab,fba,dfabdr,dfbadr
      real*8  dcdudr,dsdudr,dgdadr,dgda,dfabdadr,dfbadadr,dfabda,dfbada
      real*8  h,dhdu,dhdr,dhdudr,s,dsdu

      r2 = r*r
      a  = 3.2d0*ui
      b  = 3.2d0*uj

      if ((a+b.lt.zero).or.(r.lt.zero)) then
        dcdr = 0.0d0
        dcdr3= 0.0d0
        r    = 99999999.9d0
      else ! here 1/r-s
        if (dabs(a-b) .lt. 1.0d-5) then
          z    = 0.5d0*(a+b)
          z2   = z*z
          zr   = z*r
          g    = (48.0d0+33.0d0*zr+9.0d0*zr*zr+zr*zr*zr)/(48.0d0*r)
          dgdr = -1.0d0/r2+3.0d0*z2/16.0d0+z2*zr/24.0d0
          dsdr = dexp(-zr)*(dgdr-z*g)           
  
          dgda   = (33.0d0+18.0d0*zr+3.0d0*zr*zr)/48.0d0
          dgdadr = 0.375d0*z + 0.125d0*z2*r
          dsdudr = 3.2d0*dexp(-zr)*(g*(zr-1.0d0)-z*dgda+dgdadr-r*dgdr)
          if(xhgammahlp) then
            s    = dexp(-zr)*g
            dsdu = 3.2d0*dexp(-zr)*(dgda-r*g)
          endif
        else
          a2  = a*a
          a3  = a2*a
          a4  = a2*a2
          b2  = b*b
          b3  = b2*b
          b4  = b2*b2
          fab = a*b4/(2.0d0*(a2-b2)**2)-(b4*b2-3.0d0*a2*b4)
     &                                    /((a2-b2)**3*r)
          fba = b*a4/(2.0d0*(b2-a2)**2)-(a4*a2-3.0d0*b2*a4)
     &                                    /((b2-a2)**3*r)
          dfabdr = (b4*b2-3.0d0*a2*b4)/((a2-b2)**3*r2)
          dfbadr = (a4*a2-3.0d0*b2*a4)/((b2-a2)**3*r2)
          dsdr = dexp(-a*r)*(dfabdr-a*fab)+dexp(-b*r)*(dfbadr-b*fba)

          dfabda   = -(b2*b4+3.0d0*a2*b4)/(2.0d0*(a2-b2)**3)
     &               -12.0d0*a3*b4/(r*(a2-b2)**4)
          dfbada   = 2.0d0*b3*a3/((b2-a2)**3)
     &               +12.0d0*a3*b4/(r*(b2-a2)**4)
          dfabdadr =  12.0d0*a3*b4/(r2*(a2-b2)**4)
          dfbadadr = -12.0d0*a3*b4/(r2*(b2-a2)**4)
          dsdudr = 3.2d0*(dexp(-a*r)*(fab*(a*r-1.0d0)-a*dfabda
     &             +dfabdadr-r*dfabdr) +dexp(-b*r)*(dfbadadr-b*dfbada) )
          if(xhgammahlp) then
            s   =dexp(-a*r)*fab + dexp(-b*r)*fba
            dsdu=3.2d0*(dexp(-a*r)*(dfabda-r*fab)+dexp(-b*r)*dfbada)
          endif
        endif 
        ! check for gamma^h 
        if(xhgammahlp) then
          h      = dexp(-((a+b)*0.15625d0)**zeta*r*r)
          dhdu   = -h*zeta*r*r*((a+b)*0.15625d0)**(zeta-1.0d0)*0.5d0
          dhdr   = -h*2.0d0*r*((a+b)*0.15625d0)**zeta
          dhdudr = h*zeta*r*((a+b)*0.15625d0)**(zeta-1.0d0)*
     &             (r*r*((a+b)*0.15625d0)**zeta-1.0d0)
          dcdr   = -1.0d0/r2-(dsdr*h+s*dhdr)
          dcdudr = -(dsdudr*h+dsdu*dhdr+dsdr*dhdu+s*dhdudr)
          dcdr3  = dcdudr * udi
        else
          dcdr   = -1.0d0/r2 - dsdr
          dcdudr = -dsdudr
          dcdr3  = dcdudr * udi
        endif
      endif ! end 1/r-s

      end


