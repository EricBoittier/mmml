!==============================================================================
!      evaluate short range expression i.e. sumR (gamma - 1/R)
!      
!      INPUT:
!      REAL*8    rh(3)       vector between rmu and rnu
!      REAL*8    basis(3,3)  basis of cell(unusual!!!:one line is a basis vector)
!      REAL*8    umu         hubbard parameter of orbital mu
!      REAL*8    unu         hubbard parameter of orbital nu
!      REAL*8    udermu      hubbard derivative of orbital mu
!      logical   xhgammahlp  .true. if h=exp(-((ui+uj)/2^zeta*r^2)) for gamma^h
!      REAL*8    zeta        parameter for gamma^h (see Gaus JCTC 2011)
!      REAL*8    tol         convergence tolerance (contribution of last shell)  
!      !!NOTE THAT umu,unu,udermu in this implementation is not orbital but atom specific!!
!
!      OUTPUT:
!      REAL*8    value       value of the short range sum (gamma)
!      REAL*8    valueder    value of the short range sum (Gamma=dgamma/dq)
!==============================================================================

! MG+Guanhua_QC_UW1205
       subroutine SHORTRANGE(rh,basis,umu,unu,udermu,xhgammahlp,
     &                       zeta,tol,value,valueder)
       use sccdftbsrc, only: qsccnb,lscc3rd
       IMPLICIT NONE

       REAL*8  rh(3),umu,unu,udermu,basis(3,3),zeta,tol,value,valueder
       INTEGER i,j,k,nreal,nmax,nmin 
       REAL*8  rvec(3),R(3),result,resultder,lastshell,tmp,norm,lastder
       REAL*8  tolder,gval,gder
       LOGICAL xhgammahlp

!      QC: UW_2004 add sccnb
!       SCC Nonbond 
!       real*8  sccfnb  
!       logical qsccnb,qsccs,qsccsh
!       common /scccut/qsccnb,sccfnb,qsccs,qsccsh
!
!      rh = rmu - rnu
!
! MG+Guanhua_QC_UW1205

       result = 0.0d0
! MG+Guanhua_QC_UW1205
       resultder = 0.0d0       
       nmax = 50
       nmin = 3
       nreal = 0
       lastshell = tol+1.0d-8
! MG_UW1211: insert criteria for 3rd-order convergence
!            since in gammaall gder is always calculated as for DFTB3 
!            we just rise the tolder if DFTB2 is invoked 
       if (lscc3rd) then
         lastder = tol+1.0d-8
         tolder  = tol
       else
         lastder = 0.0d0
         tolder  = tol*1.0d20 
       endif
!      /* sum over R until tolerance is reached */
       DO WHILE ((nreal .le. nmax) .and. ((dabs(lastshell) .gt. tol)
     &      .or.(dabs(lastder).gt.tolder)  .or. (nreal .le. nmin)  )) 
       lastshell = 0.0d0
       lastder   = 0.0d0
        DO i = -nreal,nreal
         DO j = -nreal,nreal
          DO k = -nreal,nreal
!            /*only R belonging to outer shells are new ones */
             IF((nreal .eq. abs(i)) .or. (nreal .eq. abs(j))  .or. 
     &           (nreal.eq. abs(k)) ) THEN
                  
                  R(1)=i*basis(1,1)+j*basis(2,1)+k*basis(3,1) 
                  R(2)=i*basis(1,2)+j*basis(2,2)+k*basis(3,2) 
                  R(3)=i*basis(1,3)+j*basis(2,3)+k*basis(3,3) 

                  rvec(1) = rh(1) - R(1)  
                  rvec(2) = rh(2) - R(2)   
                  rvec(3) = rh(3) - R(3)  
              
                  norm   = dsqrt(rvec(1)**2+rvec(2)**2+rvec(3)**2)

!              get value for gamma or gamma^h and Gamma
! MG+Guanhua_QC_UW1205
               call gammaall(norm,umu,unu,udermu,xhgammahlp,
     &                      zeta,gval,gder)

!              subtract long range part 1/R and multiply by Z(nu)
                           
               IF (norm .lt. 1.0d-6) THEN 
                  tmp =  gval
               ELSE
                  tmp =  ( gval  - 1.0d0/norm )
               ENDIF

               result = result + tmp
! MG+Guanhua_QC_UW1205, changed MG_UW1211
               resultder = resultder + gder
               ! for physical reasonable value gder converges much faster than gval, 
               ! however gval-1/norm converges faster than gder
               ! that is why additional criterium (lastder) is applied.
               lastshell = lastshell + tmp
               lastder   = lastder + gder
              
             END IF
            END DO
           END DO
          END DO
!TMP      QC: UW_04 HERE
          if (qsccnb) go to 777
          nreal = nreal + 1
          END DO

          IF((dabs(lastshell).gt.tol).or.(dabs(lastder).gt.tolder)) THEN
           STOP "tolerance in subroutine short not reached."
          END IF
 777      CONTINUE 
          value = result 
! MG+Guanhua_QC_UW1205
          valueder = resultder

         END




!=============================================================================
! evaluate derivative of short range expression: sumR (d gamma/dR - (-1/R^2))
!      
!      INPUT:
!      REAL*8    rh(3)       vector between rmu and rnu
!      REAL*8    umu         hubbard parameter of orbital mu
!      REAL*8    unu         hubbard parameter of orbital nu
!      REAL*8    udermu      hubbard derivative of orbital mu
!      logical   xhgammahlp  .true. if h=exp(-((ui+uj)/2^zeta*r^2)) for gamma^h
!      REAL*8    zeta        parameter for gamma^h (see Gaus JCTC 2011)
!      REAL*8    basis(3,3)  basis of cell(unusual!!!:one line is a basis vector)
!      REAL*8    tol         convergence tolerance (contribution of last shell)  
!    
!      OUTPUT:
!      REAL*8    deriv(3)    derivative of the short range sum (gamma contr)
!      REAL*8    deriv3(3)   derivative of the short range sum (Gamma contr)
!==============================================================================

       subroutine SHORTRANGE1(rh,basis,umu,unu,udermu,xhgammahlp,zeta,
     &                        tol,deriv,deriv3)
       use sccdftbsrc, only: qsccnb,lscc3rd
       IMPLICIT NONE

       REAL*8 rh(3),umu,unu,udermu,basis(3,3),tol,deriv(3),deriv3(3)
       REAL*8 zeta
       INTEGER i,j,k,nreal,nmax,nmin
       REAL*8 rvec(3),R(3),lastshell,tmp,norm,lastder,tolder
       REAL*8 gdrv,gdrv3
       LOGICAL xhgammahlp

!       QC: UW_2004
!       SCC Nonbond 
!       real*8  sccfnb  
!       logical qsccnb,qsccs,qsccsh
!       common /scccut/qsccnb,sccfnb,qsccs,qsccsh

!
!      rh = rmu - rnu
!
       deriv(1) = 0.0d0
       deriv(2) = 0.0d0
       deriv(3) = 0.0d0
       deriv3(1) = 0.0d0
       deriv3(2) = 0.0d0
       deriv3(3) = 0.0d0
       nmax = 100
       nmin = 3
       nreal = 0
       
       lastshell = tol+1.0d-8
! MG_UW1211: insert criteria for 3rd-order convergence
!            since in gammaall gder is always calculated as for DFTB3 
!            we just rise the tolder if DFTB2 is invoked 
       if (lscc3rd) then
         lastder = tol+1.0d-8
         tolder  = tol
       else
         lastder = 0.0d0
         tolder  = tol*1.0d20 
       endif
!      /* sum over R until tolerance is reached */
       DO WHILE ((nreal .le. nmax) .and. ((dabs(lastshell) .gt. tol)
     &      .or.(dabs(lastder).gt.tolder)  .or. (nreal .le. nmin)  )) 
       lastshell = 0.0d0
       lastder   = 0.0d0
        DO i = -nreal,nreal
         DO j = -nreal,nreal
          DO k = -nreal,nreal
!      /*only R belonging to outer shells are new ones */
       IF((nreal .eq. abs(i)) .or. (nreal .eq. abs(j))  .or. 
     &  (nreal. eq. abs(k)) ) THEN

         R(1)=i*basis(1,1)+j*basis(2,1)+k*basis(3,1) 
         R(2)=i*basis(1,2)+j*basis(2,2)+k*basis(3,2) 
         R(3)=i*basis(1,3)+j*basis(2,3)+k*basis(3,3) 


         rvec(1) = rh(1) - R(1)
         rvec(2) = rh(2) - R(2)
         rvec(3) = rh(3) - R(3)

         norm   = dsqrt(rvec(1)**2+rvec(2)**2+rvec(3)**2)

!       get derivative of gamma and Gamma
! MG+Guanhua_QC_UW1205
        call gammaall1(norm,umu,unu,udermu,xhgammahlp,zeta,gdrv,gdrv3) 

        IF (norm .lt. 1.0d-6) THEN
         tmp = gdrv
        ElSE
         tmp = gdrv - (-1.0d0/(norm**2)) 
        ENDIF
                            
        deriv(1) = deriv(1) + tmp*rvec(1)/norm
        deriv(2) = deriv(2) + tmp*rvec(2)/norm
        deriv(3) = deriv(3) + tmp*rvec(3)/norm
        deriv3(1)= deriv3(1) + gdrv3*rvec(1)/norm
        deriv3(2)= deriv3(2) + gdrv3*rvec(2)/norm
        deriv3(3)= deriv3(3) + gdrv3*rvec(3)/norm

        lastshell = lastshell + tmp
        lastder   = lastder + gdrv3
        END IF
       END DO
       END DO
       END DO
!TMP   QC: UW_04 Skip for cut
       if (qsccnb) return
       nreal = nreal + 1
       END DO
       
       IF((dabs(lastshell) .gt. tol).or.(dabs(lastder).gt.tolder)) THEN
        STOP "tolerance in subroutine short1 not reached."
       END IF

       END

