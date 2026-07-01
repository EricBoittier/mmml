      Subroutine dxlc(igeom,iguess,nndim,nn,qzero,qmat,qaux)
C----------------------------------------------------------------------
C     DXL-BOMD alogirthm (input option vs order )
C                    105   107   109    111     113     115     117 
C     Order           3     4     5       6     7       8       9
C       
C     Guishan Zheng
C     Harvard Univ.
C     05/19/2010
C----------------------------------------------------------------------
C     igeom:   current MD step index
C     iguess:  initial guess algorithm option
C     nndim:   maximal size of molecular orbitals
C     nn:      number of atoms
C     qzero:   atomic charge array
C     qmat:    previous step converged atomic charge array
C     qaux:    auxiliary atomic chareg array
C     coef:    extrapolation coefficient in DXL-BOMD algorithm
C----------------------------------------------------------------------
      Implicit None
      Integer MaxE, igeom, iguess, nndim, nn, IDamp, Korder, i,j
      Parameter(MaxE=14)
      Real*8 qzero(*), qmat(*), qaux(NNDIM,*)
      Real*8 coef(MaxE)
      Real*8 Zero, One, Two, ckappa, alpha
      Data Zero/0.0d0/, One/1.0D0/, Two/2.0D0/ , IDamp/100/
      Save Zero, One, Two, ckappa, coef, Korder

      if(igeom.eq.1) then
         Korder = (iguess - 105)/2 + 3
         Write(*,'(A,X,I2,X,A)') 'DXLC> ',2*Korder-5,
     $      'order DXL-BOMD algorithm is turned on'
         call inicof(coef,ckappa,Korder)
      endif
      if(iguess.gt.100.and.iguess.lt.200) then
         alpha = Zero
         if(igeom .lt. 10)  then
            qaux(1:nn,igeom)  = qmat(1:nn)
         else if(igeom.lt.idamp) then
c   alpha is set to add damping into initial steps that gives smaller
c   energy fluctuation, which may cause small energy drift at initial
c   steps (10-100)
            alpha = 0.1d0
            qaux(1:nn,10) = (Two-alpha)*qmat(1:nn) - 
     $                      (One-alpha)*qaux(1:nn,8) 
         else 
            qaux(1:nn,10) = ckappa*qmat(1:nn) 
            do 30 j = 0, Korder
               qaux(1:nn,10) = qaux(1:nn,10) + coef(j)*qaux(1:nn,9-j)
  30        continue
         endif
      endif
      if(iguess.lt.200.and.iguess.ge.101.and.igeom.ge.10) then
            qaux(1:nn,3)  = qaux(1:nn,4)
            qaux(1:nn,4)  = qaux(1:nn,5)
            qaux(1:nn,5)  = qaux(1:nn,6)
            qaux(1:nn,6)  = qaux(1:nn,7)
            qaux(1:nn,7)  = qaux(1:nn,8)
            qaux(1:nn,8)  = qaux(1:nn,9)
            qaux(1:nn,9)  = qaux(1:nn,10)
            qmat(1:nn)    = qaux(1:nn,10)
      end if
      Return
      End

