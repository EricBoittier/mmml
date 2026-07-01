module entropy
  use chm_kinds
  use dimens_fcm
! Entropy terms
!     SROT   - rotational 
!     STRAN  - translational
!     SVIB   - vibtrational
!     SSUM   - sum of the components
!     TK     - temperature, K
!     SIGMA  - rotational symmetry number: 
!              C.J.Cramer, Essentials of Comp.Chem.,Wiley,2002,p327
!
! For debugging purpose only
!     Read coordinates and requencies from external file
!     UTEST  - unit for external file
!
! Declarations
      real(chm_real) SROT,STRAN,SVIB,SSUM
      real(chm_real) TK,SIGMA
      INTEGER UTEST
! For entropy standard state
      character(len=4) SSTAN
end module entropy

