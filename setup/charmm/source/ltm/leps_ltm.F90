module leps
  use chm_kinds
  use dimens_fcm
!
!     COMMON BLOCK LEPS
!
! mDTM  .Flag added for Gaussian cross-term and SVB surface dimension
!       and whether derivatives are to be computed : SVB_DIM,NDER
!       .mDTM Coupling term between 2 reaction coordinates: VCROSS
!       .Flag added for diatomic SVBs: ?

      integer, save        :: NTA, NTB, NTC, NTD, NTE, NTF, SVB_DIM, NDER
      real(chm_real), save :: DE_1(6),RE_1(6),BE_1(6),SA_1(3), &         
                              DE_2(6),RE_2(6),BE_2(6),SA_2(3), &
                              VCROSS, &
                              XLA(3),XLB(3),XLC(3),XLD(3),XLE(3),XLF(3)
      logical, save        :: QLEPS, QSVB, SVB_AB,SVB_DE, TEST, &        
                              GCROSS, G1CROSS, G2CROSS, SCROSS

!
! for detailed information: refer qmleps.src and routine PREPOT
      integer, save        :: IPTPRT_leps                                
      integer, save        :: NSURF_leps, NDUM_leps(21)                  
      real(chm_real), save :: R_leps(3), ENERGY_leps, DEDR_leps(3)       
      real(chm_real), save :: EASYAB_leps, EASYBC_leps, EASYAC_leps         
      real(chm_real), save :: DE_leps(3), RE_leps(3), BETA_leps(3), Z_leps(3)
      real(chm_real), save :: ZPO_leps(3), OP3Z_leps(3), ZP3_leps(3), &      
                              TZP3_leps(3), TOP3Z_leps(3), DO4Z_leps(3), B_leps(3)
!
end module leps

