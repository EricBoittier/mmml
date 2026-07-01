module qm2_array_locations
  use chm_kinds
  use dimens_fcm
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
!QMMM RIJ OFFSETS
      integer, parameter :: QMMMONERIJ       = 1
      integer, parameter :: QMMMRIJ          = 2
      integer, parameter :: QMMMEXP1         = 3
      integer, parameter :: QMMMEXP2         = 4
      integer, parameter :: QMMMSQRTAEE      = 5
      integer, parameter :: QMMMSQRTRRADDADE = 6
      integer, parameter :: QMMMSQRTRRMDDADE = 7
      integer, parameter :: QMMMSQRTRR2AQE   = 8
      integer, parameter :: QMMMSQRTRRAQQAQE = 9
      integer, parameter :: QMMMSQRTRRMQQAQE = 10
      integer, parameter :: QMMMSQRTRRAQQ2AQE= 11

      integer, parameter :: QMMMNORIJ        = 11

!QMQM RIJ OFFSETS
      integer, parameter :: QMQMRIJBOHRS2  = 1
      integer, parameter :: QMQMONERIJ     = 2
      integer, parameter :: QMQMRIJ        = 3
      integer, parameter :: QMQMRIJBOHRS   = 4
      integer, parameter :: QMQMEXP1I      = 5
      integer, parameter :: QMQMEXP1J      = 6
      integer, parameter :: QMQMSQRTAEE    = 7

      integer, parameter :: QMQMNORIJ      = 7

!Note, these 4 PDDG options should go last as QMQMNORIJ will be
!reduced by 4 if PDDG hamiltonian is not in use.
      integer, parameter :: QMQMPDDG_EXP1  = 8
      integer, parameter :: QMQMPDDG_EXP2  = 9
      integer, parameter :: QMQMPDDG_EXP3  = 10
      integer, parameter :: QMQMPDDG_EXP4  = 11
      integer, parameter :: QMQMNOPDDG     = 4
#endif 
end module qm2_array_locations

