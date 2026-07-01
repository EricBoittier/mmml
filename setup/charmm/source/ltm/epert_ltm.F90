module epert_mod
  use chm_kinds
  use dimens_fcm
  use energym
!CHARMM Element source/fcm/epert.fcm 1.1
!========================================================================
!
!     EPERT.FCM stores the energy data structure for PERT.
!
!========================================================================
!  LENENP - Maximum number of energy related properties.
!  LENENT - Maximum number of energy terms.
!  LENENV - Maximum number of pressure/virial terms.
!
! THIS INCLUDE FILE MUST FOLLOW energy.fcm.
!========================================================================
! . Free energy perturbation energy term arrays
!
!    ETPRTM - (1-lambda) energies.
!    ETPRTL - (lambda) energies.
!    ETPRTD - energies differences scaled.
!    ETPRTA - energy term averages.
!    ETPRTF - energy term accumulation.
!    QETPRT - logical array for which energy terms to accumulate.
!
#if KEY_PERT==1 /*epert_fcm*/
       real(chm_real) EPPRTM(LENENP), ETPRTM(LENENT), EVPRTM(LENENV), &
                      EPPRTL(LENENP), ETPRTL(LENENT), EVPRTL(LENENV), &
                      EPPRTD(LENENP), ETPRTD(LENENT), EVPRTD(LENENV), &
                      EPPRTA(LENENP), ETPRTA(LENENT), EVPRTA(LENENV), &
                      EPPRTF(LENENP), ETPRTF(LENENT), EVPRTF(LENENV), &
                      EPPRTT(LENENP), ETPRTT(LENENT), EVPRTT(LENENV)
      logical QEPPRT(LENENP), QETPRT(LENENT), QEVPRT(LENENV)
      ! QC: 11/17 based on Xiya for DFTB PERT
      LOGICAL QMMQM,FIRSTE

#endif /* (epert_fcm)*/
!
end module epert_mod

