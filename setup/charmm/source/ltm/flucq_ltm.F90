module flucq
  use chm_kinds
  use dimens_fcm
!
#if KEY_FLUCQ==0 /*flucq_fcm*/
      logical,parameter :: QFLUC = .false.
#else /* (flucq_fcm)*/
!
!     FLUCQ COMMON FILE
!     defines variables for FlucQ
!
!     FQTSTP         Timestep for FlucQ dynamics (ps)
!     FQTEMP         Charge temperature (K)
!     FQTCOU         Thermostatting coupling constant
!     FQREFE         Reference energy (KCal/mol)
!     FQBNDE         Contribution to energy from bond terms (unused)
!     FQNHS          Current Nose-Hoover 'S' value
!     FQNHSO         Nose-Hoover 'S' value from previous timestep
!     FQNHSN         New Nose-Hoover 'S' value at this timestep
!     FQNHM          Mass for Nose-Hoover thermostatting
!     FQCMV2         Current sum of charge mass * velocity**2 for KE calc
!     FQNHTL         Tolerance for Nose-Hoover iterations
!     FQSELN         Atom selection for FlucQ (allocated)
!     FQJR           Calculated J(R) intramolecular terms (allocate)
!     FQCFOR         Pointer to charge force array (allocate)
!     FQNHMX         Maximum number of Nose-Hoover iterations
!     FQCDGF         Number of charge degrees of freedom
!     FQOLDQ         Pointer to charges at last timestep
!     FQNEWQ         Pointer to new calculated charges at this timestep
!     FQUNIT         Unit to write thermostatting data to
!     FQSCAL         Frequency to scale charge velocities during dynamics
!     QFLUC          Flag: is FlucQ active?
!     FQGRUP         Set if charge is to be conserved within groups,
!                       not residues
!     FQFIXD         Set if bond lengths are taken to be fixed (i.e. do not
!                       recalculate intramolecular interaction terms)
!     FQNUCD         Set if QM/MM nucleus-point charge calculation is complete
!
      real(chm_real),allocatable,dimension(:) :: FQOLDQ
      real(chm_real),allocatable,dimension(:) :: FQNEWQ
      real(chm_real) FQTSTP,FQTEMP,FQTCOU,FQREFE,FQBNDE,FQNHS
      real(chm_real) FQNHSO,FQNHSN,FQNHM,FQCMV2,FQNHTL
      INTEGER FQNHMX,FQCDGF,FQUNIT,FQSCAL
      LOGICAL QFLUC,FQGRUP,FQFIXD,FQNUCD
contains
  subroutine flucq_init()
    qfluc=.false.
    fqscal=-1
    return
  end subroutine flucq_init

#endif /* (flucq_fcm)*/
!
end module flucq

