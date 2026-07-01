module neb
  use chm_kinds
  use dimens_fcm
#if KEY_RPATH==1

  !  NPATOMS
  !  IREPS

  integer :: NPATOMS,ICLREP
  integer,dimension(100) :: IREPS,IPVAR,NPVAR
  real(chm_real),dimension(100) :: EREPS,GREPS,GPREPS,GPSREPS 

#endif 

end module neb

