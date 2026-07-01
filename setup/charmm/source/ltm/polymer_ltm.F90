module polymer
  use chm_kinds
  use dimens_fcm
!CHARMM Element source/fcm/polymer.fcm 1.1
!
!     This files contains the values 
!     VOLUME - total occupied volume
!     FREVOl - free volume
!     FFVOL - fractional free volume
!
!     This file is included by corman2.flx and quanta.flc (qco)
!     The volume data is accesible from QUANTA and passed through the qco code
!
      real(chm_real) VOLUME,FREVOL,FFVOL
!
end module polymer

