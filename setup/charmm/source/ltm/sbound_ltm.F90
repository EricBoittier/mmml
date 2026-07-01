module sbound
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/sbound.fcm 1.1
#if KEY_NOMISC==0 /*sbound_fcm*/
  !  SBOUND.FCM
  !
  !  this is the INCLUDE block for solvent boundaries
  !
  !        QBOUND = logical flag indicating that boundary potential
  !                 and forces are to be calculated
  !        IBOUND = integer flag indicating what is the shape of the
  !                 boundary potential (sphere=1,cylinder=2,plane=3)
  !        SSXREF,SSYREF,SSZREF = spheres origin or origin of the cylinder
  !                 axis (no longer used for Langevin dynamics update LNGFIL)
  !        SXREF,SYREF,SZREF = sphere reference origin for Langevin buffer
  !        SXDIR,SYDIR,SZDIR is the cylinder unit vector axis
  !
  !        NCTABL number of tables
  !        SBR(NMFTAB,*) radius vector for table "*"
  !        APOT(NMFTAB,*) potential vector for table "*"
  !        CSPLIN(NMFTAB,1,*), CSPLIN(NMFTAB,2,*), CSPLIN(NMFTAB,3,*)
  !                        cubic spline coefficients for table "*"
  !        NCATOM(*)  number of atoms referring to table "*"
  !        CATOM(1,*),...,CATOM(NCATOM,*) atom numbers referring to table "*"
  !
  !  NMFTAB - Maximum number of distance lookup points for any table.
  !  NMCTAB - Maximum number of boundary tables.
  !  NMCATM - Maximum number of atoms for any boundary table.
  !  NSPLIN - Order of fit (3=cubic spline).
  !
  LOGICAL QBOUND
  real(chm_real)  SXREF, SYREF, SZREF, SXDIR, SYDIR, SZDIR, BSURF
  real(chm_real)  SSXREF, SSYREF, SSZREF
  INTEGER NR,IBOUND
  real(chm_real)  SBR(NMFTAB,NMCTAB),APOT(NMFTAB,NMCTAB), &
       CSPLIN(NMFTAB,NSPLIN,NMCTAB), BFORC
  INTEGER NCTABL, NCATOM(NMCTAB),CATOM(NMCATM,NMCTAB)
  !
#else /* (sbound_fcm)  NOMISC*/
  real(chm_real) SSXREF,SSYREF,SSZREF, SXREF,SYREF,SZREF
  PARAMETER (SSXREF=ZERO,SSYREF=ZERO,SSZREF=ZERO)
#endif /* (sbound_fcm)  NOMISC*/
  !
contains

  subroutine sbound_init()
#if KEY_NOMISC==0 /*sbound_init*/
    qbound=.false.
    sxref=0.0
    syref=0.0
    szref=0.0
    nctabl=0
#endif /* (sbound_init)*/
    return
  end subroutine sbound_init
end module sbound

