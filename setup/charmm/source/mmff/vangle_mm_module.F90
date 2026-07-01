module vangle_mm
  use chm_kinds
  implicit none
#if KEY_MMFF==1 /*vanglefcm*/
  !
  !     Vector stretch-bend and out-of-plane energy/force pointers.
  !     
  INTEGER NSTRBV
  INTEGER,pointer,dimension(:) :: ISBV,JSBV,KSBV,LSBV,vindsb
  real(chm_real),pointer,dimension(:) :: VFK,VBEQ,VTEQ
  !
  INTEGER NOOPLV
  INTEGER,pointer,dimension(:) :: IOPV,JOPV,KOPV,LOPV,VINDOP
  real(chm_real),pointer,dimension(:) :: vOOPLFC
#endif /* (vanglefcm)*/


!-----------------------------------------------------------------------
!     Vector torsion energy/force pointers.
!     
!     Jay Banks 13-DEC-95: added TorFC1, TorFC2, TorFC3 for MMFF.
!     Tim Miller 23-DEC-08: modularize
!

  real(chm_real),pointer,dimension(:) :: vcpc,vcpb,vcpcos,vcpsin
  INTEGER NPHIV
  INTEGER,pointer,dimension(:) :: VCPD
  INTEGER,ALLOCATABLE,DIMENSION(:) :: IPV,JPV,KPV,LPV,VIND
  real(chm_real),pointer,dimension(:) :: TorFC1,TorFC2,TorFC3

contains

  SUBROUTINE PTRINI
    implicit none
    !
    !  Initialize pointers.
    !
    !  Author: Stephen Fleischman 11/91
    !
    ! . Fast nonbond list generator (machdep.f90):
    !  CALL INIPTA(MAXCPU,JNBT)
    !
    ! . Fast torsion arrays (vphi.f90):
    nullify(VCPC)
    nullify(VCPD)
    nullify(VCPB)
    nullify(VCPCOS)
    nullify(VCPSIN)
#if KEY_MMFF==1
    nullify(TorFC1)
    nullify(TorFC2)
    nullify(TorFC3)
    ! . Fast stretch-bend and out-of-plane arrays (vangle_mm.f90):
    nullify(ISBV)
    nullify(JSBV)
    nullify(KSBV)
    nullify(LSBV)
    nullify(VINDSB)
    nullify(VFK)
    nullify(VBEQ)
    nullify(VTEQ)

    nullify(IOPV)
    nullify(JOPV)
    nullify(KOPV)
    nullify(LOPV)
    nullify(VINDOP)
    nullify(VOOPLFC)
#endif 

    RETURN
  END SUBROUTINE PTRINI

  SUBROUTINE PTRFIN
    use memory
    implicit none
    !
    !  Free pointers.
    !
    !  Author: Stephen Fleischman 11/91 (help from BRB)
    !
    ! . Fast nonbond list generator (machdep.f90):
    !  CALL FREPTA(MAXCPU,JNBT)
    !
    ! . Fast torsion arrays (vphi.f90):

    IF(ALLOCATED(IPV)) THEN
       CALL CHMDEALLOC('iniall.src','PTRFIN','IPV',siz1=SIZE(IPV),intg=IPV)
    ENDIF
    IF(ALLOCATED(JPV)) THEN
       CALL CHMDEALLOC('iniall.src','PTRFIN','JPV',siz1=SIZE(JPV),intg=JPV)
    ENDIF
    IF(ALLOCATED(KPV)) THEN
       CALL CHMDEALLOC('iniall.src','PTRFIN','KPV',siz1=SIZE(KPV),intg=KPV)
    ENDIF
    IF(ALLOCATED(LPV)) THEN
       CALL CHMDEALLOC('iniall.src','PTRFIN','LPV',siz1=SIZE(LPV),intg=LPV)
    ENDIF
    IF(ALLOCATED(VIND)) THEN
       CALL CHMDEALLOC('iniall.src','PTRFIN','VIND',siz1=SIZE(VIND),intg=VIND)
    ENDIF
    if (associated(VCPC)) then
       call chmdealloc('iniall.src','PTRFIN','VCPC',size(VCPC),crlp=VCPC)
    endif
    if (associated(VCPD)) then
       call chmdealloc('iniall.src','PTRFIN','VCPD',size(VCPD),intgp=VCPD)
    endif
    if (associated(VCPB)) then
       call chmdealloc('iniall.src','PTRFIN','VCPB',size(VCPB),crlp=VCPB)
    endif
    if (associated(VCPCOS)) then
       call chmdealloc('iniall.src','PTRFIN','VCPCOS',size(VCPCOS),crlp=VCPCOS)
    endif
    if (associated(VCPSIN)) then
       call chmdealloc('iniall.src','PTRFIN','VCPSIN',size(VCPSIN),crlp=VCPSIN)
    endif

    RETURN
  END SUBROUTINE PTRFIN

#if KEY_MMFF==1 /*vanglefcm*/
  SUBROUTINE MAKNGL_MM
    use chm_kinds
    use dimens_fcm
    use number
    use consta
    use psf
    use param
    use mmffm
    use stream
    use code
    use memory
    ! local
    INTEGER ISTRB,IC,ND,IOOPL
    !
    !  Generate arrays for MMFF fast vector estrb and eoopl routines.
    !   Adapted for MMFF by Jay Banks, 14-DEC-95, from MAKPHI by
    !  Stephen Fleischman
    !
    !
    !     determine memory requirement for stretch-bends
    NSTRBV=0
    DO ISTRB = 1,NTHETA
       DO ND=1,2
          IC=ICSTBN(ISTRB)
          IF (IC.NE.0) THEN
             NSTRBV = NSTRBV+1
          ENDIF
       ENDDO
    ENDDO
    !     allocate memory for the vector strb arrays
    IF (associated(ISBV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','ISBV',NSTRBV,intgp=ISBV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','ISBV',NSTRBV,intgp=ISBV)
    ENDIF
    IF (associated(JSBV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','JSBV',NSTRBV,intgp=JSBV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','JSBV',NSTRBV,intgp=JSBV)
    ENDIF
    IF (associated(KSBV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','KSBV',NSTRBV,intgp=KSBV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','KSBV',NSTRBV,intgp=KSBV)
    ENDIF
    IF (associated(LSBV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','LSBV',NSTRBV,intgp=LSBV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','LSBV',NSTRBV,intgp=LSBV)
    ENDIF
    IF (associated(VINDSB)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VINDSB',NSTRBV,intgp=VINDSB)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VINDSB',NSTRBV,intgp=VINDSB)
    ENDIF
    IF (associated(VFK)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VFK',NSTRBV,crlp=VFK)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VFK',NSTRBV,crlp=VFK)
    ENDIF
    IF (associated(VBEQ)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VBEQ',NSTRBV,crlp=VBEQ)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VBEQ',NSTRBV,crlp=VBEQ)
    ENDIF
    IF (associated(VTEQ)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VTEQ',NSTRBV,crlp=VTEQ)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VTEQ',NSTRBV,crlp=VTEQ)
    ENDIF

    CALL MAKSTRB(NSTRBV, &
         ISBV,JSBV,KSBV,LSBV, &
         VINDSB,VFK,VBEQ,VTEQ)
    !
    ! now do the same for out-of-planes
    !     determine memory requirement
    NOOPLV=0
    DO IOOPL = 1,NTHETA
       IC=ICOOP(IOOPL)
       IF (IC.GT.0) THEN
          NOOPLV = NOOPLV+1
       ENDIF
    ENDDO
    !     allocate memory for the vector oopl arrays
    IF (associated(IOPV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','IOPV',NOOPLV,intgp=IOPV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','IOPV',NOOPLV,intgp=IOPV)
    ENDIF
    IF (associated(JOPV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','JOPV',NOOPLV,intgp=JOPV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','JOPV',NOOPLV,intgp=JOPV)
    ENDIF
    IF (associated(KOPV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','KOPV',NOOPLV,intgp=KOPV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','KOPV',NOOPLV,intgp=KOPV)
    ENDIF
    IF (associated(LOPV)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','LOPV',NOOPLV,intgp=LOPV)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','LOPV',NOOPLV,intgp=LOPV)
    ENDIF
    IF (associated(VINDOP)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VINDOP',NOOPLV,intgp=VINDOP)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VINDOP',NOOPLV,intgp=VINDOP)
    ENDIF
    IF (associated(VOOPLFC)) THEN
       call chmrealloc('vangle_mm_module.src','MAKNGL_MM','VOOPLFC',NOOPLV,crlp=VOOPLFC)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKNGL_MM','VOOPLFC',NOOPLV,crlp=VOOPLFC)
    ENDIF

    CALL MAKOOPL(NOOPLV, &
         IOPV,JOPV,KOPV,LOPV, &
         VINDOP,VOOPLFC)
    !
    RETURN
  END SUBROUTINE MAKNGL_MM
  !
  SUBROUTINE MAKPHI_MM
    use chm_kinds
    use dimens_fcm
    use number
    use consta
    use psf
    use param
    use stream
    use code
    use memory
    implicit none
    ! local
    INTEGER IPHI,IC,SZ
    !
    !  Generate arrays for fast vector ephi routines.
    !  Author: Stephen Fleischman
    !
    !  Trigonometric tables introduced to support
    !   the new dihedral energy routines.
    !
    !   by: Arnaud Blondel
    !
    !   Adapted for MMFF by Jay Banks, 13-DEC-95
    !
    !     determine memory requirement
    NPHIV=0
    DO IPHI = 1,NPHI
       IC=ICP(IPHI)
       IF (IC.NE.0) THEN
          NPHIV = NPHIV+1
       ENDIF
    ENDDO
    !     allocate memory for the vector phi arrays
    SZ=NPHIV
    IF(ALLOCATED(IPV)) THEN
       CALL CHMREALLOC('vangle_mm_module.src','MAKPHI_MM','IPV',SZ,intg=IPV)
    ELSE
       CALL CHMALLOC('vangle_mm_module.src','MAKPHI_MM','IPV',SZ,intg=IPV)
    ENDIF
    IF(ALLOCATED(JPV)) THEN
       CALL CHMREALLOC('vangle_mm_module.src','MAKPHI_MM','JPV',SZ,intg=JPV)
    ELSE
       CALL CHMALLOC('vangle_mm_module.src','MAKPHI_MM','JPV',SZ,intg=JPV)
    ENDIF
    IF(ALLOCATED(KPV)) THEN
       CALL CHMREALLOC('vangle_mm_module.src','MAKPHI_MM','KPV',SZ,intg=KPV)
    ELSE
       CALL CHMALLOC('vangle_mm_module.src','MAKPHI_MM','KPV',SZ,intg=KPV)
    ENDIF
    IF(ALLOCATED(LPV)) THEN
       CALL CHMREALLOC('vangle_mm_module.src','MAKPHI_MM','LPV',SZ,intg=LPV)
    ELSE
       CALL CHMALLOC('vangle_mm_module.src','MAKPHI_MM','LPV',SZ,intg=LPV)
    ENDIF
    IF(ALLOCATED(VIND)) THEN
       CALL CHMREALLOC('vangle_mm_module.src','MAKPHI_MM','VIND',SZ,intg=VIND)
    ELSE
       CALL CHMALLOC('vangle_mm_module.src','MAKPHI_MM','VIND',SZ,intg=VIND)
    ENDIF
    IF (associated(TorFC1)) THEN
       call chmrealloc('vangle_mm_module.src','MAKPHI_MM','TorFC1',NPHIV,crlp=TorFC1)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKPHI_MM','TorFC1',NPHIV,crlp=TorFC1)
    ENDIF
    IF (associated(TorFC2)) THEN
       call chmrealloc('vangle_mm_module.src','MAKPHI_MM','TorFC2',NPHIV,crlp=TorFC2)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKPHI_MM','TorFC2',NPHIV,crlp=TorFC2)
    ENDIF
    IF (associated(TorFC3)) THEN
       call chmrealloc('vangle_mm_module.src','MAKPHI_MM','TorFC3',NPHIV,crlp=TorFC3)
    ELSE
       call chmalloc('vangle_mm_module.src','MAKPHI_MM','TorFC3',NPHIV,crlp=TorFC3)
    ENDIF

    CALL MAKPHI2_MM(NPHIV,IPV,JPV,KPV,LPV,VIND,TorFC1,TorFC2,TorFC3)
    RETURN
  END SUBROUTINE MAKPHI_MM
  !
#endif /* (vanglefcm)*/

end module vangle_mm

