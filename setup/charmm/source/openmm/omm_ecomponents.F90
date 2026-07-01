module omm_ecomp
  use chm_kinds
  use number
  use energym
  use stream, only: OUTU, PRNLEV
  use OpenMM
  implicit none

  private
#if KEY_OPENMM==1

  ! neterm provides a mapping between the force groups in OpenMM
  ! and the energy terms in CHARMM (from module energym).
  ! the following mapping is enforced:
  ! All nonbonded (VDW, ELEC, IMVDW, IMELEC, EWKSUM, EWSELF, EXTNDE) => group 0
  ! The following mapping occurs with the call to set-up the specific energy 
  ! term, and hence is in the order of the call
  ! BOND+UREYB:ANGLE:DIHE:IMDIHE:CMAPS:CHARM:CDIHE:RESD:GBEnr

  integer, save :: neterm(0:31), numeterms

  public :: numeterms, omm_init_eterms, omm_incr_eterms, &
       omm_assign_eterms

contains
  
  subroutine omm_init_eterms
    ! assign values to pointers
    numeterms = 0
    neterm = 0
    neterm(0) = vdw  ! Set default energy term to vdw, in general this will
                     ! be all of the non-bonded terms.
  end subroutine omm_init_eterms

  integer*4 function omm_incr_eterms(term)
    character(len=*), intent(in) :: term
    numeterms = numeterms + 1
    omm_incr_eterms = numeterms
    if(trim(term) == 'bond') then
       neterm(numeterms) = bond
    else if (trim(term) == 'angle') then
       neterm(numEterms) = angle
    else if (trim(term) == 'dihe') then
       neterm(numeterms) = dihe
    else if (trim(term) == 'imdihe') then
       neterm(numeterms) = imdihe
    else if (trim(term) == 'cmap') then
       neterm(numeterms) = cmap
    else if (trim(term) == 'charm') then
       neterm(numeterms) = charm
    else if (trim(term) == 'pcharm') then
       neterm(numeterms) = pcharm
    else if (trim(term) == 'cdihe') then
       neterm(numeterms) = cdihe
    else if (trim(term) == 'resd') then
       neterm(numeterms) = resd
    else if (trim(term) == 'geo') then
       neterm(numeterms) = geo
    else if (trim(term) == 'gbenr') then
       numeterms = numeterms - 1
       omm_incr_eterms = 0
    else
       if (prnlev >=2) write(outu,'(a,a,a)') &
            'CHARMM> omm_incr_eterm, term=',trim(term),&
            ' not recognized, ignoring'
       numeterms = numeterms - 1
       omm_incr_eterms = 0
    endif

  end function omm_incr_eterms

  subroutine omm_assign_eterms(context, enforce_periodic)
    use omm_util, only: omm_group_potential
    use OpenMM, only: OpenMM_Context, OpenMM_KJPerKcal
    implicit none

    type(OpenMM_Context), intent(in) :: context
    integer*4, intent(in) :: enforce_periodic

    real(chm_real) :: Epterm
    integer*4 :: itype, group

    if(numeterms>=0) then
      do itype = 0, numeterms
        group=ishft(1,itype)
        Epterm = omm_group_potential(context, enforce_periodic, group) / OpenMM_KJPerKcal
        ETERM(neterm(itype)) = Epterm
      end do
    end if
  end subroutine omm_assign_eterms
#endif
 end module omm_ecomp
