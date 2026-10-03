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

  ! One memo slot per custom-force bucket (CFINT, CFNON, CFEXT, CFMNY, CFCV).
  ! Each entry stores the OpenMM force-group bit assigned to that bucket the
  ! first time a force is added to it, so subsequent forces in the same
  ! bucket reuse the same bit.  -1 means "not yet allocated".
  integer, parameter :: NUM_CF_BUCKETS = 5
  integer, save :: bucket_group(NUM_CF_BUCKETS) = -1

  public :: numeterms, omm_init_eterms, omm_incr_eterms, &
       omm_assign_eterms

contains

  subroutine omm_init_eterms
    ! assign values to pointers
    numeterms = 0
    neterm = 0
    neterm(0) = vdw  ! Set default energy term to vdw, in general this will
                     ! be all of the non-bonded terms.
    bucket_group = -1
    ! Clear lazy CETERM names so a previous run's custom-force buckets
    ! do not leak into a run that has none.
    CETERM(CFINT) = '    '
    CETERM(CFNON) = '    '
    CETERM(CFEXT) = '    '
    CETERM(CFMNY) = '    '
    CETERM(CFCV)  = '    '
  end subroutine omm_init_eterms

  ! Helper: return the OpenMM force-group bit for a custom-force bucket,
  ! allocating one (and the matching ETERM slot) on first use.
  integer*4 function bucket_group_for(slot, term_idx)
    implicit none
    integer, intent(in) :: slot, term_idx
    if (bucket_group(slot) >= 0) then
       bucket_group_for = bucket_group(slot)
    else
       numeterms = numeterms + 1
       neterm(numeterms) = term_idx
       bucket_group(slot) = numeterms
       QETERM(term_idx) = .true.
       select case (term_idx)
       case (CFINT); CETERM(CFINT) = 'CFIN'
       case (CFNON); CETERM(CFNON) = 'CFNB'
       case (CFEXT); CETERM(CFEXT) = 'CFEX'
       case (CFMNY); CETERM(CFMNY) = 'CFMB'
       case (CFCV);  CETERM(CFCV)  = 'CFCV'
       end select
       bucket_group_for = numeterms
    end if
  end function bucket_group_for

  integer*4 function omm_incr_eterms(term)
#if KEY_OMMTORCH == 1
    use energym, only: nnpo
#endif
    implicit none
    character(len=*), intent(in) :: term

    ! Custom-force buckets reuse one force-group bit per bucket, so they
    ! are dispatched before the per-call increment to avoid burning bits.
    if (trim(term) == 'cfint') then
       omm_incr_eterms = bucket_group_for(1, CFINT)
       return
    else if (trim(term) == 'cfnon') then
       omm_incr_eterms = bucket_group_for(2, CFNON)
       return
    else if (trim(term) == 'cfext') then
       omm_incr_eterms = bucket_group_for(3, CFEXT)
       return
    else if (trim(term) == 'cfmny') then
       omm_incr_eterms = bucket_group_for(4, CFMNY)
       return
    else if (trim(term) == 'cfcv') then
       omm_incr_eterms = bucket_group_for(5, CFCV)
       return
    end if

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
       neterm(numeterms) = gbenr
#if KEY_OMMTORCH == 1
    else if (trim(term) == 'nnpo') then
       neterm(numeterms) = nnpo
#endif
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
