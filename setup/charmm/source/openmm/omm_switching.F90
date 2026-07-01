!> Implements a switching function multiplier on nonbonded forces
!> as described in
!>
!> Brooks, B. R., Bruccoleri, R. E., Olafson, B. D., States,
!> D. J., Swaminathan, S. and Karplus, M. (1983),
!> CHARMM: A program for macromolecular energy, minimization,
!> and dynamics calculations.
!> J. Comput. Chem., 4: 187–217.
!> doi: 10.1002/jcc.540040211
!>
!> see footnote to Equation 13
!>
!> Josh Buckner, 2013
!>

! TODO
! add nbfixes, block

module omm_switching
#if KEY_OPENMM==1
  use chm_kinds
  use number
  use stream, only: OUTU, PRNLEV
  use OpenMM
  use omm_nbopts
  implicit none

  private
  logical :: zero_charge, zero_vdw
  logical :: use_vdw, use_elec, use_soft_omm
  integer, parameter :: NOT_FOUND = -1
  integer, parameter :: MAX_EXCEPTIONS = int(1e+7)

  public :: setup_switching

contains

  subroutine setup_switching(system, nbopts)
    use stream, only: OUTU, PRNLEV
    use energym

    type(omm_nbopts_t), intent(in) :: nbopts
    type(OpenMM_System), intent(inout) :: system
    ! TODO: Need to deal with skipe options

    zero_charge = .not. QETERM(ELEC)
    zero_vdw = .not. QETERM(VDW)
    use_vdw = .not. nbopts%use_omm_vdw
    use_elec = .not. nbopts%use_omm_elec
    use_soft_omm = nbopts%use_soft_omm

    if( (use_vdw .and. .not. use_elec) .or. &
         (use_vdw .and. use_elec) )  then
       
       call setup_vdwelec_switched_force(system, nbopts)

    else
!       write(*,*) use_vdw, use_elec, nbopts%use_vdw_swit,  nbopts%use_ewald, nbopts%use_elec_rxnfld, nbopts%use_gbsaobc2
       call wrndie(-1,'omm_switching:setup_switching', &
            'Unsupported switching options for use with CHARMM/OpenMM interface')
    endif
  end subroutine setup_switching

  subroutine setup_vdwelec_switched_force(system, nbopts)
    use omm_nonbond, only : nbfix_set_exclusions, nb_set_fixes
    use omm_block, only : getlambda
    use bases_fcm, only : BNBND
    use block_ltm, only : QBLOCK, NBLOCK, QSOFT_OMM

    type(OpenMM_System), intent(inout) :: system
    type(omm_nbopts_t), intent(in) :: nbopts

    character(len=1280) :: switched_formula
    character(len=16), dimension(4) :: params
    character(len=16) :: name
    type(OpenMM_CustomNonbondedForce) :: switchedforce
    type(OpenMM_CustomBondForce) :: switchednb14force
    real*8 :: r_on, r_off, lambda
    real*8 :: box_a(3), box_b(3), box_c(3), box(3)
    integer :: i, iparam, istrt, iend
    integer :: iforce

    write(params(1), '(a)') 'charge'
    write(params(2), '(a)') 'sigma'
    write(params(3), '(a)') 'epsilon'
    write(params(4), '(a)') 'block'

    call get_switchedformula(nbopts, switched_formula)

    if(PRNLEV >= 7) write(OUTU, '(a, x, a, /, a)') &
         'CHARMM> an OpenMM force will be created', &
         'using the following OpenMM compatible formula', &
         trim(switched_formula)
    
    call OpenMM_CustomNonbondedForce_create(switchedforce, &
         trim(switched_formula))

    r_on = nbopts%switchdist / OpenMM_AngstromsPerNm
    r_off = nbopts%rcut / OpenMM_AngstromsPerNm

    if (nbopts%periodic) then
       call OpenMM_CustomNonbondedForce_setNonbondedMethod( &
            switchedforce, OpenMM_CustomNonbondedForce_CutoffPeriodic)
       
       ! If periodic box not already setup, set size here
       if(use_elec) then
          box = nbopts%box / OpenMM_AngstromsPerNm
          box_a = ZERO
          box_b = ZERO
          box_c = ZERO
          box_a(1) = box(1)
          box_b(2) = box(2)
          box_c(3) = box(3)
          call OpenMM_System_setDefaultPeriodicBoxVectors(system, &
               box_a, box_b, box_c)
       endif
    else if (nbopts%rcut < 999) then
       call OpenMM_CustomNonbondedForce_setNonbondedMethod( &
            switchedforce, OpenMM_CustomNonbondedForce_CutoffNonPeriodic)
    else
       call wrndie(-1,'omm_switching: setup_vdw_force', &
            'Unsupported cutoff scheme requested, require ctofnb < 999 with switches')
    endif
    call OpenMM_CustomNonbondedForce_setCutoffDistance( &
         switchedforce, r_off)

    iparam = OpenMM_CustomNonbondedForce_addGlobalParameter( &
         switchedforce, 'Ron', r_on)
    if(PRNLEV >= 7) then
       write(OUTU, '(2a, i4)') &
            'CHARMM> global parameters Ron added to ', &
            'OpenMM CustomNonbondedForce at index ', &
            iparam
       call OpenMM_CustomNonbondedForce_getGlobalParameterName(switchedforce,iparam,name)
       write(*,*) name
    endif

    iparam = OpenMM_CustomNonbondedForce_addGlobalParameter( &
         switchedforce, 'Roff', r_off)
    if(PRNLEV >= 7) then
       write(OUTU, '(2a, i4)') &
            'CHARMM> global parameter Roff added to ', &
            'OpenMM CustomNonbondedForce at index ', &
            iparam
       call OpenMM_CustomNonbondedForce_getGlobalParameterName(switchedforce,iparam,name)
       write(*,*) name
    endif


    if(nbopts%use_soft_omm) then
       if ( nblock > 3 ) then
          write(outu,'(a)')'setup_vdwelec_switched_force: vdW Softcore requested with NBLOK > 3'
          call wrndie(-3, '<setup_vdwelec_switched_force>', &
               'Softcore unavailable for NBLOCK > 3')
       endif
       lambda = getlambda()
       iparam = OpenMM_CustomNonbondedForce_addGlobalParameter( &
            switchedforce, 'lambda', lambda)
       if(PRNLEV >= 7) then
          write(OUTU, '(2a, i4)') &
               'CHARMM> global parameter lambda added to ', &
               'OpenMM CustomNonbondedForce at index ', &
               iparam
          call OpenMM_CustomNonbondedForce_getGlobalParameterName(switchedforce,iparam,name)
          write(*,*) name
       endif
       
    endif

    iend = 3
    istrt = 2
    if(use_elec) istrt = 1
if (.not. use_elec .and. nbopts%use_soft_omm) iend = 4
    do i = istrt, iend
       iparam = OpenMM_CustomNonbondedForce_addPerParticleParameter( &
            switchedforce, trim(params(i)))
       if(PRNLEV >= 7) write(OUTU, '(3a, i4)') &
            'CHARMM> per particle parameter ', &
            trim(params(i)), &
            ' added to OpenMM CustomNonbondedForce at index ', &
            iparam
    end do

    iforce = OpenMM_System_addForce(system, transfer(switchedforce, OpenMM_Force(0)))
    if(PRNLEV >= 7) write(OUTU, '(a, i4)') &
         'CHARMM> Switched van der Waals/Electrostatic force added to OpenMM system at index ', &
         iforce
    
    call add_switched_atoms(switchedforce)
    
    ! 1-4  and other exclusions and exceptions are removed from list
    call nbfix_set_exclusions(switchedforce)   ! This routine is really generic for CustomNonbondedForce
    
    ! Add NBFIxed interactions if they exist
    call nb_set_fixes(system, nbopts)

    ! Add 1-4 exceptions using CustomBondForce
    call get_14switchedformula(nbopts,switched_formula)
    call setup_switched14_force(system, switchednb14force, switched_formula, nbopts)
    call nbswitch_setup_vdwelec14_interactions(switchednb14force)
#if KEY_BLOCK==1
    if(qblock) then
       call blockscale_nbswitch_intrablock(switchedforce, switchednb14force, & 
            bnbnd%INB14, bnbnd%IBLO14)                                         
    endif
#endif 
    if(prnlev >= 7) then   
        call OpenMM_CustomNonbondedForce_getEnergyFunction(switchedforce,switched_formula)
        write(OUTU, '(a, x, a, /, a)') &
             'CHARMM> an OpenMM force will be created', &
             'using the following OpenMM compatible formula', &
             trim(switched_formula)
    
        call OpenMM_CustomBondForce_getEnergyFunction(switchednb14force,switched_formula)
        write(OUTU, '(a, x, a, /, a)') &
             'CHARMM> an OpenMM force will be created for nonbonded 1-4 interactions', &
             'using the following OpenMM compatible formula', &
             trim(switched_formula)
        ! Debug
        iparam = OpenMM_CustomBondForce_getNumGlobalParameters(switchednb14force)
        write(OUTU,'(a,x,i5)') &
             ' Number of global parameters for NB14 is ', iparam
        iparam = OpenMM_CustomBondForce_getNumPerBondParameters(switchednb14force)
        write(OUTU,'(a,x,i5)') &
             ' Number of per pair parameters for NB14 is ', iparam
        iparam = OpenMM_CustomBondForce_getNumBonds(switchednb14force)
        write(OUTU,'(a,x,i5)') &
             ' Number of NB14 pairs is ', iparam
     endif
   end subroutine setup_vdwelec_switched_force
  
  subroutine setup_switched14_force(system, switchednb14force, switched_formula, nbopts)
    use omm_nonbond, only : nbfix_set_exclusions

    type(OpenMM_System), intent(inout) :: system
    type(omm_nbopts_t), intent(in) :: nbopts
    type(OpenMM_CustomBondForce), intent(inout) :: switchednb14force
    character(len=1280), intent (in) :: switched_formula
 
    character(len=16), dimension(3) :: params
     real*8 :: r_on, r_off
    integer :: i, iparam, istrt
    integer :: iforce

    write(params(1), '(a)') 'charge'
    write(params(2), '(a)') 'sigma'
    write(params(3), '(a)') 'epsilon'

    call OpenMM_CustomBondForce_create(switchednb14force, &
         trim(switched_formula))


    r_on = nbopts%switchdist / OpenMM_AngstromsPerNm
    r_off = nbopts%rcut / OpenMM_AngstromsPerNm

    iparam = OpenMM_CustomBondForce_addGlobalParameter( &
         switchednb14force, 'Ron', r_on)
    if(PRNLEV >= 7) write(OUTU, '(2a, i4)') &
         'CHARMM> global parameters Ron added to ', &
         'OpenMM CustomBondForce at index ', &
         iparam

    iparam = OpenMM_CustomBondForce_addGlobalParameter( &
         switchednb14force, 'Roff', r_off)
    if(PRNLEV >= 7) write(OUTU, '(2a, i4)') &
         'CHARMM> global parameter Roff added to ', &
         'OpenMM CustomBondForce at index ', &
         iparam

    istrt = 2
    if(use_vdw .and. use_elec) istrt = 1
    do i = istrt, 3
       iparam = OpenMM_CustomBondForce_addPerBondParameter( &
            switchednb14force, trim(params(i)))
       if(PRNLEV >= 7) write(OUTU, '(3a, i4)') &
            'CHARMM> per particle parameter ', &
            trim(params(i)), &
            ' added to OpenMM CustomBondForce at index ', &
            iparam
    end do

    iforce = OpenMM_System_addForce(system, transfer(switchednb14force, OpenMM_Force(0)))
    if(PRNLEV >= 7) write(OUTU, '(a, i4)') &
         'CHARMM> Switched vdW and electrostatic CustomBondForce added to OpenMM system at index ', &
         iforce
  end subroutine setup_switched14_force

   subroutine get_14switchedformula(nbopts, formula)
    type(omm_nbopts_t), intent(in) :: nbopts
    character(*), intent(inout) :: formula

    if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
         (nbopts%rcut > nbopts%switchdist)) then                  ! vdW only - switch
       write (formula, '(8a)') &
         '(step(Ron - r) + ', &
         '(step(r - Ron) * step(Roff - r) - ', &
         'step(r - Ron) * step(Ron - r)) * ', &
         '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
         '(Roff2 - Ron2)^3)) * ', &
         '4.0 * epsilon * six * (six - 1.0); ', &
         'six = (sigma / r)^6; ', &
         'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '

    else if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
         (nbopts%rcut <= nbopts%switchdist)) then                  ! vdW only - switch
       write (formula, '(2a)') &
         '4.0 * epsilon * six * (six - 1.0); ', &
         'six = (sigma / r)^6; '

   else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
         (nbopts%rcut > nbopts%switchdist) ) then                               ! vdW only - fswitch
      write (formula, '(26a)') &
           'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
           '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &
           '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3);', &
           'cr6  = ccnbb*ofdif3*rjunk3;', &
           'cr12 = ccnba*ofdif6*rjunk6;', &
           'rjunk3 = r3-recof3;', &
           'rjunk6 = tr6-recof6;', &
           'r3 = r1*tr2;', &
           'r1 = sqrt(tr2);', &
           'tr6 = tr2 * tr2 * tr2;', &
           'tr2 = 1.0/s2;', &
           's2 = r*r;', &
           'ccnbb = 4.0*epsilon*sigma^6;', &
           'ccnba = 4.0*epsilon*sigma^12;', &
           'onoff3 = recof3/on3;', &
           'onoff6 = recof6/on6;', &
           'ofdif3 = off3/(off3 - on3);', &
           'ofdif6 = off6/(off6 - on6);', &
           'recof3 = 1.0/off3;', &
           'on6 = on3*on3;', &
           'on3 = c2onnb*Ron;', &
           'recof6 = 1.0/off6;', &
           'off6 = off3*off3;', &
           'off3 = c2ofnb*Roff;', &
           'c2ofnb = Roff*Roff;', &
           'c2onnb = Ron*Ron;'
      
   else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
         (nbopts%rcut <= nbopts%switchdist) ) then                                   ! vdW only - fswitch
       write (formula, '(13a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6);', &
            'tr6 = tr2 * tr2 * tr2;', &
            'tr2 = 1.0/s2;', &
            's2 = r*r;', &
            'ccnbb = 4.0*epsilon*sigma^6;', &
            'ccnba = 4.0*epsilon*sigma^12;', &
            'onoff3 = rcof6;', &
            'onoff6 = rcof6*rcof6;', &
            'rcof6 = 1.0/off6;', &
            'off6 = off3*off3;', &
            'off3 = c2ofnb*Roff;', &
            'c2ofnb = Roff*Roff;', &
            'c2onnb = Ron*Ron;'

    else if(nbopts%use_vdw_swit .and. nbopts%use_elec_swit .and. &
         (nbopts%rcut > nbopts%switchdist)) then     ! vdW switch, elec switch
       write (formula, '(8a)') &
            '(step(Ron - r) + ', &
            '(step(r - Ron) * step(Roff - r) - ', &
            'step(r - Ron) * step(Ron - r)) * ', &
            '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
            '(Roff2 - Ron2)^3)) * ', &
            '(4.0 * epsilon * six * (six - 1.0) + (138.935456 * charge) / r);', &
            'six = (sigma / r)^6; ', &
            'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '

    else if(nbopts%use_vdw_swit .and. nbopts%use_elec_swit .and. &
         (nbopts%rcut <= nbopts%switchdist)) then     ! vdW switch, elec switch
       write (formula, '(3a)') &
            '(4.0 * epsilon * six * (six - 1.0) + (138.935456 * charge) / r);', &
            'six = (sigma / r)^6; ', &
            'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
       
    else if(nbopts%use_vdw_swit .and. nbopts%use_elec_fshift .and. &
         (nbopts%rcut > nbopts%switchdist)) then     ! vdW switch, elec fshift
       write (formula, '(9a)') &
            '(step(Ron - r) + ', &
            '(step(r - Ron) * step(Roff - r) - ', &
            'step(r - Ron) * step(Ron - r)) * ', &
            '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
            '(Roff2 - Ron2)^3)) * ', &
            '4.0 * epsilon * six * (six - 1.0)', &
            ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge) / r);', &
            'six = (sigma / r)^6; ', &
            'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
       
    else if(nbopts%use_vdw_swit .and. nbopts%use_elec_fshift .and. &
         (nbopts%rcut <= nbopts%switchdist)) then     ! vdW switch, elec fshift
       write (formula, '(4a)') &
            '4.0 * epsilon * six * (six - 1.0)', &
            ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge) / r);', &
            'six = (sigma / r)^6; ', &
            'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
       
    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fswit .and. &
         (nbopts%rcut > nbopts%switchdist)) then        ! vdW fswitch, elec fswitch
       write (formula, '(42a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
            '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &               ! vfswit
            '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &                ! vfswit
            '+step(Ron-r)*ch*(r1+eadd)', &                                          ! cfswit
            '+(step(r-Ron)*step(Roff-r) - step(r-Ron)*step(Ron-r))*',&              ! cfswit
            'ch*( r1* (acoef - s2*(bcoef + s2*(cover3 + dover5*s2))) + const);', &  ! cfswit
            'ch=138.935456*charge;', &                                              ! cfswit
            'const  = bcoef*Roff-acoef/Roff+cover3*off3+dover5*off5;', &            ! cfswit
            'dover5 = dcoef/5.0;', &                                                ! cfswit
            'dcoef  = 2.0*denom;', &                                                ! cfswit
            'ccoef  = 3.0*cover3;', &                                               ! cfswit
            'cover3 = -(c2onnb+c2ofnb)*denom;', &                                   ! cfswit
            'acoef  = off4*(c2ofnb-3.0*c2onnb)*denom;', &                           ! cfswit
            'bcoef  = 6.0*onoff2*denom;', &                                         ! cfswit
            'eadd   = (onoff2*(Roff-Ron)-(off5-on3*c2onnb)/5.0)*8.0*denom;', &      ! cfswit
            'denom  = 1.0/(c2ofnb-c2onnb)^3;', &                                    ! cfswit
            'off4   = c2ofnb*c2ofnb;', &                                            ! cfswit
            'off5   = off3*c2ofnb;', &                                              ! cfswit
            'onoff2 = c2onnb*c2ofnb;', &                                            ! cfswit
            'cr6  = ccnbb*ofdif3*rjunk3;', &                                        ! vfswit
            'cr12 = ccnba*ofdif6*rjunk6;', &                                        ! vfswit
            'rjunk3 = r3-recof3;', &                                                ! vfswit
            'rjunk6 = tr6-recof6;', &                                               ! vfswit
            'r3 = r1*tr2;', &                                                       ! vfswit
            'r1 = sqrt(tr2);', &                                                    ! cfswit/vfswit
            'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
            'tr2 = 1.0/s2;', &                                                      ! vfswit
            's2 = r*r;', &                                                          ! cfswit/vfswit
            'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
            'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
            'onoff3 = recof3/on3;', &                                               ! vfswit
            'onoff6 = recof6/on6;', &                                               ! vfswit
            'ofdif3 = off3/(off3 - on3);', &                                        ! vfswit
            'ofdif6 = off6/(off6 - on6);', &                                        ! vfswit
            'recof3 = 1.0/off3;', &                                                 ! vfswit
            'on6 = on3*on3;', &                                                     ! vfswit
            'on3 = c2onnb*Ron;', &                                                  ! cfswit/vfswit
            'recof6 = 1.0/off6;', &                                                 ! vfswit
            'off6 = off3*off3;', &                                                  ! vfswit
            'off3 = c2ofnb*Roff;', &                                                ! cfswit/vfswit
            'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
            'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
       
    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fswit .and. &
         (nbopts%rcut <= nbopts%switchdist)) then        ! vdW fswitch, elec fswitch
       write (formula, '(17a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
            '+step(Ron-r)*ch*(r1+eadd);', &                                         ! cfswit
            'ch=138.935456*charge;', &                                              ! cfswit
            'eadd   = -1./Roff;', &                                                 ! cfswit
            'r1 = sqrt(tr2);', &                                                    ! cfswit/vfswit
            'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
            'tr2 = 1.0/s2;', &                                                      ! vfswit
            's2 = r*r;', &                                                          ! cfswit/vfswit
            'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
            'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
            'onoff3 = rcof6;', &                                                    ! vfswit
            'onoff6 = rcof6*rcof6;', &                                              ! vfswit
            'rcof6 = 1.0/off6;', &                                                  ! vfswit
            'off6 = off3*off3;', &                                                  ! vfswit
            'off3 = c2ofnb*Roff;', &                                                ! cfswit/vfswit
            'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
            'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
       
    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_swit .and. &
         (nbopts%rcut > nbopts%switchdist)) then        ! vdW fswitch, elec switch
       write (formula, '(31a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
            '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &               ! vfswit
            '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &                ! vfswit
            '+ (step(Ron - r) - ', &                                                ! cswit
            '(step(r-Ron)*step(Roff-r) - step(r-Ron)*step(Ron-r)) * ', &            ! cswit
            '(((c2ofnb - r2)^2 * (c2ofnb + 2.0 * r2 - 3.0 * c2onnb)) / ', &         ! cswit
            '(c2ofnb - c2onnb)^3)) * ', &                                           ! cswit
            '(138.935456 * charge) / r;', &                                         ! cswit
            'cr6  = ccnbb*ofdif3*rjunk3;', &                                        ! vfswit
            'cr12 = ccnba*ofdif6*rjunk6;', &                                        ! vfswit
            'rjunk3 = r3-recof3;', &                                                ! vfswit
            'rjunk6 = tr6-recof6;', &                                               ! vfswit
            'r3 = r1*tr2;', &                                                       ! vfswit
            'r1 = sqrt(tr2);', &                                                    ! vfswit
            'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
            'tr2 = 1.0/r2;', &                                                      ! vfswit
            'r2 = r*r;', &                                                          ! vfswit
            'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
            'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
            'onoff3 = recof3/on3;', &                                               ! vfswit
            'onoff6 = recof6/on6;', &                                               ! vfswit
            'ofdif3 = off3/(off3 - on3);', &                                        ! vfswit
            'ofdif6 = off6/(off6 - on6);', &                                        ! vfswit
            'recof3 = 1.0/off3;', &                                                 ! vfswit
            'on6 = on3*on3;', &                                                     ! vfswit
            'on3 = c2onnb*Ron;', &                                                  ! vfswit
            'recof6 = 1.0/off6;', &                                                 ! vfswit
            'off6 = off3*off3;', &                                                  ! vfswit
            'off3 = c2ofnb*Roff;', &                                                ! vfswit
            'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
            'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
       
    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_swit .and. &
         (nbopts%rcut <= nbopts%switchdist)) then                                   ! vdW fswitch, elec switch
       write (formula, '(13a)') &
            '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &                ! vfswit
            '+ (138.935456 * charge) / r;', &                                       ! cswit
             'tr6 = tr2 * tr2 * tr2;', &                                            ! vfswit
            'tr2 = 1.0/s2;', &                                                      ! vfswit
            's2 = r*r;', &                                                          ! vfswit
            'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
            'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
            'onoff3 = rcof6;', &                                                    ! vfswit
            'onoff6 = rcof6*rcof6;', &                                              ! vfswit
            'rcof6 = 1.0/off6;', &                                                  ! vfswit
            'off6 = off3*off3;', &                                                  ! vfswit
            'off3 = c2ofnb*Roff;', &                                                ! vfswit
            'c2ofnb = Roff*Roff;'                                                   ! vfswit
       
    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fshift .and. &
         (nbopts%rcut > nbopts%switchdist)) then     ! vdW fswitch, elec fshift
       write (formula, '(27a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
            '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &
            '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &
            ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge) * r1);', &
            'cr6  = ccnbb*ofdif3*rjunk3;', &
            'cr12 = ccnba*ofdif6*rjunk6;', &
            'rjunk3 = r3-recof3;', &
            'rjunk6 = tr6-recof6;', &
            'r3 = r1*tr2;', &
            'r1 = sqrt(tr2);', &
            'tr6 = tr2 * tr2 * tr2;', &
            'tr2 = 1.0/s2;', &
            's2 = r*r;', &
            'ccnbb = 4.0*epsilon*sigma^6;', &
            'ccnba = 4.0*epsilon*sigma^12;', &
            'onoff3 = recof3/on3;', &
            'onoff6 = recof6/on6;', &
            'ofdif3 = off3/(off3 - on3);', &
            'ofdif6 = off6/(off6 - on6);', &
            'recof3 = 1.0/off3;', &
            'on6 = on3*on3;', &
            'on3 = c2onnb*Ron;', &
            'recof6 = 1.0/off6;', &
            'off6 = off3*off3;', &
            'off3 = c2ofnb*Roff;', &
            'c2ofnb = Roff*Roff;', &
            'c2onnb = Ron*Ron;'

    else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fshift .and. &
         (nbopts%rcut <= nbopts%switchdist)) then     ! vdW fswitch, elec fshift
       write (formula, '(15a)') &
            'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
            ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge) * r1);', &
            'r1 = sqrt(tr2);', &
            'tr6 = tr2 * tr2 * tr2;', &
            'tr2 = 1.0/s2;', &
            's2 = r*r;', &
            'ccnbb = 4.0*epsilon*sigma^6;', &
            'ccnba = 4.0*epsilon*sigma^12;', &
            'onoff3 = rcof6;', &
            'onoff6 = rcof6*rcof6;', &
            'rcof6 = 1.0/off6;', &
            'off6 = off3*off3;', &
            'off3 = c2ofnb*Roff;', &
            'c2ofnb = Roff*Roff;', &
            'c2onnb = Ron*Ron;'

    else
       call wrndie(-1,'omm_switching: setup_switchedforce', &
            'Requested combination of cutoff/switching not supported')

    endif

  end subroutine get_14switchedformula

   subroutine get_switchedformula(nbopts, formula)
    type(omm_nbopts_t), intent(in) :: nbopts
    character(*), intent(inout) :: formula

    if( .not. (nbopts%use_soft_omm) ) then
       if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
            (nbopts%rcut > nbopts%switchdist)) then                  ! vdW only - switch
          write (formula, '(10a)') &
               '(step(Ron - r) + ', &
               '(step(r - Ron) * step(Roff - r) - ', &
               'step(r - Ron) * step(Ron - r)) * ', &
               '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
               '(Roff2 - Ron2)^3)) * ', &
               '4.0 * epsilon * six * (six - 1.0); ', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
          
       else if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
            (nbopts%rcut <= nbopts%switchdist)) then                  ! vdW only - switch
          write (formula, '(4a)') &
               '4.0 * epsilon * six * (six - 1.0); ', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);'
          
       else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
            (nbopts%rcut > nbopts%switchdist) ) then                               ! vdW only - fswitch
          write (formula, '(17a)') &
               'select(step(r-Ron),(cr12*rjunk12 - cr6*rjunk6),', &
               '(ccnba/r12-ccnbb/r6+ccnbb*onoff3-ccnba*onoff6));', &
               'cr6 = ccnbb*ofdif3;', &
               'cr12 = ccnba*ofdif6;', &
               'rjunk6 = (1.0/r3-1.0/Roff^3)^2;', &
               'rjunk12 = (1.0/r6-1.0/Roff^6)^2;', &
               'r12 = r6^2;', &
               'r6 = r3^2;', &
               'r3 = r^3;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = 1.0/(Roff^3 * Ron^3);', &
               'onoff6 = 1.0/(Roff^6 * Ron^6);', &
               'ofdif3 = Roff^3/(Roff^3 - Ron^3);', &
               'ofdif6 = Roff^6/(Roff^6 - Ron^6);'
          !write (formula, '(28a)') &
          !     'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
          !     '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &
          !     '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3);', &
          !     'cr6  = ccnbb*ofdif3*rjunk3;', &
          !     'cr12 = ccnba*ofdif6*rjunk6;', &
          !     'rjunk3 = r3-recof3;', &
          !     'rjunk6 = tr6-recof6;', &
          !     'r3 = r1*tr2;', &
          !     'r1 = sqrt(tr2);', &
          !     'tr6 = tr2 * tr2 * tr2;', &
          !     'tr2 = 1.0/s2;', &
          !     's2 = r*r;', &
          !     'ccnbb = 4.0*epsilon*sigma^6;', &
          !     'ccnba = 4.0*epsilon*sigma^12;', &
          !     'sigma = 0.5*(sigma1+sigma2);', &
          !     'epsilon = sqrt(epsilon1*epsilon2);', &
          !     'onoff3 = recof3/on3;', &
          !     'onoff6 = recof6/on6;', &
          !     'ofdif3 = off3/(off3 - on3);', &
          !     'ofdif6 = off6/(off6 - on6);', &
          !     'recof3 = 1.0/off3;', &
          !     'on6 = on3*on3;', &
          !     'on3 = c2onnb*Ron;', &
          !     'recof6 = 1.0/off6;', &
          !     'off6 = off3*off3;', &
          !     'off3 = c2ofnb*Roff;', &
          !     'c2ofnb = Roff*Roff;', &
          !     'c2onnb = Ron*Ron;'
          
       else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
            (nbopts%rcut <= nbopts%switchdist) ) then                                   ! vdW only - fswitch
          write (formula, '(15a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6);', &
               'tr6 = tr2 * tr2 * tr2;', &
               'tr2 = 1.0/s2;', &
               's2 = r*r;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = rcof6;', &
               'onoff6 = rcof6*rcof6;', &
               'rcof6 = 1.0/off6;', &
               'off6 = off3*off3;', &
               'off3 = c2ofnb*Roff;', &
               'c2ofnb = Roff*Roff;', &
               'c2onnb = Ron*Ron;'
          
       else if(nbopts%use_vdw_swit .and. nbopts%use_elec_swit .and. &
            (nbopts%rcut > nbopts%switchdist)) then     ! vdW switch, elec switch
          write (formula, '(10a)') &
               '(step(Ron - r) + ', &
               '(step(r - Ron) * step(Roff - r) - ', &
               'step(r - Ron) * step(Ron - r)) * ', &
               '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
               '(Roff2 - Ron2)^3)) * ', &
               '(4.0 * epsilon * six * (six - 1.0) + (138.935456 * charge1 * charge2) / r);', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
          
       else if(nbopts%use_vdw_swit .and. nbopts%use_elec_swit .and. &
            (nbopts%rcut <= nbopts%switchdist)) then     ! vdW switch, elec switch
          write (formula, '(5a)') &
               '(4.0 * epsilon * six * (six - 1.0) + (138.935456 * charge1 * charge2) / r);', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
          
       else if(nbopts%use_vdw_swit .and. nbopts%use_elec_fshift .and. &
            (nbopts%rcut > nbopts%switchdist)) then     ! vdW switch, elec fshift
          write (formula, '(11a)') &
               '(step(Ron - r) + ', &
               '(step(r - Ron) * step(Roff - r) - ', &
               'step(r - Ron) * step(Ron - r)) * ', &
               '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
               '(Roff2 - Ron2)^3)) * ', &
               '4.0 * epsilon * six * (six - 1.0)', &
               ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge1 * charge2) / r);', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
          
       else if(nbopts%use_vdw_swit .and. nbopts%use_elec_fshift .and. &
            (nbopts%rcut <= nbopts%switchdist)) then     ! vdW switch, elec fshift
          write (formula, '(6a)') &
               '4.0 * epsilon * six * (six - 1.0)', &
               ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge1 * charge2) / r);', &
               'six = (sigma / r)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; '
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fswit .and. &
            (nbopts%rcut > nbopts%switchdist)) then        ! vdW fswitch, elec fswitch
          write (formula, '(44a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
               '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &               ! vfswit
               '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &                ! vfswit
               '+step(Ron-r)*ch*(r1+eadd)', &                                          ! cfswit
               '+(step(r-Ron)*step(Roff-r)-step(r-Ron)*step(Ron-r))*', &               ! cfswit
               'ch*( r1* (acoef - s2*(bcoef + s2*(cover3 + dover5*s2))) + const);', &  ! cfswit
               'ch=138.935456*charge1*charge2;', &                                     ! cfswit
               'const  = bcoef*Roff-acoef/Roff+cover3*off3+dover5*off5;', &            ! cfswit
               'dover5 = dcoef/5.0;', &                                                ! cfswit
               'dcoef  = 2.0*denom;', &                                                ! cfswit
               'ccoef  = 3.0*cover3;', &                                               ! cfswit
               'cover3 = -(c2onnb+c2ofnb)*denom;', &                                   ! cfswit
               'acoef  = off4*(c2ofnb-3.0*c2onnb)*denom;', &                           ! cfswit
               'bcoef  = 6.0*onoff2*denom;', &                                         ! cfswit
               'eadd   = (onoff2*(Roff-Ron)-(off5-on3*c2onnb)/5.0)*8.0*denom;', &      ! cfswit
               'denom  = 1.0/(c2ofnb-c2onnb)^3;', &                                    ! cfswit
               'off4   = c2ofnb*c2ofnb;', &                                            ! cfswit
               'off5   = off3*c2ofnb;', &                                              ! cfswit
               'onoff2 = c2onnb*c2ofnb;', &                                            ! cfswit
               'cr6  = ccnbb*ofdif3*rjunk3;', &                                        ! vfswit
               'cr12 = ccnba*ofdif6*rjunk6;', &                                        ! vfswit
               'rjunk3 = r3-recof3;', &                                                ! vfswit
               'rjunk6 = tr6-recof6;', &                                               ! vfswit
               'r3 = r1*tr2;', &                                                       ! vfswit
               'r1 = sqrt(tr2);', &                                                    ! cfswit/vfswit
               'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
               'tr2 = 1.0/s2;', &                                                      ! vfswit
               's2 = r*r;', &                                                          ! cfswit/vfswit
               'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
               'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
               'sigma = 0.5*(sigma1+sigma2);', &                                       ! vfswit
               'epsilon = sqrt(epsilon1*epsilon2);', &                                 ! vfswit
               'onoff3 = recof3/on3;', &                                               ! vfswit
               'onoff6 = recof6/on6;', &                                               ! vfswit
               'ofdif3 = off3/(off3 - on3);', &                                        ! vfswit
               'ofdif6 = off6/(off6 - on6);', &                                        ! vfswit
               'recof3 = 1.0/off3;', &                                                 ! vfswit
               'on6 = on3*on3;', &                                                     ! vfswit
               'on3 = c2onnb*Ron;', &                                                  ! cfswit/vfswit
               'recof6 = 1.0/off6;', &                                                 ! vfswit
               'off6 = off3*off3;', &                                                  ! vfswit
               'off3 = c2ofnb*Roff;', &                                                ! cfswit/vfswit
               'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
               'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fswit .and. &
            (nbopts%rcut <= nbopts%switchdist)) then        ! vdW fswitch, elec fswitch
          write (formula, '(19a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
               '+step(Ron-r)*ch*(r1+eadd);', &                                         ! cfswit
               'ch=138.935456*charge1*charge2;', &                                     ! cfswit
               'eadd   = -1./Roff;', &                                                 ! cfswit
               'r1 = sqrt(tr2);', &                                                    ! cfswit/vfswit
               'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
               'tr2 = 1.0/s2;', &                                                      ! vfswit
               's2 = r*r;', &                                                          ! cfswit/vfswit
               'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
               'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
               'sigma = 0.5*(sigma1+sigma2);', &                                       ! vfswit
               'epsilon = sqrt(epsilon1*epsilon2);', &                                 ! vfswit
               'onoff3 = rcof6;', &                                                    ! vfswit
               'onoff6 = rcof6*rcof6;', &                                              ! vfswit
               'rcof6 = 1.0/off6;', &                                                  ! vfswit
               'off6 = off3*off3;', &                                                  ! vfswit
               'off3 = c2ofnb*Roff;', &                                                ! cfswit/vfswit
               'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
               'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_swit .and. &
            (nbopts%rcut > nbopts%switchdist)) then        ! vdW fswitch, elec switch
          write (formula, '(34a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &    ! vfswit
               '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &               ! vfswit
               '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &                ! vfswit
               '+ (step(Ron - r) + ', &                                                ! cswit
               '(step(r - Ron) * step(Roff - r) - ', &                                 ! cswit
               'step(r - Ron) * step(Ron - r)) * ', &                                  ! cswit
               '(((c2ofnb - r2)^2 * (c2ofnb + 2.0 * r2 - 3.0 * c2onnb)) / ', &         ! cswit
               '(c2ofnb - c2onnb)^3)) * ', &                                           ! cswit
               '(138.935456 * charge1 * charge2) / r;', &                              ! cswit
               'cr6  = ccnbb*ofdif3*rjunk3;', &                                        ! vfswit
               'cr12 = ccnba*ofdif6*rjunk6;', &                                        ! vfswit
               'rjunk3 = r3-recof3;', &                                                ! vfswit
               'rjunk6 = tr6-recof6;', &                                               ! vfswit
               'r3 = r1*tr2;', &                                                       ! vfswit
               'r1 = sqrt(tr2);', &                                                    ! vfswit
               'tr6 = tr2 * tr2 * tr2;', &                                             ! vfswit
               'tr2 = 1.0/r2;', &                                                      ! vfswit
               'r2 = r*r;', &                                                          ! vfswit
               'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
               'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'onoff3 = recof3/on3;', &                                               ! vfswit
               'onoff6 = recof6/on6;', &                                               ! vfswit
               'ofdif3 = off3/(off3 - on3);', &                                        ! vfswit
               'ofdif6 = off6/(off6 - on6);', &                                        ! vfswit
               'recof3 = 1.0/off3;', &                                                 ! vfswit
               'on6 = on3*on3;', &                                                     ! vfswit
               'on3 = c2onnb*Ron;', &                                                  ! vfswit
               'recof6 = 1.0/off6;', &                                                 ! vfswit
               'off6 = off3*off3;', &                                                  ! vfswit
               'off3 = c2ofnb*Roff;', &                                                ! vfswit
               'c2ofnb = Roff*Roff;', &                                                ! cfswit/vfswit
               'c2onnb = Ron*Ron;'                                                     ! cfswit/vfswit
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_swit .and. &
            (nbopts%rcut <= nbopts%switchdist)) then                                   ! vdW fswitch, elec switch
          write (formula, '(15a)') &
               '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &                ! vfswit
               ' + (138.935456 * charge1 * charge2) / r;', &                           ! cswit
               'tr6 = tr2 * tr2 * tr2;', &                                            ! vfswit
               'tr2 = 1.0/s2;', &                                                      ! vfswit
               's2 = r*r;', &                                                          ! vfswit
               'ccnbb = 4.0*epsilon*sigma^6;', &                                       ! vfswit
               'ccnba = 4.0*epsilon*sigma^12;', &                                      ! vfswit
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'onoff3 = rcof6;', &                                                    ! vfswit
               'onoff6 = rcof6*rcof6;', &                                              ! vfswit
               'rcof6 = 1.0/off6;', &                                                  ! vfswit
               'off6 = off3*off3;', &                                                  ! vfswit
               'off3 = c2ofnb*Roff;', &                                                ! vfswit
               'c2ofnb = Roff*Roff;'                                                   ! vfswit
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fshift .and. &
            (nbopts%rcut > nbopts%switchdist)) then     ! vdW fswitch, elec fshift
          write (formula, '(29a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
               '+step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3)', &
               '-step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3)', &
               ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge1 * charge2) * r1);', &
               'cr6  = ccnbb*ofdif3*rjunk3;', &
               'cr12 = ccnba*ofdif6*rjunk6;', &
               'rjunk3 = r3-recof3;', &
               'rjunk6 = tr6-recof6;', &
               'r3 = r1*tr2;', &
               'r1 = sqrt(tr2);', &
               'tr6 = tr2 * tr2 * tr2;', &
               'tr2 = 1.0/s2;', &
               's2 = r*r;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = recof3/on3;', &
               'onoff6 = recof6/on6;', &
               'ofdif3 = off3/(off3 - on3);', &
               'ofdif6 = off6/(off6 - on6);', &
               'recof3 = 1.0/off3;', &
               'on6 = on3*on3;', &
               'on3 = c2onnb*Ron;', &
               'recof6 = 1.0/off6;', &
               'off6 = off3*off3;', &
               'off3 = c2ofnb*Roff;', &
               'c2ofnb = Roff*Roff;', &
               'c2onnb = Ron*Ron;'
          
       else if(nbopts%use_vdw_fswit .and. nbopts%use_elec_fshift .and. &
            (nbopts%rcut <= nbopts%switchdist)) then     ! vdW fswitch, elec fshift
          write (formula, '(17a)') &
               'step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
               ' + step(Roff-r) * (1-r/Roff)^2 * ((138.935456 * charge1 * charge2) * r1);', &
               'r1 = sqrt(tr2);', &
               'tr6 = tr2 * tr2 * tr2;', &
               'tr2 = 1.0/s2;', &
               's2 = r*r;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = rcof6;', &
               'onoff6 = rcof6*rcof6;', &
               'rcof6 = 1.0/off6;', &
               'off6 = off3*off3;', &
               'off3 = c2ofnb*Roff;', &
               'c2ofnb = Roff*Roff;', &
               'c2onnb = Ron*Ron;'
          
       else
          call wrndie(-1,'omm_switching: setup_switchedforce', &
            'Requested combination of cutoff/switching not supported')
          
       endif
    else   ! Using softcore with block
       if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
            (nbopts%rcut > nbopts%switchdist)) then                  ! vdW only - switch
          write (formula, '(16a)') &
               '(step(Ron - rp) + ', &
               '(step(rp - Ron) * step(Roff - rp) - ', &
               'step(rp - Ron) * step(Ron - rp)) * ', &
               '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
               '(Roff2 - Ron2)^3)) * ', &
               '4.0 * epsilon * six * (six - 1.0); ', &
               'six = (sigma / rp)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = rp * rp;', &
               'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
               'rorlambda = step(rlambda)*(r / rlambda);',&
               'rlambda = 2 * 0.2 * ( ',&
               '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
               '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
               'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
          !0.2 is 2A r_0 value in Hayes et al, JPCB, 2017
          
       else if(nbopts%use_vdw_swit .and. .not. use_elec .and. &
            (nbopts%rcut <= nbopts%switchdist)) then                  ! vdW only - switch
          write (formula, '(10a)') &
               '4.0 * epsilon * six * (six - 1.0); ', &
               'six = (sigma / rp)^6; ', &
               'sigma = 0.5*(sigma1 + sigma2); ', &
               'epsilon = sqrt(epsilon1 * epsilon2);', &
               'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
               'rorlambda = step(rlambda)*(r / rlambda);',&
               'rlambda = 2 * 0.2 * ( ',&
               '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
               '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
               'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
          !0.2 is 2A r_0 value in Hayes et al, JPCB, 2017
          
       else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
            (nbopts%rcut > nbopts%switchdist) ) then                               ! vdW only - fswitch
          write (formula, '(34a)') &
               'step(Ron-rp)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6)', &
               '+step(rp-Ron)*step(Roff-rp)*(cr12*rjunk6 - cr6*rjunk3)', &
               '-step(rp-Ron)*step(Ron-rp)*(cr12*rjunk6 - cr6*rjunk3);', &
               'cr6  = ccnbb*ofdif3*rjunk3;', &
               'cr12 = ccnba*ofdif6*rjunk6;', &
               'rjunk3 = r3-recof3;', &
               'rjunk6 = tr6-recof6;', &
               'r3 = r1*tr2;', &
               'r1 = sqrt(tr2);', &
               'tr6 = tr2 * tr2 * tr2;', &
               'tr2 = 1.0/s2;', &
               's2 = rp*rp;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = recof3/on3;', &
               'onoff6 = recof6/on6;', &
               'ofdif3 = off3/(off3 - on3);', &
               'ofdif6 = off6/(off6 - on6);', &
               'recof3 = 1.0/off3;', &
               'on6 = on3*on3;', &
               'on3 = c2onnb*Ron;', &
               'recof6 = 1.0/off6;', &
               'off6 = off3*off3;', &
               'off3 = c2ofnb*Roff;', &
               'c2ofnb = Roff*Roff;', &
               'c2onnb = Ron*Ron;', &
               'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
               'rorlambda = step(rlambda)*(r / rlambda);',&
               'rlambda = 2 * 0.2 * ( ',&
               '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
               '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
               'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
          !0.2 is 2A r_0 value in Hayes et al, JPCB, 2017

          
       else if(nbopts%use_vdw_fswit .and. .not. use_elec .and. &
            (nbopts%rcut <= nbopts%switchdist) ) then                                   ! vdW only - fswitch
          write (formula, '(22a)') &
               'step(Ron-rp)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6);', &
               'tr6 = tr2 * tr2 * tr2;', &
               'tr2 = 1.0/s2;', &
               's2 = rp*rp;', &
               'ccnbb = 4.0*epsilon*sigma^6;', &
               'ccnba = 4.0*epsilon*sigma^12;', &
               'sigma = 0.5*(sigma1+sigma2);', &
               'epsilon = sqrt(epsilon1*epsilon2);', &
               'onoff3 = rcof6;', &
               'onoff6 = rcof6*rcof6;', &
               'rcof6 = 1.0/off6;', &
               'off6 = off3*off3;', &
               'off3 = c2ofnb*Roff;', &
               'c2ofnb = Roff*Roff;', &
               'c2onnb = Ron*Ron;', &
               'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
               'rorlambda = step(rlambda)*(r / rlambda);',&
               'rlambda = 2 * 0.2 * ( ',&
               '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
               '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
               'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
          !0.2 is 2A r_0 value in Hayes et al, JPCB, 2017
       else
          call wrndie(-1,'omm_switching: setup_switchedforce', &
               'Requested combination of cutoff/switching not supported w/BLOCK/softcore')
       endif
       
    endif

  end subroutine get_switchedformula

  subroutine add_switched_atoms(nonbond)
    use omm_util
    use omm_nonbond, only : get_nb_params
    use omm_block, only: blocknumber
    use psf, only: NATOM
    use consta, only : CCELEC
    type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond

    integer :: iatom, ijunk
    real(chm_real) :: charge, sigma, epsilon, charge_scale, atmblock
    type(OpenMM_DoubleArray) :: params

    if( (use_vdw .and. use_elec) .or. (use_vdw .and. use_soft_omm .and. .not. use_elec) ) then
       call OpenMM_DoubleArray_create(params, 3)
    else
       call OpenMM_DoubleArray_create(params, 2)
    endif

    charge_scale = sqrt((CCELEC * OpenMM_KJPerKcal) &
         / (138.935456d0 * OpenMM_AngstromsPerNm))
    do iatom = 1, NATOM
       call get_nb_params(iatom, charge, sigma, epsilon)
       if(use_vdw .and. use_elec .and. .not. use_soft_omm) then 
          call omm_param_set(params, [charge_scale*charge, sigma, epsilon])
       elseif (use_vdw .and. .not. (use_elec .or. use_soft_omm)) then
          call omm_param_set(params, [sigma, epsilon])
       elseif ( (use_vdw .and. use_soft_omm) .and. .not. use_elec ) then
          atmblock = blocknumber(iatom)
          call omm_param_set(params, [sigma, epsilon, atmblock])
       else
          call wrndie(-3, &
                         'ADD_SWITCHED_ATOMS', &
                         'Unsupported options for OpenMM soft core w/ BLOCK')
       endif
       do ijunk = 1, OpenMM_DoubleArray_getSize(params)
          call OpenMM_DoubleArray_get(params,ijunk,atmblock)
       enddo
       ijunk = OpenMM_CustomNonbondedForce_addParticle(nonbond, params)
    enddo

    call OpenMM_DoubleArray_destroy(params)
  end subroutine add_switched_atoms

   !> Sets all topology/residue based non-bond exclusions and 1-4 interactions
  subroutine nbswitch_setup_vdwelec14_interactions(nb14)
    use bases_fcm, only : bnbnd
    use psf, only: NATOM, CG, IAC, MAXATC
    use param, only: NATC, ITC, VDWR, EFF, NBFIXR
    use inbnd, only: E14FAC
    use omm_block, only : blockscale_nbpair
    use omm_nonbond, only : nbfix_scan, nbfix_index, charge_scale
    use omm_util
    
    type(OpenMM_CustomBondForce), intent(inout) :: nb14
    type(OpenMM_DoubleArray) :: params
    integer :: iatom, jatom
    integer :: ikind, jkind
    integer :: itype, jtype
    integer :: inbf, ijunk
    integer :: istrt, iend, ipair
    integer :: iex0
    real(chm_real) :: vdw14, emin14
    real(chm_real) :: charge_prod, sigma, epsln
    real(chm_real) :: fudge, scaleElec, scalevdW
    logical :: nbfix_exists(NATC), doPair

    if(use_vdw .and. use_elec) then
       call OpenMM_DoubleArray_create(params, 3)
    else
       call OpenMM_DoubleArray_create(params, 2)
    endif
    
    call nbfix_scan(nbfix_exists)
    istrt = 1
    do iatom = 1, NATOM
       if (iatom > 1) istrt = max(bnbnd%IBLO14(iatom-1)+1, 1)
       iend = bnbnd%IBLO14(iatom)
       do ipair = istrt, iend
          jatom = abs(bnbnd%INB14(ipair))
          if (bnbnd%INB14(ipair) < 0) then
             ! 1-4 pair
             if(zero_charge) then
                charge_prod = ZERO
             else
                charge_prod = charge_scale() * E14FAC * CG(iatom) * CG(jatom)
             endif
             ikind = IAC(iatom)
             jkind = IAC(jatom)
             inbf = NOT_FOUND
             ! set 1-4 nbfixes here because
             ! CustomNonbondedForce can only exclude
             if (nbfix_exists(ikind) .and. nbfix_exists(jkind)) &
                  inbf = nbfix_index(ikind, jkind)
             if (inbf == NOT_FOUND) then
                itype = ITC(ikind) + MAXATC
                jtype = ITC(jkind) + MAXATC
                vdw14 = (VDWR(itype) + VDWR(jtype)) / 2
                emin14 = sqrt(abs(EFF(itype) * EFF(jtype)))
             else
                ! TODO something different with nbfixes
                vdw14 = NBFIXR(4, inbf) / 2
                emin14 = abs(NBFIXR(3, inbf))
             endif
             if(zero_vdw) then
                sigma = ZERO
                epsln = ZERO
             else
                sigma = vdw14 * OpenMM_SigmaPerVdwRadius / OpenMM_AngstromsPerNm
                epsln = emin14 * OpenMM_KJPerKcal
             endif
             call blockscale_nbpair(iatom, jatom, scaleElec, scalevdW)
             dopair = .true.
             if(use_vdw .and. use_elec) then
                call omm_param_set(params,[scaleElec*charge_prod, sigma, scalevdW*epsln])
                if(scaleElec == ZERO .and. scalevdW == ZERO ) doPair = .false.
                if(prnlev>=6) write(OUTU, &
                     '(a,1x,i4,1x,i4,2x,f8.5,1x,f8.5,1x,f8.5)') &
                     'Exception - nbswitch_setup_vdwelec14..' &
                     ,iatom, jatom, charge_prod, &
                     sigma * OpenMM_AngstromsPerNm, &
                     scalevdW*epsln / OpenMM_KJPerKcal
             else
                if(scalevdW == ZERO) doPair = .false.
                call omm_param_set(params,[sigma, scalevdW*epsln])
                if(prnlev>=6) write(OUTU, &
                     '(a,1x,i4,1x,i4,2x,f8.5,1x,f8.5)') &
                     'Exception - nbswitch_setup_vdwelec14..' &
                     ,iatom, jatom, &
                     sigma * OpenMM_AngstromsPerNm, &
                     scalevdW*epsln / OpenMM_KJPerKcal
             endif
             if(doPair) then
                ijunk = OpenMM_CustomBondForce_addBond(nb14, &
                     iatom-1, jatom-1, params)
             endif
          endif
       enddo
    enddo
    call OpenMM_DoubleArray_destroy(params)
  end subroutine nbswitch_setup_vdwelec14_interactions

  !> scale electrostatic and vdw interaction parameters if block 
  !> is in effect.
  !> Note that at present we assume there are only three blocks
  !> with the environment represent block 1, reactant and product
  !> occupying blocks 2 and 3. We also assume blocks 2 and 3 are
  !> excluded from one another. This routine scales the interactions
  !> block 2 - block 2 and block 3 block 
   subroutine blockscale_nbswitch_intrablock(nonbond, nonbondintra, INB14, IBLO14)
     use omm_block, only : blocknumber, blockscale_nbpair
     use psf, only: NATOM, CG, IAC
     use param, only: NATC, ITC, VDWR, EFF
     use fast, only : LOWTP
     use omm_nonbond, only : nbfix_scan, nbfix_index, charge_scale
     use omm_util
    
     type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
     type(OpenMM_CustomBondForce), intent(inout) :: nonbondintra
     integer, intent(in) :: INB14(:), IBLO14(:)

     type(OpenMM_DoubleArray) :: params
     real(chm_real) :: charge, sigma, well_depth
     integer :: iatom, jatom, itc_i, itc_j, ipair0, block_i, block_j
     real(chm_real) :: scaleElec, scalevdW
     integer :: ibl, jbl, kk
     integer :: istrt, iend, ipair, ijunk
     logical :: Is14orExcl, doPair
     logical :: zero_vdw, zero_charge
     logical :: nbfix_exists(NATC)

#if KEY_BLOCK==1
     ipair0 = 0
     if(use_vdw .and. use_elec) then
        call OpenMM_DoubleArray_create(params, 3)
     else
        call OpenMM_DoubleArray_create(params, 2)
     endif
     
     call nbfix_scan(nbfix_exists)
     istrt = 1
     do iatom = 1, NATOM-1
        block_i = blocknumber(iatom)
        well_depth = ZERO
        sigma = ZERO
        if( block_i > 1 ) then
           itc_i = ITC(IAC(iatom))
           if (zero_vdw .or. itc_i == 0) then
              sigma = ZERO
              well_depth = ZERO
           endif
           do jatom = iatom+1, NATOM
              block_j = blocknumber(jatom)
              if ( block_i == block_j .or. (block_i /= 1 .and. block_j /= 1) ) then

                 Is14orExcl = .false.
                 if(iatom>1) istrt = max(IBLO14(iatom-1)+1, 1)
                 iend = IBLO14(iatom)
                 do ipair = istrt, iend
                    Is14orExcl = Is14orExcl .or. (jatom == abs(INB14(ipair)))
                 enddo
                 if(.not. Is14orExcl) then

                    if(zero_charge) then
                       charge = ZERO
                    else
                       charge = charge_scale() * CG(iatom) * CG(jatom)
                    endif
                    itc_j = ITC(IAC(jatom))
                    if (zero_vdw .or. itc_j == 0) then
                       sigma = ZERO
                       well_depth = ZERO
                    else
                       sigma = ( OpenMM_SigmaPerVdwRadius * &
                            ( VDWR(itc_i) + VDWR(itc_j) ) &
                            / OpenMM_AngstromsPerNm ) / TWO
                       well_depth =  OpenMM_KJPerKcal * &
                            sqrt( EFF(itc_i) * EFF(itc_j) )
                    endif
                    call blockscale_nbpair(iatom, jatom, scaleElec, scalevdW)
                    doPair = .true.
                    if(use_vdw .and. use_elec) then
                       if(scaleElec == ZERO .and. scalevdW == ZERO ) doPair = .false.
                       call omm_param_set(params,[scaleElec*charge, sigma, scalevdW*well_depth])
                    else
                       if(scalevdW == ZERO ) doPair = .false.
                       call omm_param_set(params,[sigma,scalevdW*well_depth])
                    endif
                    if(doPair) then
                       ipair0 = OpenMM_CustomBondForce_addBond(nonbondintra, &
                            iatom-1, jatom-1, params)
                    endif
                    ijunk = OpenMM_CustomNonbondedForce_addExclusion(nonbond, &
                         iatom-1, jatom-1)
                    if(prnlev>6) write(OUTU, &
                         '(a,i4,1x,i4,1x,i4,1x,i4,2x,f5.2,1x,f5.2,1x,f5.2)') &
                         ' Exception - blockscale_nbswitch_intra: ' &
                         , iatom, jatom, block_i, block_j, scaleElec*charge, &
                         sigma*OpenMM_AngstromsPerNm, &
                         scalevdW*well_depth/OpenMM_KJPerKcal
                    if (ipair0 >= MAX_EXCEPTIONS) call wrndie(-3, &
                         'BLOCKSCALE_NBSWITCH_INTRABLOCK', &
                         'Too many nonbonded exceptions for OpenMM')
                 endif
              endif
           enddo
        endif
     enddo
#endif 
     return
   end subroutine blockscale_nbswitch_intrablock
  
#endif 
end module omm_switching

