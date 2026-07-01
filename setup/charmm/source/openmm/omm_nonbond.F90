!> Initializes OpenMM's nonbonded force calculator.
!> Mike Garrahan and Charlie Brooks, 2012
! Algorithm described in Eastman and Pande, JCC Apr 2010, DOI 10.1002/jcc.21413
module omm_nonbond
#if KEY_OPENMM==1
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   use OpenMM

   implicit none

   private

   logical :: zero_charge, zero_vdw
   logical :: use_omm_elec, use_omm_vdw
   integer, parameter :: NOT_FOUND = -1
   integer, parameter :: MAX_EXCEPTIONS = int(1e+7)

   public :: setup_nonbonded, get_nb_params, nbfix_set_exclusions, &
        nbfix_scan, nbfix_index, charge_scale, nb_set_fixes

contains

   !> Adds a nonbonded potential to the system.
  subroutine setup_nonbonded(system, nbopts)
    use omm_nbopts
    use energym
    use bases_fcm, only : BNBND
    use block_ltm, only : QBLOCK
    
    type(OpenMM_System), intent(inout) :: system
    type(omm_nbopts_t), intent(in) :: nbopts

    type(OpenMM_NonbondedForce) :: nonbond

    integer :: ijunk
    real*8 :: nb_cutoff
    
    zero_charge = .not. QETERM(ELEC)
    zero_vdw = .not. QETERM(VDW)

    use_omm_elec = nbopts%use_omm_elec
    use_omm_vdw = nbopts%use_omm_vdw

    if(prnlev>=2) write(outu,'(a)') 'CHARMM> Using OpenMM nb routines.'
    call OpenMM_NonbondedForce_create(nonbond)
    if(use_omm_elec) then
      call OpenMM_NonbondedForce_setReactionFieldDielectric(nonbond, &
        nbopts%rf_diel)
    endif
    
    nb_cutoff = nbopts%rcut / OpenMM_AngstromsPerNm
    if (nbopts%periodic) then
       call setup_periodic(system, nonbond, nb_cutoff, nbopts)
    else
       call setup_nonperiodic(system, nonbond, nb_cutoff, nbopts)
    endif
    ijunk = OpenMM_System_addForce(system, transfer(nonbond, OpenMM_Force(0)))
    
    call nb_add_particles(nonbond)
    if (use_omm_vdw) call nb_set_fixes(system, nbopts)
    call nb_set_exclusions(nonbond, BNBND%INB14, BNBND%IBLO14)
#if KEY_BLOCK==1
    if (qblock) then
       call blockscale_intrablock(nonbond, BNBND%INB14, BNBND%IBLO14)
    endif
#endif 
    ! This is a debug statement. Please leave for the time being.
!    call show_NBParams(nonbond)
  end subroutine setup_nonbonded
  
   !> Returns a factor by which to scale a product of two charges,
   !> or zero if electrostatics are disabled.
   function charge_scale()
      use consta, only: CCELEC
      real(chm_real) :: charge_scale
      ! OpenMM/SimTK internal ONE_4PI_EPS0
      real(chm_real), parameter :: omm_ccelec = 138.935456d0 &
            * OpenMM_AngstromsPerNm / OpenMM_KJPerKcal

      charge_scale = CCELEC / omm_ccelec
   end function charge_scale

   !> Sets charges and vdW parameters for each atom.
   subroutine nb_add_particles(nonbond)
      use psf, only: NATOM

      type(OpenMM_NonbondedForce), intent(inout) :: nonbond

      real(chm_real) :: charge, sigma, epsln
      integer :: iatom, itc_i, ijunk
      real(chm_real) :: fudge

      fudge = sqrt(charge_scale())
      do iatom = 1, NATOM
         call get_nb_params(iatom, charge, sigma, epsln)
         if(use_omm_elec .and. .not. zero_charge) then 
            charge = fudge * charge
         else
            charge = ZERO
         endif
         if(zero_vdw .or. .not. use_omm_vdw) then
            sigma = ZERO
            epsln = ZERO
         endif
         ! "Particle Parameters" here are for single atoms
         ijunk = OpenMM_NonbondedForce_addParticle(nonbond, &
                   charge, sigma, epsln)
      enddo
   end subroutine nb_add_particles

   !> Fetches parameters for a given atom, applying BLOCK coefficients if active
   subroutine get_nb_params(iatom, charge, sigma, epsln)
      use block_ltm, only : QBLOCK
      use psf, only: CG, IAC
      use param, only: ITC, VDWR, EFF
      use omm_block, only : blockscale_nonbond
      use energym

      integer, intent(in) :: iatom
      real(chm_real), intent(out) :: charge, sigma, epsln
      integer :: itc_i

      if(zero_charge) then
         charge = ZERO
      else
         charge = CG(iatom)
      endif

      itc_i = ITC(IAC(iatom))
      if (itc_i == 0 .or. zero_vdw) then
         sigma = ZERO
         epsln = ZERO
      else
         sigma = OpenMM_SigmaPerVdwRadius * VDWR(itc_i) / OpenMM_AngstromsPerNm
         epsln = -OpenMM_KJPerKcal * EFF(itc_i)
      endif
#if KEY_BLOCK==1
      if (qblock) call blockscale_nonbond(iatom, charge, epsln)
#endif
   end subroutine get_nb_params

   !> Sets all topology/residue based non-bond exclusions and 1-4 interactions
   subroutine nb_set_exclusions(nonbond, INB14, IBLO14)
      use psf, only: NATOM, CG, IAC, MAXATC
      use param, only: NATC, ITC, VDWR, EFF, NBFIXR
      use inbnd, only: E14FAC
      use omm_block, only : blockscale_nbpair

      type(OpenMM_NonbondedForce), intent(inout) :: nonbond
      integer, intent(in) :: INB14(:), IBLO14(:)

      integer :: iatom, jatom
      integer :: ikind, jkind
      integer :: itype, jtype
      integer :: inbf, ijunk
      integer :: istrt, iend, ipair
      integer :: iex0
      real(chm_real) :: vdw14, emin14
      real(chm_real) :: charge_prod, sigma, epsln
      real(chm_real) :: scaleElec, scalevdW
      logical :: nbfix_exists(NATC)

      call nbfix_scan(nbfix_exists)
      istrt = 1
      do iatom = 1, NATOM
         if (iatom > 1) istrt = max(IBLO14(iatom-1)+1, 1)
         iend = IBLO14(iatom)
         do ipair = istrt, iend
            jatom = abs(INB14(ipair))
            if (INB14(ipair) < 0) then
               ! 1-4 pair
               if(zero_charge .or. .not. use_omm_elec ) then
                  charge_prod = ZERO
               else 
                  charge_prod = charge_scale() * E14FAC * CG(iatom) * CG(jatom)
               endif
               if (zero_vdw .or. .not. use_omm_vdw) then
                  sigma = ZERO
                  epsln = ZERO
               else
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
                  sigma = vdw14 * OpenMM_SigmaPerVdwRadius / OpenMM_AngstromsPerNm
                  epsln = emin14 * OpenMM_KJPerKcal
               endif
               call blockscale_nbpair(iatom, jatom, scaleElec, scalevdW)
               if(prnlev>=6) write(OUTU, &
                    '(a,1x,i4,1x,i4,2x,f8.5,1x,f8.5,1x,f8.5)') &
                    'Exception - nb_set_exclusions' &
                    ,iatom, jatom, scaleElec*charge_prod, &
                    sigma * OpenMM_AngstromsPerNm, &
                    scalevdW*epsln / OpenMM_KJPerKcal
            else
               ! bond, angle, other exclusion
               charge_prod = ZERO
               sigma = ZERO
               epsln = ZERO
               ! avoid NaN
               scaleElec = ONE
               scalevdW = ONE
            endif
            ijunk = OpenMM_NonbondedForce_addException(nonbond, &
                      iatom-1, jatom-1, scaleElec * charge_prod, sigma, &
                      scalevdW*epsln, OpenMM_TRUE)
            if(prnlev>=6) write(OUTU, '(a,1x,i4,1x,i4)') &
              'Exclusion - nb_set_exclusions' ,iatom, jatom
         enddo
      enddo
    end subroutine nb_set_exclusions

   !> Adds a Force for the difference between the ordinary LJ potential
   !> and any LJ interactions having NBFIXes.
   ! XXX not BLOCK aware
   subroutine nb_set_fixes(system, nbopts)
      use block_ltm, only : QBLOCK
      use omm_nbopts
      use param, only: NATC
      use omm_block, only : getlambda
      type(OpenMM_System), intent(inout) :: system
      type(omm_nbopts_t), intent(in) :: nbopts
      type(OpenMM_CustomNonbondedForce) :: nbfixes
      real*8  :: r_on, r_off, lambda
      logical :: nbfix_exists(NATC)
      logical :: nbfixedpairs
      integer :: ntab, ijunk
      character(len=1500) :: formula

      call nbfix_scan(nbfix_exists)
      ! Commenting this call out as it involves an N^2 loop
       call nbfix_inpsf(nbfixedpairs,nbfix_exists, natc)
      ! number of atom types subject to any nbfix
      ! XXX there may be no atoms of those types
      ntab = count(nbfix_exists)
      if (ntab == 0 .or. .not. nbfixedpairs) return
      !if (ntab == 0) return

      call get_nbfswitchedformula(nbopts,ntab,formula)
      r_on = nbopts%switchdist / OpenMM_AngstromsPerNm
      r_off = nbopts%rcut / OpenMM_AngstromsPerNm
      call OpenMM_CustomNonbondedForce_create(nbfixes, trim(formula))
      if( ( nbopts%use_vdw_swit .and. (nbopts%rcut > nbopts%switchdist) ) .or.    &   ! vdW only - switch
           ( nbopts%use_vdw_fswit ) ) then                                            ! vdW only - fswitch
         ijunk = OpenMM_CustomNonbondedForce_addGlobalParameter( &
              nbfixes, 'Ron', r_on)
         ijunk = OpenMM_CustomNonbondedForce_addGlobalParameter( &
              nbfixes, 'Roff', r_off)
      endif
      if(nbopts%use_soft_omm) then
         lambda = getlambda()
         ijunk = OpenMM_CustomNonbondedForce_addGlobalParameter( &
              nbfixes, 'lambda', lambda)
         if(PRNLEV >= 7) write(OUTU, '(2a, i4)') &
              'CHARMM> global parameter lambda added to ', &
              'OpenMM CustomNonbondedForce at index ', &
              ijunk
      endif

      call nbfix_set_cutoff(nbfixes, nbopts)
      call nbfix_build_tables(nbfixes, nbfix_exists, ntab)

      ijunk = OpenMM_System_addForce(system, transfer(nbfixes, OpenMM_Force(0)))
      if (PRNLEV >= 2) write (OUTU, '(a)') &
            'NBFIxes in effect, using CustomNonbondedForce'
      call OpenMM_CustomNonbondedForce_getEnergyFunction(nbfixes,formula)
      if(PRNLEV >= 6) write(OUTU, '(a, x, a, /, a)') &
           'CHARMM> an OpenMM force will be created for NBFIxed interactions', &
           'using the following OpenMM compatible formula', &
           trim(formula)

      call nbfix_add_atoms(nbfixes, nbfix_exists, nbopts)
      call nbfix_set_exclusions(nbfixes)
      ! Check whetehr there are intrablock nonbonded interactions
      ! that aren't 1-4 or bonded and exclude
#if KEY_BLOCK==1
      if(qblock) call block_intrablock_nbfixexcl(nbfixes)  
#endif

   end subroutine nb_set_fixes

   subroutine nbfix_set_cutoff(nbfixes, nbopts)
      use omm_nbopts
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nbfixes
      type(omm_nbopts_t), intent(in) :: nbopts
      integer*4 :: nb_method
      real*8 :: nb_cutoff

      nb_cutoff = nbopts%rcut / OpenMM_AngstromsPerNm
      if (nbopts%periodic) then
         nb_method = OpenMM_CustomNonbondedForce_CutoffPeriodic
      else if (nb_cutoff < 99) then
         nb_method = OpenMM_CustomNonbondedForce_CutoffNonPeriodic
      else
         nb_method = OpenMM_CustomNonbondedForce_NoCutoff
      endif
      call OpenMM_CustomNonbondedForce_setNonbondedMethod(nbfixes, nb_method)
!      if (nb_cutoff < 99) then
         call OpenMM_CustomNonbondedForce_setCutoffDistance(nbfixes, nb_cutoff)
!      endif
   end subroutine nbfix_set_cutoff

   !> Tabulates nbfix sigma and epsilon as functions of (ikind, jkind).
   ! Technique suggested by Peter Eastman, Oct 2012
   subroutine nbfix_build_tables(nbfixes, nbfix_exists, ntab)
      use omm_util
      use param, only: NATC, NBFIXR
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nbfixes
      logical, intent(in) :: nbfix_exists(NATC)
      integer, intent(in) :: ntab
      integer :: identity(NATC)
      integer :: iac_kind(ntab)
      integer :: ntab2, itab, jtab, lindex
      integer :: ikind, jkind, inbf, ijunk
      real(chm_real), dimension((ntab*(ntab-1))/2+ntab+1) :: sigma, epsln, has_nbf
      real(chm_real) :: vdw, emin
      real*8 :: fmin, fmax
      type(OpenMM_DoubleArray) :: params
      type(OpenMM_Discrete1DFunction) :: resultd


      ! e.g. nbfix_exists [F,T,F,F,T] -> iac_kind [2,5]
      do ikind = 1, NATC
         identity(ikind) = ikind
      enddo
      iac_kind = pack(identity, nbfix_exists)

      sigma = ZERO
      epsln = ZERO
      has_nbf = ZERO
      if (.not. zero_vdw) then
         do itab = 1, ntab
            ikind = iac_kind(itab)
            do jtab = itab, ntab
               jkind = iac_kind(jtab)
               inbf = nbfix_index(ikind, jkind)
               if (inbf == NOT_FOUND) cycle
               vdw = NBFIXR(2, inbf) / 2
               emin = abs(NBFIXR(1, inbf))
               lindex = ( itab - 1 ) * ntab - ( itab * ( itab - 1 ) ) / 2 + jtab + 1
               sigma(lindex) = vdw * OpenMM_SigmaPerVdwRadius / OpenMM_AngstromsPerNm
               epsln(lindex) = emin * OpenMM_KJPerKcal
               has_nbf(lindex) = ONE
            enddo
         enddo
      endif

      ntab2 = ( ntab * (ntab-1) )/ 2 + ntab + 1
      fmin = ONE
      fmax = ntab2 + 1
      call OpenMM_DoubleArray_create(params, ntab2)

      call omm_param_set(params, sigma)
      call OpenMM_Discrete1DFunction_create(resultd, params)      
      ijunk = OpenMM_CustomNonbondedForce_addTabulatedFunction(nbfixes, 'nbfSig', &
           transfer(resultd,OpenMM_TabulatedFunction(0)))

      call omm_param_set(params, epsln)
      call OpenMM_Discrete1DFunction_create(resultd, params)      
      ijunk = OpenMM_CustomNonbondedForce_addTabulatedFunction(nbfixes, 'nbfEps', &
           transfer(resultd,OpenMM_TabulatedFunction(0)))

      call omm_param_set(params, has_nbf)
      call OpenMM_Discrete1DFunction_create(resultd, params)  
      ijunk = OpenMM_CustomNonbondedForce_addTabulatedFunction(nbfixes, 'hasNbf', &
           transfer(resultd,OpenMM_TabulatedFunction(0)))

      call OpenMM_DoubleArray_destroy(params)
   end subroutine nbfix_build_tables

   subroutine nbfix_add_atoms(nbfixes, nbfix_exists, nbopts)
      use omm_util
      use psf, only: NATOM, IAC
      use param, only: NATC
      use omm_nbopts
      use omm_block, only: blocknumber
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nbfixes
      type(omm_nbopts_t), intent(in) :: nbopts
      logical, intent(in) :: nbfix_exists(NATC)
      integer :: tab_index(NATC)
      integer :: ikind, itab, iatom, ijunk, iend
      real(chm_real) :: charge, sigma, epsln, atmblock
      real(chm_real) :: pval(4)
      type(OpenMM_DoubleArray) :: params

      iend = 3
      ! e.g. nbfix_exists [F,T,F,F,T] -> tab_index [0,1,0,0,2]
      itab = 0
      tab_index = 0
      do ikind = 1, NATC
         if (nbfix_exists(ikind)) then
            itab = itab + 1
            tab_index(ikind) = itab
         endif
      enddo

      ijunk = OpenMM_CustomNonbondedForce_addPerParticleParameter(nbfixes, 'tab')
      ijunk = OpenMM_CustomNonbondedForce_addPerParticleParameter(nbfixes, 'ljSig')
      ijunk = OpenMM_CustomNonbondedForce_addPerParticleParameter(nbfixes, 'ljEps')
      if(nbopts%use_soft_omm) then
         ijunk = OpenMM_CustomNonbondedForce_addPerParticleParameter(nbfixes, 'block')
         iend = 4
         call OpenMM_DoubleArray_create(params, 4)
      else
         call OpenMM_DoubleArray_create(params, 3)
      endif

      do iatom = 1, NATOM
         ikind = IAC(iatom)
         if (nbfix_exists(ikind)) then
            atmblock = blocknumber(iatom)
            call get_nb_params(iatom, charge, sigma, epsln)
            pval(1) = tab_index(ikind)
            pval(2) = sigma
            pval(3) = epsln
            pval(4) = atmblock
         else
            pval = ZERO
         endif
            call omm_param_set(params, pval(1:iend))
            ijunk = OpenMM_CustomNonbondedForce_addParticle(nbfixes, params)
      enddo
      call OpenMM_DoubleArray_destroy(params)
   end subroutine nbfix_add_atoms

   subroutine nbfix_set_exclusions(nbfixes)
      use bases_fcm, only: BNBND
      use psf, only: NATOM
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nbfixes
       integer :: istrt, iend, ipair, iatom, jatom, ijunk
      istrt = 1
      do iatom = 1, NATOM
         ! XXX extract a NB list iterator
         if (iatom > 1) istrt = max(BNBND%IBLO14(iatom-1)+1, 1)
         iend = BNBND%IBLO14(iatom)
         do ipair = istrt, iend
            jatom = abs(BNBND%INB14(ipair))
            ijunk = OpenMM_CustomNonbondedForce_addExclusion(nbfixes, &
                 iatom-1, jatom-1)
            if(prnlev>6) write(OUTU, '(a,i4,1x,i4)') &
                 ' Exclusion - nbfix_set_exclusions ', iatom, jatom
         enddo
      enddo
   end subroutine nbfix_set_exclusions

   ! This routine determines whether any of the atom pairs in the PSF are 
   ! subject to NBFIxed interactions. It does a scan of all pairs
   subroutine nbfix_inpsf(nbfixedpairs,nbfix_exists,natc)
      use psf, only: NATOM, IAC

      logical :: nbfix_exists(NATC)
      logical :: nbfixedpairs
      integer :: natc
      integer :: iatom, jatom, itype, jtype

      nbfixedpairs = .false.
      do iatom=1, NATOM-1
         itype = IAC(iatom)
         if(nbfix_exists(itype)) then
            do jatom=iatom+1, NATOM
               jtype = IAC(jatom)
               if(nbfix_index(itype,jtype)>0) then
                  nbfixedpairs = .true.
                  exit
               endif
            enddo
         endif
      enddo
!      do iatom=1, NATOM
!         itype = IAC(iatom)
!         if(nbfix_exists(itype)) then
!            nbfixedpairs = .true.
!            exit
!         endif
!      enddo
      return
    end subroutine nbfix_inpsf

   subroutine nbfix_scan(nbfix_exists)
      use param, only: NBFIXN, NBFIXI
      logical, intent(out) :: nbfix_exists(:)
      integer :: inbf, anbf(2)

      nbfix_exists = .false.
      do inbf = 1, NBFIXN
         anbf = NBFIXI(:, inbf)
         nbfix_exists(anbf(1)) = .true.
         nbfix_exists(anbf(2)) = .true.
      enddo
   end subroutine nbfix_scan

   integer function nbfix_index(ikind, jkind)
      use param, only: NBFIXN, NBFIXI
      integer, intent(in) :: ikind, jkind
      integer :: inbf, nbf_pair(2), pair_ij(2), pair_ji(2)

      nbfix_index = NOT_FOUND
      pair_ij = [ikind, jkind]
      pair_ji = [jkind, ikind]
      do inbf = 1, NBFIXN
         nbf_pair = NBFIXI(:, inbf)
         if (all(nbf_pair == pair_ij) .or. all(nbf_pair == pair_ji)) then
            nbfix_index = inbf
            exit
         endif
      enddo
   end function nbfix_index

   subroutine get_nbfswitchedformula(nbopts, ntab, formula)
     use omm_nbopts
     type(omm_nbopts_t), intent(in) :: nbopts
     character(*), intent(inout) :: formula
     integer, intent(in) :: ntab

     if(.not. nbopts%use_soft_omm) then
        if( nbopts%use_vdw_ommswit .and. (nbopts%rcut > nbopts%switchdist) ) then    ! OpenMM vdW switch
           write (formula, '(11a,i0,2a)') &
                '(step(Ron - r) + ', &
                'step(r - Ron) * step(Roff - r) * ', &
                'S ) * hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / r) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / r) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2); ', &
                'S=1-6*x^5+15*x^4-10*x^3;x = (r - Ron)/(Roff-Ron); k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
        
        else if( nbopts%use_vdw_swit .and. (nbopts%rcut > nbopts%switchdist) ) then    ! vdW only - switch
           write (formula, '(13a,i0,2a)') &
                '(step(Ron - r) + ', &
                'step(r - Ron) * step(Roff - r) * ', &
                '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
                '(Roff2 - Ron2)^3)) * ', &
                'hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / r) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / r) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2);', &
                'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = r * r; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
           
        else if( nbopts%use_vdw_swit .and. (nbopts%rcut <= nbopts%switchdist) ) then  ! vdW only - switch
           write (formula, '(8a,i0,2a)') &
                'hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / r) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / r) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2); k = k - delta( i * j ) * k; ', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
           
        else if( (nbopts%use_vdw_fswit .or. nbopts%use_vdw_fswit) &
             .and. (nbopts%rcut > nbopts%switchdist) ) then   ! vdW only - fswitch
           write (formula, '(36a,i0,2a)') &
                'step(Ron-r)*hasNbf(k)*( ', &
                '(ccnbaF*tr6*tr6-ccnbbF*tr6+ccnbbF*onoff3-ccnbaF*onoff6) -', &
                '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) )', &
                '+step(r-Ron)*step(Roff-r)*hasNbf(k)*( ', &
                '(cr12nbf*rjunk6 - cr6nbf*rjunk3) -', &
                '(cr12*rjunk6 - cr6*rjunk3) );', &
                'cr6  = ccnbb*ofdif3*rjunk3;', &
                'cr12 = ccnba*ofdif6*rjunk6;', &
                'cr6nbf  = ccnbbF*ofdif3*rjunk3;', &
                'cr12nbf = ccnbaF*ofdif6*rjunk6;', &
                'rjunk3 = r3-recof3;', &
                'rjunk6 = tr6-recof6;', &
                'r3 = r1*tr2;', &
                'r1 = sqrt(tr2);', &
                'tr6 = tr2 * tr2 * tr2;', &
                'tr2 = 1.0/s2;', &
                's2 = r*r;', &
                'ccnbbF = 4.0*nbfEps(k)*nbfSig(k)^6;', &
                'ccnbaF = 4.0*nbfEps(k)*nbfSig(k)^12;', &
                'ccnbb = 4.0*epsilon*sigma^6;', &
                'ccnba = 4.0*epsilon*sigma^12;', &
                'sigma = 0.5*(ljSig1+ljSig2);', &
                'epsilon = sqrt(ljEps1*ljEps2);', &
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
                'c2onnb = Ron*Ron; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
           
        else if( nbopts%use_vdw_fswit .and. (nbopts%rcut <= nbopts%switchdist) ) then ! vdW only - fswitch
           write (formula, '(20a,i0,2a)') &
                'step(Ron-r)*hasNbf(k)*(', &
                '(ccnbaF*tr6*tr6-ccnbbF*tr6+ccnbbF*onoff3-ccnbaF*onoff6) - ', &
                '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) );', &
                'tr6 = tr2 * tr2 * tr2;', &
                'tr2 = 1.0/s2;', &
                's2 = r*r;', &
                'ccnbbF = 4.0*nbfEps(k)*nbfSig(k)^6;', &
                'ccnbaF = 4.0*nbfEps(k)*nbfSig(k)^12;', &
                'ccnbb = 4.0*epsilon*sigma^6;', &
                'ccnba = 4.0*epsilon*sigma^12;', &
                'sigma = 0.5*(ljSig1+ljSig2);', &
                'epsilon = sqrt(ljEps1*ljEps2);', &
                'onoff3 = rcof6;', &
                'onoff6 = rcof6*rcof6;', &
                'rcof6 = 1.0/off6;', &
                'off6 = off3*off3;', &
                'off3 = c2ofnb*Roff;', &
                'c2ofnb = Roff*Roff;', &
                'c2onnb = Ron*Ron; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
           
        else if( .not. (nbopts%use_vdw_swit .or. nbopts%use_vdw_fswit .or. nbopts%use_vdw_ommswit) ) then  !vdW normal nbfix potential
           write (formula, '(8a,i0,2a)') &
                'hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / r) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / r) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2); k = k - delta( i * j ) * k;', & 
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2)'
        
        else
           call wrndie(-1,'omm_switching: get_nbfswitchedformula', &
                'Requested combination of cutoff/switching not supported')
        endif
        
     elseif(nbopts%use_soft_omm) then
        if( nbopts%use_vdw_swit .and. (nbopts%rcut > nbopts%switchdist) ) then    ! vdW only - switch
           write (formula, '(13a,i0,8a)') &
                '(step(Ron - rp) + ', &
                'step(rp - Ron) * step(Roff - rp) * ', &
                '(((Roff2 - r2)^2 * (Roff2 + 2.0 * r2 - 3.0 * Ron2)) / ', &
                '(Roff2 - Ron2)^3)) * ', &
                'hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / rp) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / rp) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2);', &
                'Ron2 = Ron * Ron; Roff2 = Roff * Roff; r2 = rp * rp; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2);', &
               'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
               'rorlambda = step(rlambda)*(r / rlambda);',&
               'rlambda = 2 * 0.2 * ( ',&
               '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
               '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
               'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'

                
           
        else if( nbopts%use_vdw_swit .and. (nbopts%rcut <= nbopts%switchdist) ) then  ! vdW only - switch
           write (formula, '(8a,i0,8a)') &
                'hasNbf(k) * (nbf - lj); ', &
                'nbf = 4 * nbfEps(k) * (nsr6^2 - nsr6); ', &
                'nsr6 = (nbfSig(k) / rp) ^ 6; ', &
                'lj = 4 * eps * (lsr6^2 - lsr6); ', &
                'lsr6 = (sig / rp) ^ 6; ', &
                'sig = (ljSig1 + ljSig2) / 2; ', &
                'eps = sqrt(ljEps1 * ljEps2); k = k - delta( i * j ) * k; ', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2);', &
                'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
                'rorlambda = step(rlambda)*(r / rlambda);',&
                'rlambda = 2 * 0.2 * ( ',&
                '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
                '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
                'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
                
           
        else if( (nbopts%use_vdw_fswit .or. nbopts%use_vdw_fswit) &
             .and. (nbopts%rcut > nbopts%switchdist) ) then   ! vdW only - fswitch
           write (formula, '(36a,i0,8a)') &
                'step(Ron-rp)*hasNbf(k)*( ', &
                '(ccnbaF*tr6*tr6-ccnbbF*tr6+ccnbbF*onoff3-ccnbaF*onoff6) -', &
                '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) )', &
                '+step(rp-Ron)*step(Roff-rp)*hasNbf(k)*( ', &
                '(cr12nbf*rjunk6 - cr6nbf*rjunk3) -', &
                '(cr12*rjunk6 - cr6*rjunk3) );', &
                'cr6  = ccnbb*ofdif3*rjunk3;', &
                'cr12 = ccnba*ofdif6*rjunk6;', &
                'cr6nbf  = ccnbbF*ofdif3*rjunk3;', &
                'cr12nbf = ccnbaF*ofdif6*rjunk6;', &
                'rjunk3 = r3-recof3;', &
                'rjunk6 = tr6-recof6;', &
                'r3 = r1*tr2;', &
                'r1 = sqrt(tr2);', &
                'tr6 = tr2 * tr2 * tr2;', &
                'tr2 = 1.0/s2;', &
                's2 = rp*rp;', &
                'ccnbbF = 4.0*nbfEps(k)*nbfSig(k)^6;', &
                'ccnbaF = 4.0*nbfEps(k)*nbfSig(k)^12;', &
                'ccnbb = 4.0*epsilon*sigma^6;', &
                'ccnba = 4.0*epsilon*sigma^12;', &
                'sigma = 0.5*(ljSig1+ljSig2);', &
                'epsilon = sqrt(ljEps1*ljEps2);', &
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
                'c2onnb = Ron*Ron; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2);', &
                'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
                'rorlambda = step(rlambda)*(r / rlambda);',&
                'rlambda = 2 * 0.2 * ( ',&
                '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
                '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
                'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
           
        else if( nbopts%use_vdw_fswit .and. (nbopts%rcut <= nbopts%switchdist) ) then ! vdW only - fswitch
           write (formula, '(20a,i0,8a)') &
                'step(Ron-rp)*hasNbf(k)*(', &
                '(ccnbaF*tr6*tr6-ccnbbF*tr6+ccnbbF*onoff3-ccnbaF*onoff6) - ', &
                '(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) );', &
                'tr6 = tr2 * tr2 * tr2;', &
                'tr2 = 1.0/s2;', &
                's2 = rp*rp;', &
                'ccnbbF = 4.0*nbfEps(k)*nbfSig(k)^6;', &
                'ccnbaF = 4.0*nbfEps(k)*nbfSig(k)^12;', &
                'ccnbb = 4.0*epsilon*sigma^6;', &
                'ccnba = 4.0*epsilon*sigma^12;', &
                'sigma = 0.5*(ljSig1+ljSig2);', &
                'epsilon = sqrt(ljEps1*ljEps2);', &
                'onoff3 = rcof6;', &
                'onoff6 = rcof6*rcof6;', &
                'rcof6 = 1.0/off6;', &
                'off6 = off3*off3;', &
                'off3 = c2ofnb*Roff;', &
                'c2ofnb = Roff*Roff;', &
                'c2onnb = Ron*Ron; k = k - delta( i * j ) * k;', &
                'k = ( i - 1 ) * ', ntab,' - ( i * ( i - 1 ) ) / 2 + j; ',  &
                'j = max(tab1, tab2); i = min(tab1, tab2);', &
                'rp = select(step(rlambda-r),rlambda * ( 0.5+rorlambda^3-0.5*rorlambda^4 ),r);', &
                'rorlambda = step(rlambda)*(r / rlambda);',&
                'rlambda = 2 * 0.2 * ( ',&
                '(delta(block1-1)*delta(block2-2)+ delta(block1-2)*delta(block2-1))*lambda +',&
                '(delta(block1-1)*delta(block2-3)+ delta(block1-3)*delta(block2-1))*(1-lambda) +',&
                'delta(block1-2)*delta(block2-2)*lambda + delta(block1-3)*delta(block2-3)*(1-lambda));'
        else
           call wrndie(-1,'omm_switching: get_nbfswitchedformula', &
                'Requested combination of cutoff/switching BLOCK softcore not supported')
        endif
     endif
         
   end subroutine get_nbfswitchedformula
   
   subroutine setup_nonperiodic(system, nonbond, rcut, nbopts)
      use omm_nbopts
      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_NonbondedForce), intent(inout) :: nonbond

      real*8, intent(in) :: rcut
      type(omm_nbopts_t), intent(in) :: nbopts
      integer*4 :: nb_method
      integer*4 :: using_lrc, using_vdw_ommswit
      real*8 :: diel_rf

      if (rcut >= 99) then
         nb_method = OpenMM_NonbondedForce_NoCutoff
         if (PRNLEV >= 2) write (OUTU, '(X,A)') &
               'Informational: OpenMM Using No Cutoff (cut>=990).'
      else
         nb_method = OpenMM_NonbondedForce_CutoffNonPeriodic
         if (PRNLEV >= 6) write (OUTU, '(X,A,2(/,10X,A))') &
               'Warning: Using OpenMM Cutoff method.  This differs from', &
               'CHARMM in that a reaction field approximation is used', &
               'for the region outside the cutoff.'
      endif
      
      using_lrc = OpenMM_False
      if (nbopts%use_lrc) using_lrc = OpenMM_True

      using_vdw_ommswit = OpenMM_False
      if(nbopts%use_vdw_ommswit .and. &
           (nbopts%rcut > nbopts%switchdist)) &
           using_lrc = OpenMM_True

      call OpenMM_NonbondedForce_setNonbondedMethod(nonbond, nb_method)
      call OpenMM_NonbondedForce_setCutoffDistance(nonbond, rcut)
      diel_rf = OpenMM_NonbondedForce_getReactionFieldDielectric(nonbond)
      call OpenMM_NonbondedForce_setUseDispersionCorrection(nonbond, using_lrc)
      if (nbopts%use_vdw_ommswit .and.(nbopts%rcut > nbopts%switchdist)) then
        call OpenMM_NonbondedForce_setUseSwitchingFunction(nonbond, &
               using_vdw_ommswit)
        call OpenMM_NonbondedForce_setSwitchingDistance(nonbond, &
               nbopts%switchdist / OpenMM_AngstromsPerNm )
      endif

      if ((nb_method == OpenMM_NonbondedForce_CutoffNonPeriodic) .and. &
           (PRNLEV >= 5)) &
           write (OUTU, '(10X,A,F8.3)') &
           'Reaction field dielectric constant = ', diel_rf
   end subroutine setup_nonperiodic

   subroutine setup_periodic(system, nonbond, rcut, nbopts)
     use omm_nbopts, only: omm_nbopts_t
     implicit none

     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_NonbondedForce), intent(inout) :: nonbond

     real*8, intent(in) :: rcut
     type(omm_nbopts_t), intent(in) :: nbopts

     character(len=1024) :: env_string
     integer*4 :: nb_method
     real*8 :: rfd_const, openmm_tol, mesh_scale, omm_kappa
     real*8 :: box_a(3), box_b(3), box_c(3), box(3)
     integer*4 :: using_lrc, using_vdw_ommswit

     if (nbopts%use_pme) then
       nb_method = OpenMM_NonbondedForce_PME
       if (PRNLEV >= 7) write (OUTU, '(X,A,/,10X,A)') &
         'Warning: Using the OpenMM PME method.  See the PME section', &
         'of the OpenMM documentation for details.'
     else if (nbopts%use_ewald) then
       nb_method = OpenMM_NonbondedForce_Ewald
       if (PRNLEV >= 7) write (OUTU, '(X,A,/,10X,A)') &
         'Warning: Using the OpenMM Ewald method.  See the Ewald section', &
         'of the OpenMM documentation for details.'
     else !if(nbopts%use_elec_rxnfld) then
       nb_method = OpenMM_NonbondedForce_CutoffPeriodic
       if (PRNLEV >= 7) write (OUTU, '(X,A,2(/,10X,A))') &
         'Warning: Using OpenMM Cutoff method.  This differs from', &
         'CHARMM in that a reaction field approximation is used', &
         'for the region outside the cutoff.'
         if (PRNLEV >= 7) then
           rfd_const = OpenMM_NonbondedForce_getReactionFieldDielectric(nonbond)
           write (OUTU, '(10X,A,F8.3)') &
             'CHARMM> Reaction field dielectric constant = ',  rfd_const
         end if
     end if
      
     box = nbopts%box / OpenMM_AngstromsPerNm
     box_a = ZERO
     box_b = ZERO
     box_c = ZERO
     box_a(1) = box(1)
     box_b(2) = box(2)
     box_c(3) = box(3)
     call OpenMM_System_setDefaultPeriodicBoxVectors(system, &
            box_a, box_b, box_c)

     omm_kappa = nbopts%alpha * OpenMM_AngstromsPerNm

     using_lrc = OpenMM_False
     if (nbopts%use_lrc) using_lrc = OpenMM_True

     using_vdw_ommswit = OpenMM_False
     if(nbopts%use_vdw_ommswit .and. (nbopts%rcut > nbopts%switchdist) ) &
       using_vdw_ommswit = OpenMM_True

     call OpenMM_NonbondedForce_setNonbondedMethod(nonbond, nb_method)
     call OpenMM_NonbondedForce_setCutoffDistance(nonbond, rcut)
     call OpenMM_NonbondedForce_setUseDispersionCorrection(nonbond, using_lrc)

     if (nbopts%use_vdw_ommswit .and. (nbopts%rcut > nbopts%switchdist)) then
       call OpenMM_NonbondedForce_setUseSwitchingFunction(nonbond, &
              using_vdw_ommswit)
       call OpenMM_NonbondedForce_setSwitchingDistance(nonbond, &
              nbopts%switchdist / OpenMM_AngstromsPerNm )
     end if

     if (nbopts%use_pme) then
       if (prnlev >= 2) write(OUTU, '(a,/,a,x,f10.4,/,a,x,i4,x,i4,x,i4,/)') &
         'CHARMM> configuring OpenMM PME with', &
         '        alpha', omm_kappa / OpenMM_AngstromsPerNM, &
         '        fft', nbopts%mesh_dim(1), &
         nbopts%mesh_dim(2), nbopts%mesh_dim(3)

       call OpenMM_NonbondedForce_setPMEParameters(nonbond, omm_kappa, &
              nbopts%mesh_dim(1), nbopts%mesh_dim(2), nbopts%mesh_dim(3))

       if (prnlev >= 7) then
         openmm_tol = est_pme_tol(box, rcut, nbopts%mesh_dim)
         write (OUTU,'(a,x,f10.4,/)') 'OpenMM Error Tolerance is', openmm_tol
       end if
     else if (nbopts%use_ewald) then
       openmm_tol = est_ewald_tol(box, rcut)
       call OpenMM_NonbondedForce_setEwaldErrorTolerance(nonbond, openmm_tol)

       if (prnlev >= 7) write (OUTU,'(a,x,f10.4,/)') &
         'OpenMM Error Tolerance is', openmm_tol

     else
       return
     end if
   end subroutine setup_periodic

   ! Estimates a PME error tolerance (dimensionless)
   ! for OpenMM based on NFFT values from CHARMM input.
   ! Derived from NonbondedForceImpl::calcPMEParameters.
   real*8 function est_pme_tol(box, rcut, mesh_dim)
      real*8, intent(in) :: box(3)
      real*8, intent(in) :: rcut
      integer, intent(in) :: mesh_dim(3)

      real*8 :: nmesh_in(3), tol_out(3)
      real*8 :: tol0, tol1, tol2, nmesh0, nmesh1, nmesh2
      integer :: i, idim

      ! need to hit interval (NFFT-1, NFFT), calcPMEParameters rounds up
      nmesh_in = mesh_dim - HALF

      do idim = 1, 3
         tol0 = 1.0d-7
         tol1 = 1.0d-1
         nmesh0 = calc_nmesh(box(idim), tol0)
         nmesh1 = calc_nmesh(box(idim), tol1)
         if (.not. goal_between(nmesh_in(idim), nmesh0, nmesh1)) then
            if(prnlev>=2) write (OUTU, '(A, F8.3)') &
                  ' est_pme_tol: cannot obtain NFFT = ', nmesh_in(idim)
            tol_out(idim) = 3.0d-4
            cycle
         endif

         ! bisection
         do i = 1, 20
            tol2 = sqrt(tol0 * tol1)
            nmesh2 = calc_nmesh(box(idim), tol2)
            if (abs(nmesh2 - nmesh_in(idim)) < HALF) exit
            if (goal_between(nmesh_in(idim), nmesh1, nmesh2)) then
               tol0 = tol2
               nmesh0 = nmesh2
            else
               tol1 = tol2
               nmesh1 = nmesh2
            endif
         enddo
         tol_out(idim) = tol2
      enddo

      est_pme_tol = minval(tol_out)

   contains

      logical function goal_between(goal, val1, val2)
         real*8, intent(in) :: goal, val1, val2
         goal_between = (val1 - goal) * (val2 - goal) < 0
      end function goal_between

      real*8 function calc_nmesh(rbox, tol)
         real*8, intent(in) :: rbox, tol
         real*8 :: alpha, nmesh

         alpha = sqrt(-log(TWO * tol)) / rcut
         nmesh = TWO/THREE * alpha * rbox / tol**0.2
!        if(prnlev>=5) write (OUTU, '(A, ES10.3, A, F8.3, A, F8.3)') &
!              ' est_pme_tol: tol = ', tol, ', alpha = ', alpha, ', nmesh = ', nmesh
         calc_nmesh = nmesh
      end function calc_nmesh

   end function est_pme_tol

   ! Estimates an Ewald error tolerance (dimensionless)
   ! for OpenMM based on KMAX value from CHARMM input.
   ! Derived from NonbondedForceImpl::calcEwaldParameters.
   real*8 function est_ewald_tol(box, rcut)
      use ewald, only: KMAX

      real*8, intent(in) :: box(3)
      real*8, intent(in) :: rcut
      real*8 :: tol_out(3)
      real*8 :: tol0, tol1, tol2
      integer :: kmax_in, kmax0, kmax1, kmax2
      integer :: i, idim

      kmax_in = KMAX

      do idim = 1, 3
         tol0 = 1.0d-7
         tol1 = 1.0d-1
         kmax0 = calc_kmax(box(idim), tol0)
         kmax1 = calc_kmax(box(idim), tol1)
         if (.not. goal_between(kmax_in, kmax0, kmax1)) then
            if(prnlev>=2) write (OUTU, '(A, I0)') &
                  ' est_ewald_tol: cannot obtain KMAX = ', kmax_in
            tol_out(idim) = 3.0d-4
            cycle
         endif

         ! bisection
         do i = 1, 20
            tol2 = sqrt(tol0 * tol1)
            kmax2 = calc_kmax(box(idim), tol2)
            if (kmax2 == kmax_in) exit
            if (goal_between(kmax_in, kmax1, kmax2)) then
               tol0 = tol2
               kmax0 = kmax2
            else
               tol1 = tol2
               kmax1 = kmax2
            endif
         enddo
         tol_out(idim) = tol2
      enddo

      est_ewald_tol = product(tol_out) ** THIRD

   contains

      logical function goal_between(goal, val1, val2)
         integer, intent(in) :: goal, val1, val2
         goal_between = (val1 - goal) * (val2 - goal) < 0
      end function goal_between

      integer function calc_kmax(rbox, tol)
         real*8, intent(in) :: rbox, tol
         real*8 :: alpha, ewe
         integer :: k

         alpha = sqrt(-log(TWO * tol)) / rcut
         k = 10
         ! based on NonbondedForceImpl::findZero
         ewe = ewald_err(rbox, alpha, tol, k)
         if (ewe > ZERO) then
            do while (ewe > ZERO .and. k > 0)
               k = k - 1
               ewe = ewald_err(rbox, alpha, tol, k)
            enddo
            k = k + 1
         else
            do while (ewe < ZERO)
               k = k + 1
               ewe = ewald_err(rbox, alpha, tol, k)
            enddo
         endif
!        if(prnlev>=5) write (OUTU, '(A, F9.3, A, ES12.3, A, I0)') &
!              ' est_ewald_tol: alpha = ', alpha, ', tol = ', tol, ', k = ', k
         calc_kmax = k
      end function calc_kmax

      ! based on NonbondedForceImpl::EwaldErrorFunction
      real*8 function ewald_err(rbox, alpha, tol, k)
         use consta, only: PI
         real*8, intent(in) :: rbox, alpha, tol
         integer, intent(in) :: k
         real*8 :: tmp

         tmp = k * PI / (rbox * alpha)
         ewald_err = tol - 0.05 * sqrt(rbox * alpha) * k * exp(-tmp**2)
      end function ewald_err

   end function est_ewald_tol

  !> scale electrostatic and vdw interaction parameters if block 
  !> is in effect.
  !> Note that at present we assume there are only three blocks
  !> with the environment represent block 1, reactant and product
  !> occupying blocks 2 and 3. We also assume blocks 2 and 3 are
  !> excluded from one another. This routine scales the interactions
  !> block 2 - block 2 and block 3 block 
   subroutine blockscale_intrablock(nonbond, INB14, IBLO14)
     use omm_block, only : blocknumber, blockscale_nbpair
     use psf, only: NATOM, CG, IAC
     use param, only: ITC, VDWR, EFF
     use fast, only : LOWTP
     implicit none

     type(OpenMM_NonbondedForce), intent(inout) :: nonbond
     integer, intent(in) :: INB14(:), IBLO14(:)

     real(chm_real) :: charge, sigma, well_depth
     integer :: iatom, jatom, itc_i, itc_j, ipair0, block_i, block_j
     real(chm_real) :: scaleElec, scalevdW
     integer :: ibl, jbl, kk
     integer :: istrt, iend, ipair
     logical :: Is14orExcl
     
#if KEY_BLOCK==1
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

                    if(zero_charge .or. .not. use_omm_elec) then
                       charge = ZERO
                    else
                       charge = charge_scale() * CG(iatom) * CG(jatom)
                    endif
                    itc_j = ITC(IAC(jatom))
                    if (zero_vdw .or. itc_j == 0 .or. .not. use_omm_vdw) then
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

                    ipair0 = OpenMM_NonbondedForce_addException(nonbond, &
                               iatom-1, jatom-1, scaleElec*charge,  &
                               sigma, scalevdW*well_depth, OpenMM_TRUE)

                    if(prnlev>6) write(OUTU, &
                         '(a,i4,1x,i4,1x,i4,1x,i4,2x,f5.2,1x,f5.2,1x,f5.2)') &
                         ' Exception - blockscale_intrablock: ' &
                         , iatom, jatom, block_i, block_j, scaleElec*charge, &
                         sigma*OpenMM_AngstromsPerNm, &
                         scalevdW*well_depth/OpenMM_KJPerKcal
                    if (ipair0 >= MAX_EXCEPTIONS) call wrndie(-3, &
                         'BLOCKSCALE_INTRABLOCK', &
                         'Too many nonbonded exceptions for OpenMM')
                 endif
              endif
           enddo
        endif
     enddo
#endif 
     return
   end subroutine blockscale_intrablock

  !> scale electrostatic and vdw interaction parameters if block 
  !> is in effect.
  !> Note that at present we assume there are only three blocks
  !> with the environment represent block 1, reactant and product
  !> occupying blocks 2 and 3. We also assume blocks 2 and 3 are
  !> excluded from one another. This routine scales the interactions
  !> block 2 - block 2 and block 3 block 
   subroutine block_intrablock_nbfixexcl(nonbond)
     use bases_fcm, only: BNBND
     use omm_block, only : blocknumber
     use psf, only: NATOM
     
     type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond

     integer :: iatom, jatom, ipair0, block_i, block_j
     integer :: ibl, jbl, kk
     integer :: istrt, iend, ipair
     logical :: Is14orExcl
     
#if KEY_BLOCK==1
     istrt = 1
     do iatom = 1, NATOM-1
        block_i = blocknumber(iatom)
        if( block_i > 1 ) then 
           do jatom = iatom+1, NATOM
              block_j = blocknumber(jatom)
              if( ( block_i == block_j ) .or. (block_i /= 1 .and. block_j /= 1) ) then

                 Is14orExcl = .false.
                 if(iatom>1) istrt = max(BNBND%IBLO14(iatom-1)+1, 1)
                 iend = BNBND%IBLO14(iatom)
                 do ipair = istrt, iend
                    Is14orExcl = Is14orExcl .or. (jatom == abs(BNBND%INB14(ipair)))
                 enddo
                 if(.not. Is14orExcl) then

                    ipair0 = OpenMM_CustomNonbondedForce_addExclusion(nonbond, &
                          iatom-1, jatom-1 )
                    if(prnlev>6) write(OUTU, &
                         '(a,i4,1x,i4,1x,i4,1x,i4,2x,f5.2,1x,f5.2,1x,f5.2)') &
                         ' Exclusion - block_intrablock_nbfixexcl: ' &
                         , iatom, jatom, block_i, block_j
                    if (ipair0 >= MAX_EXCEPTIONS) call wrndie(-3, &
                         'BLOCK_INTRABLOCK_NBFIXEXCL', &
                         'Too many nonbonded exceptions for OpenMM')
                 endif
              endif
           enddo
        endif
     enddo
#endif 
     return
   end subroutine block_intrablock_nbfixexcl

   subroutine show_NBParams(nonbond)
     use psf, only: NATOM
     implicit none

     type(OpenMM_NonbondedForce), intent(in) :: nonbond

     integer :: i
     real(chm_real) :: charge, sigma, epsilon

     do i = 1, NATOM
       call OpenMM_NonbondedForce_getParticleParameters(nonbond, i - 1, &
              charge, sigma, epsilon)
        if (prnlev>6) write(OUTU, '(i4,2x,f5.2,1x,f5.2,1x,f5.2)') i, charge,  &
          sigma*OpenMM_AngstromsPerNm, epsilon/OpenMM_KJPerKcal
     enddo
     return
   end subroutine show_NBParams
#endif 
 end module omm_nonbond
