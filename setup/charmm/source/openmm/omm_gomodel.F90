!> Implements a Go-like model as described in
!> Karanicolas and Brooks, Prot Sci 2002, DOI 10.1110/ps.0205402
!> Assumes no charges
module omm_gomodel
#if KEY_OPENMM==1
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   use OpenMM
   use omm_nbopts
   implicit none

   private

   integer, allocatable :: atom_of_type(:)
   logical :: zero_elec, zero_vdw
   character(len=2), parameter :: box_param(3) = ['bx', 'by', 'bz']

   public :: setup_gomodel

contains

   subroutine setup_gomodel(system, nbopts)
      use energym
      use param, only: NATC
      type(omm_nbopts_t), intent(in) :: nbopts
      type(OpenMM_System), intent(inout) :: system

      zero_elec = .not. QETERM(ELEC)
      zero_vdw = .not. QETERM(VDW)
      allocate(atom_of_type(NATC))
      call map_types_to_atoms()
      if (nbopts%periodic) call setup_system_periodic(system, nbopts)
      call setup_go_nonbond(system, nbopts)
      call setup_go_bonded(system, nbopts)
      deallocate(atom_of_type)
   end subroutine setup_gomodel

   subroutine setup_system_periodic(system, nbopts)
      type(omm_nbopts_t), intent(in) :: nbopts
      type(OpenMM_System), intent(inout) :: system
      real*8 :: box_a(3), box_b(3), box_c(3), box(3)

      box = nbopts%box / OpenMM_AngstromsPerNm
      box_a = ZERO
      box_b = ZERO
      box_c = ZERO
      box_a(1) = box(1)
      box_b(2) = box(2)
      box_c(3) = box(3)
      call OpenMM_System_setDefaultPeriodicBoxVectors(system, &
            box_a, box_b, box_c)
   end subroutine setup_system_periodic

   subroutine map_types_to_atoms()
      use psf, only: NATOM, IAC
      use param, only: NATC, ITC
      character(len=*), parameter :: procname = 'map_types_to_atoms'
      integer :: iatom, itype

      if (NATOM /= NATC) then
         write (OUTU, '(a,i0,a,i0)') &
               'NATOM = ', NATOM, ', NATC = ', NATC
         call wrndie(-1, procname, 'Atoms and types not in 1-1 correspondence')
      endif

      atom_of_type = 0
      do iatom = 1, NATOM
         itype = ITC(IAC(iatom))
         if (atom_of_type(itype) == 0) then
            atom_of_type(itype) = iatom
         else
            write (OUTU, '(a,i0,a,i0,a)') &
                  'Atoms ', atom_of_type(itype), ' and ', iatom, ' have same type'
            call wrndie(-1, procname, 'Atom types are not unique')
         endif
      enddo
   end subroutine map_types_to_atoms

   !> Uses CustomNonbondedForce for general pairs,
   !> excluding bonds, angles, and pairs having nbfixes.
   subroutine setup_go_nonbond(system, nbopts)
      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_CustomNonbondedForce) :: nonbond
      character(len=200) :: formula
      integer :: iforce, iparm
      type(omm_nbopts_t), intent(in) :: nbopts
      real*8 :: rcut

      rcut = nbopts%rcut / OpenMM_AngstromsPerNm

      write (formula, '(2a)') &
            'sqrt(epsilon1 * epsilon2) * (13*sr^12 - 18*sr^10 + 4*sr^6); ', &
            'sr = (rmin1 + rmin2) / (2 * r)'
      call OpenMM_CustomNonbondedForce_create(nonbond, formula)

      if (nbopts%periodic) then
          call setup_nonbond_periodic(nonbond, rcut)
      else
          call setup_nonbond_nonperiodic(nonbond, rcut)
      endif

      iparm = OpenMM_CustomNonbondedForce_addPerParticleParameter(nonbond, &
                    'rmin')
      iparm = OpenMM_CustomNonbondedForce_addPerParticleParameter(nonbond, &
                    'epsilon')

      call go_add_atoms(nonbond)
      call go_exclude_bonds(nonbond)
      call go_exclude_nbfixes(nonbond)
      iforce = OpenMM_System_addForce(system, &
                    transfer(nonbond, OpenMM_Force(0)))
   end subroutine setup_go_nonbond

   subroutine setup_nonbond_periodic(nonbond, rcut)
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
      real*8, intent(in) :: rcut

      if (PRNLEV >= 5) write (OUTU, '(X,A)') &
           'Informational: OpenMM will use Cutoff'

      call OpenMM_CustomNonbondedForce_setNonbondedMethod(nonbond, &
                OpenMM_CustomNonbondedForce_CutoffPeriodic)
      call OpenMM_CustomNonbondedForce_setCutoffDistance(nonbond, rcut)
   end subroutine setup_nonbond_periodic

   subroutine setup_nonbond_nonperiodic(nonbond, rcut)
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
      real*8, intent(in) :: rcut
      integer*4 :: nb_method

      if (rcut >= 99) then
         nb_method = OpenMM_CustomNonbondedForce_NoCutoff
         if (PRNLEV >= 2) write (OUTU, '(X,A)') &
               'Informational: OpenMM Using No Cutoff (cut>=999).'
      else
         nb_method = OpenMM_CustomNonbondedForce_CutoffNonPeriodic
         if (PRNLEV >= 5) write (OUTU, '(X,A)') &
               'Informational: OpenMM will use Cutoff'
      endif

      call OpenMM_CustomNonbondedForce_setNonbondedMethod(nonbond, nb_method)
      if (nb_method == OpenMM_CustomNonbondedForce_CutoffNonPeriodic) &
         call OpenMM_CustomNonbondedForce_setCutoffDistance(nonbond, rcut)
   end subroutine setup_nonbond_nonperiodic

   subroutine go_add_atoms(nonbond)
      use omm_util
      use omm_nonbond
      use psf, only: NATOM
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
      integer :: ikind, iatom, ijunk
      real(chm_real) :: charge, sigma, epsln, rmin
      type(OpenMM_DoubleArray) :: params

      call OpenMM_DoubleArray_create(params, 2)
      do iatom = 1, NATOM
         call get_nb_params(iatom, charge, sigma, epsln)
         if (charge /= ZERO .and. .not. zero_elec) then
            write (OUTU, '(a,i0,a,f5.1,a)') &
                  'Atom ', iatom, ' charge = ', charge, ' ignored'
            call wrndie(0, 'go_add_atoms', &
                  'Charged particles in Go model not supported with OpenMM')
         endif
         rmin = 2 * sigma / OpenMM_SigmaPerVdwRadius
         call omm_param_set(params, [rmin, epsln])
         ijunk = OpenMM_CustomNonbondedForce_addParticle(nonbond, params)
      enddo
      call OpenMM_DoubleArray_destroy(params)
   end subroutine go_add_atoms

   subroutine go_exclude_bonds(nonbond)
      use bases_fcm, only: BNBND
      use psf, only: NATOM
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
      integer :: istrt, iend, ipair, iatom, jatom, iexcl

      istrt = 1
      do iatom = 1, NATOM
         if (iatom > 1) istrt = max(BNBND%IBLO14(iatom-1)+1, 1)
         iend = BNBND%IBLO14(iatom)
         do ipair = istrt, iend
            jatom = abs(BNBND%INB14(ipair))
            iexcl = OpenMM_CustomNonbondedForce_addExclusion(nonbond, &
                  iatom-1, jatom-1)
         enddo
      enddo
   end subroutine go_exclude_bonds

   subroutine go_exclude_nbfixes(nonbond)
      use param, only: NBFIXN, NBFIXI
      type(OpenMM_CustomNonbondedForce), intent(inout) :: nonbond
      integer :: inbf, iatom, jatom, ipair

      do inbf = 1, NBFIXN
         iatom = atom_of_type(NBFIXI(1, inbf))
         jatom = atom_of_type(NBFIXI(2, inbf))
         ipair = OpenMM_CustomNonbondedForce_addExclusion(nonbond, &
               jatom-1, iatom-1)
      enddo
   end subroutine go_exclude_nbfixes

   !> Uses CustomCompoundBondForce for nbfixed particle pairs.
   !> Assumes that each nbfix probably applies to a single pair.
   subroutine setup_go_bonded(system, nbopts)
      use param, only: NBFIXN
      type(OpenMM_System), intent(inout) :: system
      type(omm_nbopts_t), intent(in) :: nbopts

      real*8 :: rcut 
      rcut = nbopts%rcut / OpenMM_AngstromsPerNm

      if(nbopts%periodic) then
        call setup_bonded_periodic(system, rcut)
      else
        call setup_bonded_nonperiodic(system, rcut)
      endif
   end subroutine setup_go_bonded

   subroutine setup_bonded_periodic(system, rcut)
      type(OpenMM_System), intent(inout) :: system
      real*8, intent(in) :: rcut

      type(OpenMM_CustomCompoundBondForce) :: gobonded
      character(len=400) :: formula
      integer :: iforce, iparm, i
      real*8 :: box_a(3), box_b(3), box_c(3), box(3)

      if (PRNLEV >= 5) write (OUTU, '(X,A)') &
               'Informational: OpenMM will use Cutoff w/ PBCs'
      write (formula, '(5a)') &
           'step(cutoff^2-pr2) * epsilon * (13*sr^6 - 18*sr^5 + 4*sr^3); ', &
           'sr = rmin^2 / pr2; pr2 = dx^2 + dy^2 + dz^2; ', &
           'dx = dx - bx*ceil(dx/bx-0.5); dx = x2 - x1; ', &
           'dy = dy - by*ceil(dy/by-0.5); dy = y2 - y1; ', &
           'dz = dz - bz*ceil(dz/bz-0.5); dz = z2 - z1'

      call OpenMM_CustomCompoundBondForce_create(gobonded, 2, formula)
      iparm = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
                gobonded, 'rmin')
      iparm = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
                gobonded, 'epsilon')
      iparm = OpenMM_CustomCompoundBondForce_addGlobalParameter(gobonded, &
                    'cutoff', rcut)

      ! assumes rectangular box aligned with coordinate axes
      call OpenMM_System_getDefaultPeriodicBoxVectors(system, &
                box_a, box_b, box_c)
      box = [box_a(1), box_b(2), box_c(3)]
      ! each Force must add its own references to global Context parameters
      do i = 1, 3
        iparm = OpenMM_CustomCompoundBondForce_addGlobalParameter(gobonded, &
                    box_param(i), abs(box(i)))
      enddo

      call go_add_nbfixes_periodic(gobonded)
      iforce = OpenMM_System_addForce(system, &
                transfer(gobonded, OpenMM_Force(0)))
   end subroutine setup_bonded_periodic

   subroutine setup_bonded_nonperiodic(system, rcut)
      type(OpenMM_System), intent(inout) :: system
      real*8, intent(in) :: rcut

      type(OpenMM_CustomBondForce) :: bonded

      character(len=200) :: formula
      integer :: iforce, iparm

      if (rcut < 99) then
      write (formula, '(2a)') &
        'step(cutoff - r) * epsilon * (13 * sr^12 - 18 * sr^10 + 4 * sr^6); ', &
        'sr = rmin / r'
      else
        if (PRNLEV >= 2) write (OUTU, '(X,A)') &
          'Informational: OpenMM Using No Cutoff (cut>=999).'
        write (formula, '(2a)') &
          'epsilon * (13 * sr^12 - 18 * sr^10 + 4 * sr^6); ', &
          'sr = rmin / r'
      endif

      call OpenMM_CustomBondForce_create(bonded, formula)

      if (rcut < 99) then
        iparm = OpenMM_CustomBondForce_addGlobalParameter(bonded, &
                    'cutoff', rcut)
      endif

      iparm = OpenMM_CustomBondForce_addPerBondParameter(bonded, &
                    'rmin')
      iparm = OpenMM_CustomBondForce_addPerBondParameter(bonded, &
                    'epsilon')
      call go_add_nbfixes_nonperiodic(bonded)
      iforce = OpenMM_System_addForce(system, &
                    transfer(bonded, OpenMM_Force(0)))
   end subroutine setup_bonded_nonperiodic

   subroutine go_add_nbfixes_periodic(gobonded)
      use omm_util
      use param, only: NBFIXN, NBFIXI, NBFIXR
      type(OpenMM_CustomCompoundBondForce), intent(inout) :: gobonded
      real(chm_real) :: vdwr, emin, rmin, epsln
      type(OpenMM_DoubleArray) :: params
      type(OpenMM_IntArray) :: particle_pair
      integer :: inbf, iatom, jatom, ipair

      call OpenMM_DoubleArray_create(params, 2)
      call OpenMM_IntArray_create(particle_pair, 2)

      do inbf = 1, NBFIXN
         vdwr = NBFIXR(2, inbf) / 2
         emin = abs(NBFIXR(1, inbf))
         if (zero_vdw) then
            rmin = ZERO
            epsln = ZERO
         else
            rmin = 2 * vdwr / OpenMM_AngstromsPerNm
            epsln = emin * OpenMM_KJPerKcal
         endif

         call omm_param_set(params, [rmin, epsln])

         iatom = atom_of_type(NBFIXI(1, inbf))
         jatom = atom_of_type(NBFIXI(2, inbf))
         call OpenMM_IntArray_set(particle_pair, 1, jatom - 1)

         call OpenMM_IntArray_set(particle_pair, 2, iatom - 1)

         ipair = OpenMM_CustomCompoundBondForce_addBond(gobonded, &
                    particle_pair, params)
      enddo
      call OpenMM_DoubleArray_destroy(params)
      call OpenMM_IntArray_destroy(particle_pair)
   end subroutine go_add_nbfixes_periodic

   subroutine go_add_nbfixes_nonperiodic(bonded)
      use omm_util
      use param, only: NBFIXN, NBFIXI, NBFIXR
      type(OpenMM_CustomBondForce), intent(inout) :: bonded
      real(chm_real) :: vdwr, emin, rmin, epsln
      type(OpenMM_DoubleArray) :: params
      integer :: inbf, iatom, jatom, ipair

      call OpenMM_DoubleArray_create(params, 2)
      do inbf = 1, NBFIXN
         vdwr = NBFIXR(2, inbf) / 2
         emin = abs(NBFIXR(1, inbf))
         if (zero_vdw) then
            rmin = ZERO
            epsln = ZERO
         else
            rmin = 2 * vdwr / OpenMM_AngstromsPerNm
            epsln = emin * OpenMM_KJPerKcal
         endif

         call omm_param_set(params, [rmin, epsln])

         iatom = atom_of_type(NBFIXI(1, inbf))
         jatom = atom_of_type(NBFIXI(2, inbf))
         ipair = OpenMM_CustomBondForce_addBond(bonded, iatom -1, jatom - 1, &
                    params)
      enddo
      call OpenMM_DoubleArray_destroy(params)
   end subroutine go_add_nbfixes_nonperiodic
#endif 
end module omm_gomodel

