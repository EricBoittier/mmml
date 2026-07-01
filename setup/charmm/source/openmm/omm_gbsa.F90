module omm_gbsa
#if KEY_OPENMM==1
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   use OpenMM
   implicit none

   private

   logical :: qgbsa, zero_gbsaobc
   real*8 :: soluteEPS, solventEPS

   public :: qgbsa, soluteEPS, solventEPS, setup_gbsaobc2

 contains

    subroutine setup_gbsaobc2(system, group, nbopts)
      use omm_nbopts
      use energym

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(omm_nbopts_t), intent(in) :: nbopts

      type(OpenMM_GBSAOBCForce) :: gbsaobc
      integer :: nb_method, ijunk
      real*8 :: nb_cutoff
      real*8 :: box_a(3), box_b(3), box_c(3), box(3)

      zero_gbsaobc = .not. QETERM(GBEnr)

      call OpenMM_GBSAOBCForce_create(gbsaobc)
       call OpenMM_Force_setForceGroup(   &
           transfer(gbsaobc, OpenMM_Force(0)),group)
     call OpenMM_GBSAOBCForce_setSoluteDielectric(gbsaobc, soluteEPS)
      call OpenMM_GBSAOBCForce_setSolventDielectric(gbsaobc, solventEPS)

      nb_cutoff = nbopts%rcut / OpenMM_AngstromsPerNm

      if (nbopts%periodic) then
         nb_method = OpenMM_GBSAOBCForce_CutoffPeriodic
      else if (nb_cutoff < 99) then
         nb_method = OpenMM_GBSAOBCForce_CutoffNonPeriodic
      else
         nb_method = OpenMM_GBSAOBCForce_NoCutoff
      endif
      call OpenMM_GBSAOBCForce_setNonbondedMethod(gbsaobc, nb_method)
      call OpenMM_GBSAOBCForce_setCutoffDistance(gbsaobc, nb_cutoff)

      if (nbopts%periodic) then
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

      ijunk = OpenMM_System_addForce(system, transfer(gbsaobc, &
           OpenMM_Force(0)))

      call gbsa_add_particles(gbsaobc)

    end subroutine setup_gbsaobc2

    subroutine gbsa_add_particles(gbsaforce)
      use psf, only: NATOM, CG
      use coord, only: WMAIN
      use coordc, only : WCOMP
      use omm_nonbond, only : charge_scale

      type(OpenMM_GBSAOBCForce), intent(inout) :: gbsaforce

      real(chm_real) :: charge, radius, scaleFactor
      integer :: iatom, ijunk
      real(chm_real) :: fudge

      fudge = sqrt(charge_scale())

      do iatom = 1, natom
         if(.not.zero_gbsaobc) then
            charge = fudge*cg(iatom)
         else
            charge = ZERO
         endif
         radius = WMAIN(iatom) / OpenMM_AngstromsPerNm
         scaleFactor = WCOMP(iatom)

         ijunk = OpenMM_GBSAOBCForce_addParticle( gbsaforce, &
              charge, radius, scaleFactor )

      enddo
    end subroutine gbsa_add_particles

#endif
 end module omm_gbsa
