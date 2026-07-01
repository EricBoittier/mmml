module omm_gbmv
    use chm_kinds
    use number
    use stream, only: OUTU, PRNLEV ! for possible printing
#if KEY_OPENMM==1
    use OpenMM
    use OpenMMGBMV_Types
    use OpenMMGBMV
#endif    
    !! import variables from GBMV
    use gbmv, only: BETA_GBMV2   => Beta, &
                    LAMBDA_GBMV2 => Lambda1, &
                    SLOPE_GBMV2     => SLOPE, &
                    SHIFT_GBMV2     => Shift, &
                    HSX1_GBMV2     => HSX1, &
                    HSX2_GBMV2     => HSX2, &
                    ONX_GBMV2     => ON_X, &
                    OFFX_GBMV2     => OFF_X, &
                    P1_GBMV2     => P1, & 
                    P2_GBMV2     => P2, & 
                    P3_GBMV2     => P3, & 
                    P6_GBMV2     => P6, & 
                    nrad         => NumR, &
                    nang         => nphi_GB, &
                    sgamma       => SA_GB, &
                    kappa        => KAPPA_GB, & 
                    CUTNUM_GBMV2 => CUTNUM 
    
    implicit none
    
    private
    
    logical, save :: &
      qgbmv_import_settings, &
      zero_gbmv, &
      qomm_cutoff
    real*8 :: soluteEPSgbmv, solventEPSgbmv

#if KEY_OPENMM==1
    type(OpenMMGBMV_GBMVForce), save :: gbmvforce
#endif
    
    public :: qgbmv_import_settings, soluteEPSgbmv, solventEPSgbmv, &
              qomm_cutoff
#if KEY_OPENMM==1 
    public :: setup_gbmv, gbmvforce

 contains

    subroutine setup_gbmv(system, group, nbopts)
        use omm_nbopts
        use energym
        
        type(OpenMM_System), intent(inout) :: system
        integer*4, intent(in) :: group
        type(omm_nbopts_t), intent(in) :: nbopts
        
        integer :: nb_method, ijunk
        real*8 :: nb_cutoff, nb_cuton
        real*8 :: box_a(3), box_b(3), box_c(3), box(3)
        
        zero_gbmv = .not. QETERM(GBEnr)
        
        call OpenMMGBMV_GBMVForce_create(gbmvforce)
        call OpenMM_Force_setForceGroup(   &
            transfer(gbmvforce, OpenMM_Force(0)),group)
        
        ! if the "omm gbmv gbon" syntax is used in the CHARMM command line
        ! we need to let the default parameters for these be set by OpenMM
        if (.not. qgbmv_import_settings) then
            
            ! solute dielectric
            call OpenMMGBMV_GBMVForce_setSoluteDielectric(gbmvforce, soluteEPSgbmv)
            
            ! solvent dielectric
            call OpenMMGBMV_GBMVForce_setSolventDielectric(gbmvforce, solventEPSgbmv)
            
            ! surface area coefficient: convert [kcal/mol/Angs**2] to [kJ/mol/nm**2]
            call OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(gbmvforce, &
                sgamma * OpenMM_AngstromsPerNm*OpenMM_AngstromsPerNm*OpenMM_KJPerKcal)
            
            ! Debye-Huckel length: convert [ang] to [nm] ! inverse length (gbmv) vs. length (gbmv)
            if (kappa .ne. 0.D0) then 
            call OpenMMGBMV_GBMVForce_setDebyeHuckelLength(gbmvforce, 1.D0 / kappa / OpenMM_AngstromsPerNm)
            else
            call OpenMMGBMV_GBMVForce_setDebyeHuckelLength(gbmvforce, 0.D0)
            endif
            
            ! number of Lebedev angles and Gaussian-Legendre radii
            call OpenMMGBMV_GBMVForce_setNumLebAng(gbmvforce, nang)
            call OpenMMGBMV_GBMVForce_setNumGauLegRad(gbmvforce, nrad)
            
            ! Parameters for the GBMV2 model
            call OpenMMGBMV_GBMVForce_setBETA_GBMV2(gbmvforce, BETA_GBMV2)
            call OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(gbmvforce, LAMBDA_GBMV2)
            call OpenMMGBMV_GBMVForce_setALPHA_GBMV2(gbmvforce, -1.98D0 * OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(gbmvforce, SLOPE_GBMV2)
            call OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(gbmvforce, SHIFT_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setHSX1_GBMV2(gbmvforce, HSX1_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setHSX2_GBMV2(gbmvforce, HSX2_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setONX_GBMV2(gbmvforce, ONX_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setOFFX_GBMV2(gbmvforce, OFFX_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setP1_GBMV2(gbmvforce, P1_GBMV2 / OpenMM_AngstromsPerNm**2)
            call OpenMMGBMV_GBMVForce_setP2_GBMV2(gbmvforce, P2_GBMV2 / OpenMM_AngstromsPerNm)
            call OpenMMGBMV_GBMVForce_setP3_GBMV2(gbmvforce, P3_GBMV2)
            call OpenMMGBMV_GBMVForce_setP6_GBMV2(gbmvforce, 1.D0/P6_GBMV2)
            call OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(gbmvforce, CUTNUM_GBMV2)
            
        endif
        
        ! GBMV does not support periodic
        ! input cutoff options
        nb_cutoff = nbopts%rcut / OpenMM_AngstromsPerNm
        nb_cuton = nbopts%switchdist / OpenMM_AngstromsPerNm
        
        if (nbopts%periodic) then
            nb_method = OpenMMGBMV_GBMVForce_CutoffPeriodic
            qomm_cutoff = .true.
        else if (nb_cutoff < 99) then
            nb_method = OpenMMGBMV_GBMVForce_CutoffNonPeriodic
            qomm_cutoff = .true.
        else
            nb_method = OpenMMGBMV_GBMVForce_NoCutoff
            qomm_cutoff = .false.
        endif
        call OpenMMGBMV_GBMVForce_setNonbondedMethod(gbmvforce, nb_method)
        call OpenMMGBMV_GBMVForce_setCutoffDistance(gbmvforce, nb_cutoff)
        call OpenMMGBMV_GBMVForce_setCutonDistance(gbmvforce, nb_cuton)
        call OpenMMGBMV_GBMVForce_setReactionFieldDielectric(gbmvforce, nbopts%rf_diel)
        
        ! periodic box bounds (if applicable)
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
        
        ijunk = OpenMM_System_addForce(system, transfer(gbmvforce, OpenMM_Force(0)))
        call gbmv_add_particles(gbmvforce)
        
    end subroutine setup_gbmv

    subroutine gbmv_add_particles(gbmvforce)
        use psf, only: NATOM, CG
        use coord, only: WMAIN
        use coordc, only : WCOMP
        use omm_nonbond, only : charge_scale
        
        type(OpenMMGBMV_GBMVForce), intent(inout) :: gbmvforce
        
        real(chm_real) :: charge, radius, refChargeState1, refChargeState2, &
            chargestate1, chargestate2, fudge, a0, a1, a2, a3, a4, a5, a6, a7, a8, &
            tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp
        integer :: iatom, jatom, I, ijunk, titrateresid, rescounter, &
            istart, iend, aa, bb, tmpint
        character(len=100) :: cphmdOutFile = "output.lamb"//CHAR(0)
        logical :: isCphmdOpened
        
        fudge = sqrt(charge_scale())
        
        ! just add the particles for classic GBMV
        do iatom = 1, natom
            if(.not.zero_gbmv) then
                charge = fudge*cg(iatom)
            else
                charge = ZERO
            endif
            radius = WMAIN(iatom) / OpenMM_AngstromsPerNm
            
            ijunk = OpenMMGBMV_GBMVForce_addParticle( gbmvforce, charge, radius )
        enddo
        
    end subroutine gbmv_add_particles
    
#endif /* KEY_OPENMM */
 end module omm_gbmv
