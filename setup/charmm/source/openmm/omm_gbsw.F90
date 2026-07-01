module omm_gbsw
    use chm_kinds
    use number
    use stream, only: OUTU, PRNLEV
#if KEY_OPENMM==1
    use OpenMM
    use OpenMMGBSW_Types
    use OpenMMGBSW
#endif    
    !! import variables from GBSW
    use gbsw, only: nang, nrad, sw, aa0, aa1, sgamma, kappa, tmembgb, msw
    
    !! import variables from PHMD
#if KEY_PHMD==1
    use phmd, only: qphmd, grplist, TitrRes, qstate1, qstate2, residuePKA, &
        pH_PHMD, Temp_PHMD, QMass_PHMD, NTitr, ParA, ParB, Barr, SP_PAR, &
        ParMod, SP_GRP, PHBETA, NPrint_PHMD, TFile_PHMD
#endif
    
    !! import 1-4 nonbonded interactions (PHMD needs these)
    use bases_fcm, only : BNBND
    
    !! grab the timestep for lambda dynamics in CPHMD
    use reawri, only: TIMEST
    
    implicit none
    
    private
    
    logical, save :: &
      qphmd_omm = .false., &
      qphmd_initialized = .false., &
      qgbsw_import_settings, &
      zero_gbsw, &
      qomm_cutoff
    real*8 :: soluteEPSgbsw, solventEPSgbsw

#if KEY_OPENMM==1
    type(OpenMMGBSW_GBSWForce), save :: gbswforce
#endif
    
    public :: qgbsw_import_settings, soluteEPSgbsw, solventEPSgbsw, &
              qphmd_omm, qphmd_initialized, qomm_cutoff
#if KEY_OPENMM==1 && KEY_PHMD==1
    public :: setup_gbsw, gbswforce

 contains

    subroutine setup_gbsw(system, group, nbopts)
        use omm_nbopts
        use energym
        
        type(OpenMM_System), intent(inout) :: system
        integer*4, intent(in) :: group
        type(omm_nbopts_t), intent(in) :: nbopts
        
        integer :: nb_method, ijunk
        real*8 :: nb_cutoff
        real*8 :: box_a(3), box_b(3), box_c(3), box(3)
        
        zero_gbsw = .not. QETERM(GBEnr)
        
        call OpenMMGBSW_GBSWForce_create(gbswforce)
        call OpenMM_Force_setForceGroup(   &
            transfer(gbswforce, OpenMM_Force(0)),group)
        
        ! if the "omm gbsw gbon" syntax is used in the CHARMM command line
        ! we need to let the default parameters for these be set by OpenMM
        if (.not. qgbsw_import_settings) then
            
            ! solute dielectric
            call OpenMMGBSW_GBSWForce_setSoluteDielectric(gbswforce, soluteEPSgbsw)
            
            ! solvent dielectric
            call OpenMMGBSW_GBSWForce_setSolventDielectric(gbswforce, solventEPSgbsw)
            
            ! switching function length: convert [ang] to [nm]
            call OpenMMGBSW_GBSWForce_setSwitchingLength(gbswforce, sw / OpenMM_AngstromsPerNm)
            
            ! set GBSW delta-G0 and delta-G1 constants AA0 and AA1
            call OpenMMGBSW_GBSWForce_setAA0(gbswforce, aa0)
            call OpenMMGBSW_GBSWForce_setAA1(gbswforce, aa1)
            
            ! surface area coefficient: convert [kcal/mol/Angs**2] to [kJ/mol/nm**2]
            call OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(gbswforce, &
                sgamma * OpenMM_AngstromsPerNm*OpenMM_AngstromsPerNm*OpenMM_KJPerKcal)
            
            ! Debye-Huckel length: convert [ang] to [nm]
            call OpenMMGBSW_GBSWForce_setDebyeHuckelLength(gbswforce, kappa / OpenMM_AngstromsPerNm)
            
            ! number of Lebedev angles and Gaussian-Legendre radii
            call OpenMMGBSW_GBSWForce_setNumLebAng(gbswforce, nang)
            call OpenMMGBSW_GBSWForce_setNumGauLegRad(gbswforce, nrad)
            
            ! membrane function about the XY-plane
            if (tmembgb > 0.0) then
                call OpenMMGBSW_GBSWForce_setMembraneThickness(gbswforce, &
                    tmembgb / OpenMM_AngstromsPerNm)
                call OpenMMGBSW_GBSWForce_setMembraneSwLen(gbswforce, &
                    msw / OpenMM_AngstromsPerNm)
            endif
            
        endif
        
        ! input cutoff options
        nb_cutoff = nbopts%rcut / OpenMM_AngstromsPerNm
        
        if (nbopts%periodic) then
            nb_method = OpenMMGBSW_GBSWForce_CutoffPeriodic
            qomm_cutoff = .true.
        else if (nb_cutoff < 99) then
            nb_method = OpenMMGBSW_GBSWForce_CutoffNonPeriodic
            qomm_cutoff = .true.
        else
            nb_method = OpenMMGBSW_GBSWForce_NoCutoff
            qomm_cutoff = .false.
        endif
        call OpenMMGBSW_GBSWForce_setNonbondedMethod(gbswforce, nb_method)
        call OpenMMGBSW_GBSWForce_setCutoffDistance(gbswforce, nb_cutoff)
        call OpenMMGBSW_GBSWForce_setReactionFieldDielectric(gbswforce, nbopts%rf_diel)
        
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
        
        ijunk = OpenMM_System_addForce(system, transfer(gbswforce, OpenMM_Force(0)))
        call gbsw_add_particles(gbswforce)
        
    end subroutine setup_gbsw

    subroutine gbsw_add_particles(gbswforce)
        use psf, only: NATOM, CG
        use coord, only: WMAIN
        use coordc, only : WCOMP
        use omm_nonbond, only : charge_scale
        
        type(OpenMMGBSW_GBSWForce), intent(inout) :: gbswforce
        
        real(chm_real) :: charge, radius, refChargeState1, refChargeState2, &
            chargestate1, chargestate2, fudge, a0, a1, a2, a3, a4, a5, a6, a7, a8, &
            tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp
        integer :: iatom, jatom, I, ijunk, titrateresid, rescounter, &
            istart, iend, aa, bb, tmpint
        character(len=100) :: cphmdOutFile = "output.lamb"//CHAR(0)
        logical :: isCphmdOpened
        
        fudge = sqrt(charge_scale())
        
        ! if CPHMD is being used, give particles alternate charge states
        if (qphmd_omm) then
            
            ! manage the output file: find its name, and close it if it's open
            inquire( unit=TFile_PHMD, opened=isCphmdOpened )
            if (isCphmdOpened) then
                inquire( unit=TFile_PHMD, name=cphmdOutFile )
                close(TFile_PHMD)
                I = len(trim(cphmdOutFile)) + 1
                cphmdOutFile(I:I) = CHAR(0) ! don't forget the null character for C
            endif
            
            ! send CPHMD flag to OpenMM
            call OpenMMGBSW_GBSWForce_addCPHMDForce( gbswforce, &
                pH_PHMD, Temp_PHMD, QMass_PHMD, TIMEST, PHBETA, NPrint_PHMD, cphmdOutFile )
            
            ! add particles, and send alternate charge states to GBSW
            ! instead of 4 charge states, OpenMM uses a reference state
            ! with up to 2 alternate charge states
            rescounter = 0
            do iatom = 1, natom
                if(.not.zero_gbsw) then
                    charge = fudge*cg(iatom)
                else
                    charge = ZERO
                endif
                radius = WMAIN(iatom) / OpenMM_AngstromsPerNm
                tmpint = grplist(iatom)
                titrateresid = TitrRes(tmpint)
                
                refChargeState1 = qstate1(1, iatom) * fudge
                refChargeState2 = qstate1(2, iatom) * fudge
                chargestate1    = qstate2(1, iatom) * fudge
                chargestate2    = qstate2(2, iatom) * fudge
                
                ijunk = OpenMMGBSW_GBSWForce_addParticle( gbswforce, &
                    charge, radius )
                ijunk = OpenMMGBSW_GBSWForce_addCphmdParameters( gbswforce, &
                    titrateresid, refChargeState1, refChargeState2, &
                    chargestate1, chargestate2 )
            enddo
            
            ! send titrating group information to OpenMM.
            ! rather than have a residue-specific formalism for each model 
            ! potential Umod, we use the general-case eq. 17 from the paper
            ! 'Constant pH Molecular Dynamics with Proton Tautomerism'
            ! here we untangle the inputs to fit that standard format
            
            ! a0 = lambda^2 * x^2 term
            ! a1 = lambda^2 * x   term
            ! a2 = lambda   * x^2 term
            ! a3 = lambda   * x   term
            ! a4 = lambda^2       term
            ! a5 =            x^2 term
            ! a6 = lambda         term
            ! a7 =            x   term
            ! a8 =                term
            
            rescounter = 0
            I = 1
            do while ( I <= NTitr )
                
                if ( SP_GRP(I) == 0 ) then
                    
                    ! single-site, no tautomeric x-coordinate (i.e. LYS)
                    a0 = 0.0
                    a1 = 0.0
                    a2 = 0.0
                    a3 = 0.0
                    a4 = ParA(I)
                    a5 = 0.0
                    a6 = -2.0 * ParA(I) * ParB(I)
                    a7 = 0.0
                    a8 = ParA(I) * ParB(I) * ParB(I)
                    rescounter = rescounter + 1
                    
                    tmp = 0.0
                    ijunk = OpenMMGBSW_GBSWForce_addTitratingGroupParameters( gbswforce, &
                        residuePKA(I), residuePKA(I), tmp, Barr(I), &
                        a0, a1, a2, a3, a4, a5, a6, a7, a8 )
                    
                elseif ( SP_GRP(I) == 2 ) then
                    
                    ! 2-site residue where pKa1 .ne. pKa2 (i.e. HSP)
                    tmp1 = -2.0 * ParA(I) * ParB(I)
                    tmp2 = -2.0 * ParA(I-1) * ParB(I-1)
                    tmp3 = -2.0 * SP_PAR(1,I) * SP_PAR(2,I)
                    
                    a0 = SP_PAR(1,I)
                    a1 = tmp3 - tmp2 + tmp1
                    a2 = 0.0
                    a3 = tmp2 - tmp1
                    a4 = ParA(I)
                    a5 = 0.0
                    a6 = tmp1
                    a7 = 0.0
                    a8 = 0.0
                    rescounter = rescounter + 1
                    
                    ijunk = OpenMMGBSW_GBSWForce_addTitratingGroupParameters( gbswforce, &
                        residuePKA(I-1), residuePKA(I), Barr(I-1), Barr(I), &
                        a0, a1, a2, a3, a4, a5, a6, a7, a8 )
                    
                elseif ( SP_GRP(I) == 4 ) then
                    
                    ! 2-site residue where pKa1 .eq. pKa2 (i.e. GLU)
                    if ( ParMod(1,I) /= zero .and. ParMod(2,I) /= zero ) then
                        tmp1 = ParMod(1,I)
                        tmp2 = ParMod(2,I)
                        tmp3 = ParMod(3,I)
                        tmp4 = ParMod(4,I)
                        tmp5 = ParMod(5,I) - tmp1*tmp4*tmp4
                        tmp6 = -2.0 * ParMod(5,I) * ParMod(6,I) - tmp2*tmp4*tmp4
                    else
                        tmp  = 1.0 - 2.0 * SP_PAR(2,I)
                        tmp1 = (ParA(I-1) - ParA(I)) / tmp
                        tmp2 = 2.0 * (ParA(I)*ParB(I) - ParA(I-1)*ParB(I-1)) / tmp
                        tmp3 = SP_PAR(1,I)
                        tmp4 = SP_PAR(2,I)
                        tmp5 = ParA(I) - tmp1*tmp4*tmp4
                        tmp6 = -2.0 * ParA(I) * ParB(I) - tmp2*tmp4*tmp4
                    endif
                    
                    a0 = tmp1
                    a1 = -2.0 * tmp1 * tmp4
                    a2 = tmp2
                    a3 = -2.0 * tmp2 * tmp4
                    a4 = tmp1*tmp4*tmp4 + tmp5
                    a5 = tmp3
                    a6 = tmp2*tmp4*tmp4 + tmp6
                    a7 = -2.0 * tmp3 * tmp4
                    a8 = tmp3 * tmp4 * tmp4
                    
                    ijunk = OpenMMGBSW_GBSWForce_addTitratingGroupParameters( gbswforce, &
                        residuePKA(I-1), residuePKA(I), Barr(I-1), Barr(I), &
                        a0, a1, a2, a3, a4, a5, a6, a7, a8 )
                    
                endif
                
                I = I + 1
            enddo
            
            ! CPHMD in CHARMM uses the 1-4 nonbonded interactions for forces
            ! on the lambda/theta variables. here we fix the nonbonded exceptions
            ! list in OpenMM to include these interactions
            ! also, don't forget that in OpenMM atom indicies start at 0, not 1
            
            istart = 1
            do iatom = 1, natom
                
                ! locate iatom's nonbonded list start/end indicies
                if (iatom > 1) istart = max(BNBND%IBLO14(iatom-1)+1, 1)
                iend = BNBND%IBLO14(iatom)
                
                ! scan for 1-4 interactions
                do I = istart, iend
                if ( BNBND%INB14(I) < 0 ) then
                    
                    ! send exception fix to OpenMM
                    jatom = abs( BNBND%INB14(I) )
                    aa = iatom - 1
                    bb = jatom - 1
                    ijunk = OpenMMGBSW_GBSWForce_addNonbondedException( gbswforce, aa, bb )
                endif
                enddo
            enddo
            
            qphmd_initialized = .true.
            
        else
        ! if no CPHMD, just add the particles for classic GBSW
            
            do iatom = 1, natom
                if(.not.zero_gbsw) then
                    charge = fudge*cg(iatom)
                else
                    charge = ZERO
                endif
                radius = WMAIN(iatom) / OpenMM_AngstromsPerNm
                
                ijunk = OpenMMGBSW_GBSWForce_addParticle( gbswforce, charge, radius )
            enddo
        endif
        
    end subroutine gbsw_add_particles
    
#endif /* KEY_OPENMM && KEY_PHMD==1 */
 end module omm_gbsw
