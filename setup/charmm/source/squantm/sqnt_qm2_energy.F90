#if KEY_SQUANTM==1 /*mainsquatn*/
!************************************************************************
!                              AMBER                                   **
!                                                                      **
!                        Copyright (c) 2003                            **
!                Regents of the University of California               **
!                       All Rights Reserved.                           **
!                                                                      **
!  This software provided pursuant to a license agreement containing   **
!  restrictions on its disclosure, duplication, and use. This software **
!  contains confidential and proprietary information, and may not be   **
!  extracted or distributed, in whole or in part, for any purpose      **
!  whatsoever, without the express written permission of the authors.  **
!  This notice, and the associated author list, must be attached to    **
!  all copies, or extracts, of this software. Any additional           **
!  restrictions set forth in the license agreement also apply to this  **
!  software.                                                           **
!************************************************************************

      SUBROUTINE qm2_scf(fock_matrix,H,W,escf,P,POLD,ENUCLR,scf_mchg, &
                         indx_for_qm)
!
! MAIN SCF ROUTINE
!    This routine generates a self consistent field
!    and returns the energy in the variable ESCF.
!
! The main arrays used are:
!     fock_matrix - Starts off containing the one-electron matrix
!                   and is used to hold the Fock matrix.
!     H           - Only ever contains the one electron matrix.
!     W           - Only ever contains the two electron matrix.
!     P           - Only ever contains the total density matrix of 
!                   the current SCF step.
!     POLD        - Contains the density matrix from the previous 
!                   SCF step used by qm2_cnvg to damp oscillations
!                   in the fock matrix.
!
!     qm2_struct%matsize : Size of packed triangle 
!                          (NORBS*(NORBS+1)/2)
!     qm2_struct%n2el    : Number of 2 electron integrals
!                          50*nheavy(nheavy-1)+10*nheavy*nlight+ 
!                          (nlight*(nlight-1))/2
!
! References:
! PSEUDO DIAGONALISATION :"FAST SEMIEMPIRICAL CALCULATIONS",
!                         STEWART. J.J.P., CSASZAR, P., PULAY, P.,
!                         J. COMP. CHEM.,3, 227, (1982)
!
! Written by Ross Walker (TSRI, 2005)
!

  use qmmm_module, only : qmmm_nml,qmmm_struct, qm2_struct,   &
                          qm2_params, qm2_ghos, qmmm_ewald
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions

  use stream
#if KEY_PARALLEL==1
  use parallel 
#endif
!
      implicit none

!Passed in
      real(chm_real), intent(in)    :: ENUCLR
      real(chm_real), intent(inout) :: fock_matrix(qm2_struct%matsize)
      real(chm_real), intent(in)    :: H(qm2_struct%matsize)
      real(chm_real), intent(out)   :: W(qm2_struct%n2el)
      real(chm_real), intent(inout) :: P(qm2_struct%matsize),   &
                               POLD(qm2_struct%matsize)
      real(chm_real), intent(inout) :: escf
      real(chm_real), intent(inout) :: scf_mchg(qmmm_struct%nquant)
      integer,  intent(in)    :: indx_for_qm

!Local variables
      real(chm_real) :: density_conv              ! Density matrix convergence criteria.
      real(chm_real) :: enuclr_and_hof            ! Enuclr*EV_TO_KCAL + qm2_params%tot_heat_form 
      real(chm_real) :: eold                      ! SCF energy on previous step
      real(chm_real) :: energy_diff, density_diff ! Difference in energy and density from 
!                                                 ! previous step.a
      real(chm_real) :: small_local,smallsum      ! Precision limits of this machine.
      real(chm_real) :: smallest_energy_diff(2)   ! Smallest energy diff found so far (1) 
!                                             ! and density diff for this step(2)
      integer  :: smallest_energy_diff_step_number
      integer  :: scf_iteration             ! Number of scf iteration that have been done.
      logical  :: converged, first_iteration
      logical  :: doing_pseudo_diag         ! Initially false, set to true if we are doing 
!                                           ! a pseudo diagonalisation.
      logical  :: allow_pseudo_diag         ! Initially set to false. Set to true after at 
!                                           ! least 2 SCF iterations have been done and
!                                           ! allowed by the namelist option. Set back to 
!                                           ! full on last step to force a full diagonalization.
      logical  :: pseudo_converged          ! Flag used to indicate when doing pseudo 
!                                           ! diagonalisations has converged the SCF.
!                                           ! Once this is set to try the value of 
!                                           ! allow_pseudo_diag will not be set to true
!                                           ! again. This is used to force the last few SCF 
!                                           ! steps to be full diagonalisations.
      logical  :: first_call

! GHO specific variables. local and will be cleaned up soon
      integer  :: norbhb, naos
      logical  :: q_diis_conv

!qm2_Helect is a function
      real(chm_real)   :: qm2_HELECT
      integer    :: i

!QM/MM-Ewald extra energy correction term
      real(chm_real)   :: elec_eng_elec

! for printing control
      logical    :: qprint

      integer    :: mstart,mstop
      integer    :: ISTRT_CHECK                ! for external function
      integer    :: old_MatSize 
      integer    :: old_MatSize_keep(2)=0
      integer    :: mstart_keep(2)=0
      integer    :: mstop_keep(2)=0

! for timing
! old:::      real(chm_real)   :: ETIME,tarray(2)
! old:::      external ETIME

!Saves
      data first_call /.true./
      save first_call, smallsum
      save density_conv, old_MatSize_keep, mstart_keep,mstop_keep

      qprint=.false.
      If(Prnlev.ge.2) qprint=.true.

      if( old_MatSize_keep(indx_for_qm) .ne. qm2_struct%matsize ) then
         old_MatSize_keep(indx_for_qm) = qm2_struct%matsize
         mstart_keep(indx_for_qm)      = 1
         mstop_keep(indx_for_qm)       = qm2_struct%matsize
#if KEY_PARALLEL==1
         if(numnod.gt.1) mstart_keep(indx_for_qm)= &
                 ISTRT_CHECK(mstop_keep(indx_for_qm),qm2_struct%matsize)
      else 
         if(QMPI) then
            mstart_keep(indx_for_qm) = 1
            mstop_keep(indx_for_qm)  = qm2_struct%matsize
         end if
#endif 
      end if

      old_MatSize = old_MatSize_keep(indx_for_qm)
      mstart      = mstart_keep(indx_for_qm)
      mstop       = mstop_keep(indx_for_qm)

! Initialisation on first call
      If (first_call) then
! SCF convergence set
         if (qmmm_nml%tight_p_conv) then
             density_conv = qmmm_nml%scfconv
         else
             density_conv = half*sqrt(qmmm_nml%scfconv)
         end if

! Find precision limits of this machine
         call qm2_smallest_number(small_local,smallsum)

! smallsum should be the smallest number for 
! which 1.0D0 + smallsum /= 1.0D0.
! We will increase it in order to allow for 
! roundoff in our calculations.
         smallsum = ten * sqrt(smallsum)
         first_call=.false.
      End if

! Initialisation on every call
      converged                        =.false.
      first_iteration                  =.true.
      doing_pseudo_diag                =.false.
      allow_pseudo_diag                =.false.
      pseudo_converged                 =.false.

! little note: each node, now, has same number! (So, don't resum over node.)
      enuclr_and_hof                   = enuclr*EV_TO_KCAL +  &
                                         qm2_params%tot_heat_form

      eold                             = zero
      density_diff                     = zero
      energy_diff                      = zero
      smallest_energy_diff(1)          = 1.0D30
      smallest_energy_diff(2)          = 1.0D30
      smallest_energy_diff_step_number = 0

      q_diis_conv                      =.false.

! GHO part of work
      If( (qm2_ghos%QGHO) .and. (qm2_ghos%nqmlnk.gt.0) ) then
         norbhb = qm2_struct%norbs - 3*qm2_ghos%nqmlnk
         naos   = norbhb - qm2_ghos%nqmlnk
      End if

      if (qmmm_nml%verbosity .GT. 2 .and. qprint) then
         write(6,'("QMMM: ")')
         write(6,'("QMMM: SCF Convergence Information")')
         if(allow_pseudo_diag)  &
                  write(6,'("QMMM: (*) = Pseudo Diagonalisation")')
         write(6,'("QMMM: Cycle         Energy       ",$)')
         write(6,'("           dE                    dP")')
      end if

! MAIN SCF LOOP
 
      do_scf: do scf_iteration=1, qmmm_nml%itrmax
         if (.NOT. first_iteration) then

! Step 0 - check to use DIIS or other converger.
            if(qmmm_nml%Q_Diis .and. scf_iteration .gt. 100) then
               q_diis_conv =.true.
               allow_pseudo_diag = .false.  ! only do full diagonalization.
            else
               q_diis_conv =.false.
            end if

            if (q_diis_conv) then
                If( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
                   call qm2_diis(qm2_ghos%FAHB,qm2_ghos%PAOLD, &
                                 qm2_struct%diis_norbs, &
                                 qm2_struct%diis_linear,indx_for_qm, &
                                 qm2_struct%diis_fppf, &
                                 qm2_struct%diis_fock, &
                                 qm2_struct%diis_emat, &
                                 qm2_struct%diis_lfock, &
                                 qm2_struct%diis_nfock, &
                                 density_diff, &
                                 qm2_struct%diis_start)
                else
                   call qm2_diis(fock_matrix,P, &
                                 qm2_struct%diis_norbs, &
                                 qm2_struct%diis_linear,indx_for_qm, &
                                 qm2_struct%diis_fppf, &
                                 qm2_struct%diis_fock, &
                                 qm2_struct%diis_emat, &
                                 qm2_struct%diis_lfock, &
                                 qm2_struct%diis_nfock, &
                                 density_diff, &
                                 qm2_struct%diis_start)
                end if
!           write(6,*)' density_diff=',density_diff

            end if
! Step 1 - We haven't converged so we need to get a better
!         Fock matrix. 
!         Diagonalise the RHF secular determinant
! Diagonalise the RHF secular determinant
! We have two options here. If we are allowed, and the 
! density matrix fluctuations are small enough we should do
! a pseudo diagonalisation instead of a full one.

! timing check
!              call wrttim('start time of diagonalization')
!              write(6,*)'time=',etime(tarray)

            if ((allow_pseudo_diag) .AND.  &
                (density_diff .LE. qmmm_nml%pseudo_diag_criteria)) then

!We can do a pseudo diagonalisation.
! Flag used to control SCF routine to do a full 
! diagonalization before Return
               doing_pseudo_diag = .true. 

!Dimension 6 of mat_diag_workspace contains the eignvalues
!Dimension 5 is used as scratch space.

               If( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then 
                  call qm2_pseudo_diag(norbhb,norbhb,   &
                                  qm2_struct%norbs,qm2_struct%norbs,   &
                                  qm2_struct%nopenclosed,   &
                                  qm2_ghos%FAHB,   &
                                  qm2_ghos%CAHB,   &
                                  qm2_struct%mat_diag_workspace(1,6),   &
                                  qm2_struct%mat_diag_workspace(1,5),   &
                                  qm2_struct%pseudo_diag_matrix,   &
                                  smallsum)
               Else
                  call qm2_pseudo_diag(qm2_struct%norbs,   &
                                       qm2_struct%norbs,   &
                                  qm2_struct%norbs,qm2_struct%norbs,   &
                                  qm2_struct%nopenclosed,   &
                                  fock_matrix,   &
                                  qm2_struct%eigen_vectors,   &
                                  qm2_struct%mat_diag_workspace(1,6),   &
                                  qm2_struct%mat_diag_workspace(1,5),   &
                                  qm2_struct%pseudo_diag_matrix,   &
                                  smallsum)
               End if
            else
!Do a full diagonalisation
!!!            ! temporarily disable USE_LAPACK
!!!#ifdef USE_LAPACK
!!!            If((qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
!!!            Else
!!!               call dspev('V','U',qm2_struct%norbs,fock_matrix,  
!!!  *                     qm2_struct%mat_diag_workspace(1,6),    
!!!  *                     qm2_struct%eigen_vectors,qm2_struct%norbs,  
!!!  *                     qm2_struct%mat_diag_workspace,lapack_info)
!!!            End if
!!!#else

! Note: on our test machine - Intel P4 and Altix 
! this qm2_mat_diag routine is generally faster
! than the LAPACK diagonaliser.

! Dimension 6 of mat_diag_workspace contains 
! the eignvalues

               If( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
                  call qm2_mat_diag(norbhb, norbhb,   &
                               qm2_struct%norbs,qm2_struct%norbs,   &
                               qm2_ghos%FAHB, qm2_ghos%CAHB,   &
                               qm2_struct%mat_diag_workspace(1,1),   &
                               qm2_struct%mat_diag_workspace(1,2),   &
                               qm2_struct%mat_diag_workspace(1,3),   &
                               qm2_struct%mat_diag_workspace(1,4),   &
                               qm2_struct%mat_diag_workspace(1,5),   &
                               qm2_struct%mat_diag_workspace(1,6))
               Else
                  call qm2_mat_diag(qm2_struct%norbs,qm2_struct%norbs,   &
                               qm2_struct%norbs, qm2_struct%norbs,   &
                               fock_matrix, qm2_struct%eigen_vectors,   &
                               qm2_struct%mat_diag_workspace(1,1),   &
                               qm2_struct%mat_diag_workspace(1,2),   &
                               qm2_struct%mat_diag_workspace(1,3),   &
                               qm2_struct%mat_diag_workspace(1,4),   &
                               qm2_struct%mat_diag_workspace(1,5),   &
                               qm2_struct%mat_diag_workspace(1,6))
               End if
!!!#endif
            end if                                        ! Pseudo-diag.

! timing check
!              call wrttim('end time of diagonalization')
!              write(6,*)'time=',etime(tarray)

! Step 2 - Calculate the density matrix:
            If( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
               call qm2_densit(qm2_ghos%CAHB, norbhb,   &
                               qm2_struct%norbs, qm2_struct%nclosed,   &
                               qm2_struct%nopenclosed, indx_for_qm, &
                               qm2_ghos%PAHB)
            Else
               call qm2_densit(qm2_struct%eigen_vectors,   &
                               qm2_struct%norbs,   &
                               qm2_struct%norbs, qm2_struct%nclosed,   &
                               qm2_struct%nopenclosed, indx_for_qm, &
                               P)
            End if

! timing check
!              call wrttim('time of density matrix')
!              write(6,*)'time=',etime(tarray)

!***Note***
! In case of using GHO boundary atoms
!
! GHO expansions and Transform orbitals to AO basis, used 
! Mulliken analysis
            If( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
               Call gho_densit_expansion(qm2_ghos%nqmlnk,   &
                                         qm2_ghos%mqm16,   &
                                         qm2_struct%norbs,   &
                                         qm2_ghos%QMATMQ, qm2_ghos%Bt,   &
                                         qm2_ghos%Btm,   &
                                         qm2_ghos%CAHB,   &
                                         qm2_struct%eigen_vectors,   &
                                         qm2_ghos%PAHB, P, &
                                         qm2_ghos%PAOLD)
            End if
!

! timing check
!              call wrttim('time of gho expansion')
!              write(6,*)'time=',etime(tarray)

! Extrapolation/Damping
            If(.not. q_diis_conv) then
               call qm2_cnvg(P, POLD, qm2_struct%old2_den_matrix,   &
                             qm2_struct%norbs, scf_iteration-1,   &
                             density_diff)
            End if

! timing check
!              call wrttim('time of cnvg')
!              write(6,*)'time=',etime(tarray)

         end if ! if (.NOT. first_iteration)

! Step 3 - Build the fock matrix 
!        - Step 1 if this is our first SCF iteration.

! Construct FOCK matrix
!        - copy the one electron matrix into the FOCK matrix
!          and other things.

! timing check
!              call wrttim('start time of fock construction')
!              write(6,*)'time=',etime(tarray)

#if KEY_PARALLEL==1
           if(numnod.gt.1) then
              fock_matrix(1:mstart-1)                = zero
              fock_matrix(mstart:mstop)              = H(mstart:mstop)
              fock_matrix(mstop+1:qm2_struct%matsize)= zero
           else
              fock_matrix(mstart:mstop) = H(mstart:mstop)        ! (1:qm2_struct%matsize)
           end if
#else /**/
           fock_matrix(mstart:mstop) = H(mstart:mstop)        ! (1:qm2_struct%matsize)
#endif 

           call qm2_fock2(fock_matrix,P,W,qmmm_struct%nquant_nlink,   &
                          qm2_params%orb_loc,indx_for_qm)
 
           call qm2_fock1(fock_matrix,P,indx_for_qm)

#if KEY_PARALLEL==1
           if(numnod.gt.1) call GCOMB(fock_matrix,qm2_struct%matsize)
#endif 

! Modify Fock matrix in case of using QM/MM-Ewald sum methods.
           If (qmmm_ewald%QEwald) then
! a) calculate Mulliken charges for the current density 
!    matrix and save the results in the scf_mchg array.
!    currenlty, it is only implemented with Ewmode=1 
!    option
              if(qmmm_ewald%Ewmode .eq. 1)   &
                 call qm2_calc_mulliken(qmmm_struct%nquant,scf_mchg)

! Compute Ewald correction potential on QM atom site and 
! modify Fock matrix
! Since Eslf(nquant,nquant) matrix, it is only need to do 
! matrix multiplication to get the correction terms from
! QM images to be SCF iterated.
              Call qm_ewald_add_fock(qmmm_struct%nquant,fock_matrix,   &
                                     scf_mchg,   &
                                     qmmm_ewald%empot,qmmm_ewald%eslf)
           End if

! timing check
!              call wrttim('time of fock construction')
!              write(6,*)'time=',etime(tarray)

! GHO part
           If( (qm2_ghos%QGHO) .and. (qm2_ghos%nqmlnk.gt.0) ) then
!!***Note:
!! It is important to have unchanged Fock matrix in 
!! AO basis to be used in routine DQLINK to compute 
!! gradient correction for GHO atoms
!! Store Fock matrix in AO basis for derivative
!! However, since fock_matrix has been recontructed before
!! return to parent routine, we don't need FAOA copying 
!! and will use "qm2_struct%fock_matrix" in routine DQLINK
!!               qm2_ghos%FAOA(1:qm2_struct%matsize)= 
!!   *                             fock_matrix(1:qm2_struct%matsize)

! Transform F into HB for GHO atoms
              Call FTOFHB(fock_matrix,qm2_ghos%FAHB,qm2_ghos%Bt,   &
                          qm2_ghos%natqm,   &
                          qm2_ghos%nqmlnk,qm2_struct%norbs,   &
                          qm2_params%orb_loc)
           End if

! Step 4 - Calculate the energy in KCal/mol
           qmmm_struct%elec_eng = qm2_HELECT(qm2_struct%NORBS, &
                                             indx_for_qm, &
                                             P,H,fock_matrix)   &
                                  *two*EV_TO_KCAL

! In case of using QM/MM-Ewald, since MM atom should 
! contribute fully.
! For details, refer routine "qm_ewald_correct_ee"
           If (qmmm_ewald%QEwald) then
              call qm_ewald_correct_ee(qmmm_struct%nquant,indx_for_qm,   &
                                       elec_eng_elec,   &
                                       qmmm_ewald%empot,P)
              qmmm_struct%elec_eng = qmmm_struct%elec_eng +   &
                                     elec_eng_elec*EV_TO_KCAL
           End if

! add nuclear and heats of formation of atoms
#if KEY_PARALLEL==1
           if(numnod.gt.1) call GCOMB(qmmm_struct%elec_eng,1)
#endif 
           escf                 = qmmm_struct%elec_eng + enuclr_and_hof
           energy_diff          = escf - eold

! timing check
!              call wrttim('time of energy evaluation')
!              write(6,*)'time=',etime(tarray)

! debug
!          if(qprint.and.scf_iteration.lt.2) then
!          if(qprint.and.scf_iteration.gt.990) then
!             write(6,*)'Niter=',scf_iteration,' Escf=',escf
!          end if

! check scf convergence
           if (abs(energy_diff) .LT. abs(smallest_energy_diff(1))) then
! Keep track of the smallest difference we have found. 
! Useful for when we fail to converge.
!  - we can tell the user how close we every got.
              smallest_energy_diff(1)          = energy_diff
              smallest_energy_diff(2)          = density_diff
              smallest_energy_diff_step_number = scf_iteration
           end if

! If verbosity is >2 then print some info about this SCF step.
           if (qmmm_nml%verbosity .GT. 2 .and. qprint) then
              if (doing_pseudo_diag) then
                 write(6,'("QMMM: ",i5,"*",3G22.14)')   &
                                               scf_iteration, escf,   &
                                               energy_diff, density_diff
              else
                 write(6,'("QMMM: ",i5," ",3G22.14)')   &
                                               scf_iteration, escf,   &
                                               energy_diff, density_diff
              end if

              if (qmmm_nml%verbosity .GT. 4) then
! also print info in KJ/mol
                 write(6,'("QMMM: KJ/mol  ",3G22.14)') escf*4.184d0,   &
                           energy_diff*4.184d0, density_diff*4.184d0
              end if
           end if                              ! (qmmm_nml%verbosity > 2)

! Step 5 - Check if we have converged.
!        check energy difference is less than qmmm_nml%scfconv
!        check density difference is either zero or less than 
!        density_conv
!        Make sure we have done at least 2 SCF iterations to
!        avoid quiting due to some fluke.

           if (scf_iteration .GT. 2) then
! Since we have now done more than 2 iterations we can 
! allow the pseudo diagonalisation as long as it is 
! allowed from the namelist pseudo_diag option.
! Also, if doing pseudo diagonalisations has allowed us to
! converge and we are just doing a few more scf steps with
! full pseudo diagonalisation then make sure we don't turn
! pseudo diagonalisation back on by accident.

              if (qmmm_nml%allow_pseudo_diag .AND.   &
                                         (.not. pseudo_converged)) then
                  allow_pseudo_diag = .true.
              end if

              if ( ( abs(energy_diff) .LT. qmmm_nml%scfconv) .AND.   &
                   ( (density_diff .LT. density_conv) .OR.   &
                     (density_diff .EQ. zero) )    &
                 ) then

! We have converged. However, if the last SCF step was 
! a pseudo diagonalisation we need to do one more loop
! with full diagonalization. Otherwise, the forces will
! not be accurate enough.

                 if (doing_pseudo_diag) then
                    doing_pseudo_diag   = .false.
                    allow_pseudo_diag   = .false.
                    pseudo_converged    = .true. 
! Stops any more pseudo diags being done for the rest
! of the SCF loop.
                 else
                    converged           = .true.
                    exit do_scf  
! Break out of the SCF loop since we have converged.
                 end if
              end if
           end if                         ! scf_iteration>2

! for time-reversible Born-Oppenheimer MD simulations.
           if(qm2_struct%q_apply_tr_bomd .and. &
              qmmm_nml%num_scf_cycle.le.scf_iteration) then
              converged           = .true.
              exit do_scf
           end if

! Copy the energy that this step got us for use next time
         eold            = escf 
         first_iteration = .false.

      end do do_scf                  ! do scf_iteration, qmmm_nml%itrmax 

! END MAIN SCF LOOP


! We get to this point in the code for 2 reasons.
!  1) because we exceeded the maximum number of scf iterations.
!  2) because we converged and so broke out of the above loop.
!
!  Check which condition it is based on the value of the 
!  logical converged.
      if (.NOT. converged .and. qprint) then
! Convergence failed. Print a warning message and return with
! the current unconverged density. This will mean the forces
! are not accurate but in an MD simulation they may be good
! enough to allow the user to get out of a potentially bad 
! geometry.
         write(6,'(/,''QMMM: '',''WARNING!'')')
         write(6,'(''QMMM: '',   &
                   ''Unable to achieve self consistency '',   &
                   ''to the tolerances specified'')')
         write(6,'(''QMMM: '',   &
                   ''No convergence in SCF after '',i6,   &
                   '' steps.'')') qmmm_nml%itrmax
         write(6,'(''QMMM: '',   &
                   ''Job will continue with unconverged SCF.'',   &
                   '' Warning energies'')')
         write(6,'(''QMMM: '',   &
                   ''and forces for this step will not be accurate.'')')
         write(6,'(''QMMM: '',   &
                   ''E = '',E12.4,'' DeltaE = '',E12.4,   &
                   '' DeltaP = '',E12.4)')    &
                   escf, energy_diff, density_diff
         write(6,'(''QMMM: '',   &
                   ''Smallest DeltaE = '',E12.4,   &
                   '' DeltaP = '',E12.4,'' Step = '',i6,/)')   &
                   smallest_energy_diff(1), smallest_energy_diff(2),   &
                   smallest_energy_diff_step_number
      end if

! Make a copy of the density matrix for use next time the SCF 
! routine is called.
      pold(1:qm2_struct%matsize) = p(1:qm2_struct%matsize)

! Make a copy of several informations for GHO boundary atom method
      If ( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
! Transform orbitals to AO basis, used by Mulliken analysis
          Call CTRASF(qm2_struct%norbs,qm2_ghos%nqmlnk,qm2_ghos%mqm16,   &
                      qm2_ghos%Bt,qm2_ghos%CAHB,   &
                      qm2_struct%eigen_vectors) 

          qm2_ghos%PHO(1:qm2_struct%matsize) =   &
                                     qm2_ghos%PAHB(1:qm2_struct%matsize)

          if(qm2_ghos%UHFGHO) then
             ! Call CTRASF(...)
             qm2_ghos%PBHO(1:qm2_struct%matsize) =   &
                                     qm2_ghos%PBHB(1:qm2_struct%matsize)
          end if

      End if

      if (qmmm_nml%verbosity .GT. 0 .and. converged .and. qprint) then
         write(6,'(''QMMM: '',   &
                   ''SCF Converged to '',G10.4,'' in: '',i5,   &
                   '' Cycles '')')    &
                   qmmm_nml%scfconv,scf_iteration
      end if

      return
      END SUBROUTINE qm2_scf


      SUBROUTINE qm2_densit(eigen_vecs,NORBS,ndim1,NDUBL,NSINGL, &
                            indx_for_qm,P)            
!
! DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, 
! AND INFORMATION ABOUT THE M.O. OCCUPANCY.
!
! INPUT
!   eigen_vecs : SQUARE EIGENVECTOR MATRIX, 
!                (eigen_vecs IS OF SIZE norbs BY norbs)
!                AND THE EIGENVECTORS ARE STORED IN THE
!                TOP LEFT-HAND CORNER
!   NORBS      : NUMBER OF ORBITALS
!   NDUBL      : NUMBER OF DOUBLY-OCCUPIED M.O.S
!   NSINGL     : NUMBER OF DOUBLE+SINGLY OR OCCUPIED M.O.S
!
! ON EXIT
!   P          : DENSITY MATRIX
!
! SET UP LIMITS FOR SUMS
!  NL1         : BEGINING OF ONE ELECTRON SUM
!  NU1         : END OF SAME
!  NL2         : BEGINING OF TWO ELECTRON SUM
!  NU2         : END OF SAME
!

  use chm_kinds

  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

!Passed in
      integer, intent(in)    :: norbs, ndubl, ndim1, indx_for_qm
      integer, intent(inout) :: nsingl
      real(chm_real),  intent(in)  :: eigen_vecs(ndim1,ndim1)
      real(chm_real),  intent(out) :: P(*)

!Local variables
      integer :: unoc_start, sing_oc_start, i,j,k,l
      real(chm_real):: sum1,sum2
      integer :: mstart,mstop
      integer :: mstart_keep(2),mstop_keep(2)
      integer :: ISTRT_CHECK                   ! for external function
      integer :: old_Norbs(2) = 0
      save old_Norbs, mstart_keep,mstop_keep
!
      if(old_Norbs(indx_for_qm).ne.NORBS) then
         old_Norbs(indx_for_qm)  = NORBS
         mstart_keep(indx_for_qm)= 1
         mstop_keep(indx_for_qm) = NORBS
#if KEY_PARALLEL==1
! get mstart and mstop
         if(numnod.gt.1) mstart_keep(indx_for_qm) = &
                         ISTRT_CHECK(mstop_keep(indx_for_qm),NORBS)
      else
         if(QMPI) then
            mstart_keep(indx_for_qm)=1
            mstop_keep(indx_for_qm) =NORBS
         end if
#endif 
      end if

      mstart = mstart_keep(indx_for_qm)
      mstop  = mstop_keep(indx_for_qm)

#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         L = NORBS*(NORBS+1)/2
         P(1:L)     =zero
      end if
#endif 


! NSINGL=MAX(NDUBL,NSINGL)
      L          = 0
      unoc_start = NSINGL+1
      if (ndubl .EQ. nsingl) then
! All  M.O.s doubly occupied.
#if KEY_PARALLEL==1
         L = (mstart-1)*mstart / 2    ! same as "do i=1,mstart-1; L=L+i; end do"
#endif 
         do I=mstart,mstop            ! 1,NORBS
            do J=1,I
               L    = L+1
               SUM2 = zero
               do K=unoc_start,NORBS
! Loop over unoccupied MOs.
                  SUM2=SUM2+eigen_vecs(I,K)*eigen_vecs(J,K)
               end do
               P(L) =-two*SUM2
            end do
            P(L) = two+P(L)
         end do
      else
! We have some singularly occupied M.O.s - note we
! are restricted to ROHF here.
         sing_oc_start = NDUBL+1
#if KEY_PARALLEL==1
         L = (mstart-1)*mstart / 2    ! same as "do i=1,mstart-1; L=L+i; end do"
#endif 
         do I=mstart,mstop            ! 1,NORBS
            do J=1,I
               L    = L+1
               SUM2 = zero
               SUM1 = zero
               do K=unoc_start,NORBS
! Loop over unoccupied MOs.
                  SUM2=SUM2+eigen_vecs(I,K)*eigen_vecs(J,K)
               end do

               SUM2 = two*SUM2
               do K=sing_oc_start,NSINGL
! Loop over the singularly occupied orbitals
                  SUM1 = SUM1+eigen_vecs(I,K)*eigen_vecs(J,K)
               end do
               P(L) =-SUM2-SUM1
            end do
            P(L) = two+P(L)
         end do
      end if

#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         L=NORBS*(NORBS+1)/2
         call GCOMB(P,L)
      end if
#endif 

      return
      END SUBROUTINE qm2_densit


      SUBROUTINE qm2_cnvg(PNEW, P, P1,NORBS, NITER, PL)
!
! CNVG IS A TWO-POINT INTERPOLATION ROUTINE FOR SPEEDING
!      CONVERGENCE OF THE DENSITY MATRIX
!
! On output
!    PNEW  : New density matrix
!    P1    : Diagonal of old density matrix
!    P2    : Largest difference between old and new density
!            matrix diagonal elements
!

  use qmmm_module, only : qm2_params

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

!Passed in
      real(chm_real),  intent(inout) :: P1(*), P(*), PNEW(*)
      real(chm_real),  intent(out)   :: pl
      integer, intent(in)    :: norbs, niter

!Local variables
      LOGICAL :: EXTRAP
      real(chm_real)  :: faca, damp, facb, fac, sum, sum0, sum1, sum2
      real(chm_real)  :: sa, a
      integer :: ii,k,i,j,ie

      PL     = zero
      FACA   = zero
      DAMP   = TENTOPLUS10                       ! 1.D10
      IF(NITER.GT.3) DAMP=0.05D0
      FACB   = zero
      FAC    = zero
      II     = MOD(NITER,3)
      EXTRAP = II.NE.0
      SUM1   = zero
      K      = 0

      do I=1,NORBS
         K    = K+I
         A    = PNEW(K)
         SUM1 = SUM1+A
         SA   = ABS(A-P(K))

         IF (SA.GT.PL) PL = SA
         if (.NOT.EXTRAP) then
            FACA = FACA+SA**2
            FACB = FACB+(A-two*P(K)+P1(I))**2
         end if
         P1(I)= P(K)
         P(K) = A
      end do

      if (FACB .GT. zero) then
         IF (FACA.LT.(100.D00*FACB)) FAC = SQRT(FACA/FACB)
      end if

      IE   = 0
      SUM2 = zero
      do I=1,NORBS
         II = I-1
         do J=1,II
            IE       = IE+1
            A        = PNEW(IE)
            P(IE)    = A+FAC*(A-P(IE))
            PNEW(IE) = P(IE)
         end do

         IE = IE+1
         IF (ABS(P(IE)-P1(I)) .GT. DAMP) THEN
            P(IE)=P1(I)+SIGN(DAMP,P(IE)-P1(I))
         ELSE
            P(IE)=P(IE)+FAC*(P(IE)-P1(I))
         ENDIF

         P(IE)    = MIN(two,MAX(P(IE),zero))
         SUM2     = SUM2+P(IE)
         PNEW(IE) = P(IE)
      end do

! Re-narmalize if any density matrix elements have been truncated
      SUM0 = SUM1
      do
         IF (SUM2.GT.TEN_TO_MINUS3) THEN                ! 1.D-3
            SUM = SUM1/SUM2
         ELSE
            SUM = zero
         ENDIF
         SUM1 = SUM0
         IF (SUM2 .LE. TEN_TO_MINUS3 .OR.  &
             ABS(SUM-one) .LE. TEN_TO_MINUS5) exit      ! 1.D-3 , 1.D-5

         SUM2 = zero
         do I=1,NORBS
            J = qm2_params%pascal_tri2(i)

! Add on a small number in case an occupancy is exactly zero
            P(J) = P(J)*SUM+1.D-20
            P(J) = MAX(P(J),zero)

! Setup renormalization over partly occupied M.O.'s only.
! Full M.O.'s can not be filled any more
            IF (P(J).GT.two) THEN
               P(J) = two
               SUM1 = SUM1-two
            ELSE
               SUM2 = SUM2+P(J)
            ENDIF
            PNEW(J) = P(J)
         end do
      end do

      return
      END SUBROUTINE qm2_cnvg


      SUBROUTINE qm2_diis(F,P,n,linear,indx_for_qm, &
                          fppf,fock,emat,lfock,nfock,PL,qstart)
!
  use chm_kinds
  use qm2_double
  use qm2_constants

      implicit none
!
! Passed in
      integer   :: n,linear,lfock,nfock,indx_for_qm
      real(chm_real)  :: F(*),P(*)   ! F(Linear),P(Linear)
      real(chm_real)  :: fppf(Linear,6),fock(Linear,6)
      real(chm_real)  :: emat(20,20)
      real(chm_real)  :: PL
      logical   :: qstart

! note:
! fppf : pold  =pold1(1:lpulay)  => pold1(linear,6)
! fock : pold2(1:lpulay)         => pold2(linear,6)
! emat : pold3(1:norbs) (or pold3(1:36)) ? below..it is pold3(20,20)
! lfock: jalp = 0 at every energy call.
! nfock: ialp = 1 to be 1 since (start.eq..true.)
! msize: lpulay=6*linear
! start: =.true. at every energy call.

!***********************************************************************
!
!   PULAY USES DR. PETER PULAY'S METHOD FOR CONVERGENCE.
!         A MATHEMATICAL DESCRIPTION CAN BE FOUND IN
!         "P. PULAY, J. COMP. CHEM. 3, 556 (1982).
!
! ARGUMENTS:-
!         ON INPUT F      = FOCK MATRIX, PACKED, LOWER HALF TRIANGLE.
!                  P      = DENSITY MATRIX, PACKED, LOWER HALF TRIANGLE.
!                  N      = NUMBER OF ORBITALS.
!                  FPPF   = WORKSTORE OF SIZE MSIZE, CONTENTS WILL BE
!                           OVERWRITTEN.
!                  FOCK   =      "       "              "         "
!                  EMAT   = WORKSTORE OF AT LEAST 15**2 ELEMENTS.
!                  START  = LOGICAL, = TRUE TO START PULAY.
!                  PL     = UNDEFINED ELEMENT.
!      ON OUTPUT   F      = "BEST" FOCK MATRIX, = LINEAR COMBINATION
!                           OF KNOWN FOCK MATRICES.
!
!                  PL     = MEASURE OF NON-SELF-CONSISTENCY
!                         = [F*P] = F*P - P*F.
!
!***********************************************************************

! Local variables
      integer   :: ii,il,nfock1,i,j,l,k,jj,kk
      integer, parameter :: maxlim=6, mfock=6
      integer            :: ncnt
      real(chm_real)           :: evec(1000),coeffs(20)
      real(chm_real)           :: sum,const,D
      real(chm_real)           :: r_linear, r_linear_keep(2), sum_1, sum_2
      real(chm_real)           :: emat_2(20,20)

      save r_linear_keep
!
      if(qstart) then
         nfock  = 1
         lfock  = 1
         qstart =.false.

         r_linear_keep(indx_for_qm)=one/real(linear)
      else
         if(nfock.lt.mfock) nfock=nfock+1  ! nfock=1,6 during iterations.
         if(lfock.ne.mfock) then
            lfock=lfock+1
         else
            lfock=1                        ! reset lfock=1, if lfock=6
         end if
      end if

      r_linear = r_linear_keep(indx_for_qm)

!
! store Fock matrix for future referene.
      fock(1:linear,lfock) = f(1:linear)     ! here, I change a little bit of matrix
                                             ! shape.

!
! form /fock*density - density*fock/, and store this in FPPF.
      ncnt = 0
      do i=1,n
         ii =((i-1)*i) / 2
         do j=1,i
            jj    =((j-1)*j) / 2
            sum_1 = zero
            sum_2 = zero
            do k=1,j
               sum_1 = sum_1 + P(ii+k)*F(jj+k)
            end do
            do k=j+1,i
               sum_1 = sum_1 + P(ii+k)*F(((k-1)*k) / 2 + j)
            end do
            do k=i+1,n
               kk= (k*(k-1)) / 2
               sum_1 = sum_1 + P(kk+i)*F(kk+j)
            end do

            do k=1,j
               sum_2 = sum_2 + F(ii+k)*P(jj+k)
            end do
            do k=j+1,i
               sum_2 = sum_2 + F(ii+k)*P(((k-1)*k)/2 +j)
            end do
            do k=i+1,n
               kk= (k*(k-1)) / 2
               sum_2 = sum_2 + F(kk+i)*P(kk+j)
            end do

            ncnt = ncnt + 1
            fppf(ncnt,lfock) = (sum_2 - sum_1)
         end do
      end do

!
! fppf contains the result of F*P-P*F.
      nfock1=nfock+1
      do i=1,nfock
         emat(nfock1,i)=-one
         emat(i,nfock1)=-one
         sum_1 = zero
         do j=1,linear
            sum_1 = sum_1 + fppf(j,i)*fppf(j,lfock)
         end do
         emat(lfock,i) = sum_1
         emat(i,lfock) = sum_1
      end do
      pl =  emat(lfock,lfock) * r_linear  ! PL=EMAT(LFOCK,LFOCK)/LINEAR
      emat(nfock1,nfock1)=zero

      const=one/emat(lfock,lfock)

      emat_2(1:nfock1,1:nfock1)=emat(1:nfock1,1:nfock1)
      emat_2(1:nfock,1:nfock)  =emat_2(1:nfock,1:nfock)*const

      ncnt=0
      do i=1,nfock1
         do j=1,nfock1
            ncnt=ncnt+1
            evec(ncnt) = emat_2(i,j)
         end do
      end do

!********************************************************************
!   THE MATRIX EMAT SHOULD HAVE FORM
!
!      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
!      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
!      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
!      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
!      |     .            .      ...     . |
!      |   -1.0         -1.0     ...    0. |
!
!   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
!   TIMES [F*P] FOR ITERATION J.
!
!********************************************************************
      CALL OSINV(EVEC,NFOCK1,D)      ! invert square matrix.

      if(abs(D).lt. 1.0d-6) then
         qstart=.true.
         return
      end if

      if(nfock.lt.2) return

      il=nfock*nfock1
      coeffs(1:nfock)=-evec(1+il:nfock+il)

      do i=1,linear
         sum=zero
         do j=1,nfock
            sum=sum+coeffs(j)*fock(i,j)
         end do
         sum_1 = f(i)
         f(i)=sum
      end do

      RETURN
      END


      SUBROUTINE OSINV (A,N,D)
!
  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer  :: N
      real(chm_real) :: A(1000),D
!***********************************************************************
!
!    OSINV INVERTS A GENERAL SQUARE MATRIX OF ORDER UP TO MAXORB. SEE
!          DIMENSION STATEMENTS BELOW.
!
!   ON INPUT       A = GENERAL SQUARE MATRIX STORED LINEARLY.
!                  N = DIMENSION OF MATRIX A.
!                  D = VARIABLE, NOT DEFINED ON INPUT.
!
!   ON OUTPUT      A = INVERSE OF ORIGINAL A.
!                  D = DETERMINANT OF ORIGINAL A, UNLESS A WAS SINGULAR,
!                      IN WHICH CASE D = 0.0
!
!***********************************************************************
!
! Local variable
      integer  :: ij,ji,ik,ki,jk,kj,kk,nk,jp,jq,jr,iz,i,j,k
      real(chm_real) :: biga,holo

      integer, dimension(21) :: L,M              ! L(MAXORB), M(MAXORB)
      real(chm_real), parameter    :: TOL=1.0d-8       ! this can be tightened.
      real(chm_real), parameter    :: one=1.0d0,zero=0.0d0

!
!***********************************************************************
      D  = one
      NK =-N

      DO K=1,N
         NK  =NK+N
         L(K)=K
         M(K)=K
         KK  =NK+K
         BIGA=A(KK)

         do J=K,N
            IZ=N*(J-1)
            do I=K,N
               IJ=IZ+I
               if( ABS(BIGA) .lt. ABS(A(IJ)) ) then
                  BIGA=A(IJ)
                  L(K)=I
                  M(K)=J
               end if
            end do
         end do

         J=L(K)
         if(J .gt. K) then
            KI = K-N
            do I=1,N
               KI   = KI+N
               HOLO =-A(KI)
               JI   = KI-K+J
               A(KI)= A(JI)
               A(JI)= HOLO
            end do
         end if

         I=M(K)
         if(i.gt.k) then
            JP = N*(I-1)
            do J=1,N
               JK   = NK+J
               JI   = JP+J
               HOLO =-A(JK)
               A(JK)= A(JI)
               A(JI)= HOLO
            end do
         end if

         if(ABS(BIGA) .lt. TOL) then
            D=zero
            return
         end if

         do I=1,N
            if(i.ne.k) then
               IK   = NK+I
               A(IK)= A(IK)/(-BIGA)
            end if
         end do

         do I=1,N
            IK = NK+I
            IJ = I-N
            do J=1,N
               IJ = IJ+N
               if(i.ne.k) then
                  if(j.ne.k) then
                     KJ   = IJ-I+K
                     A(IJ)= A(IK)*A(KJ)+A(IJ)
                  end if
               end if
            end do
         end do

         KJ = K-N
         do J=1,N
            KJ = KJ+N
            if(j.ne.k) A(KJ)=A(KJ)/BIGA
         end do

         D    =D*BIGA
         A(KK)=one/BIGA
      End do                     ! K=1,N

      K=N
  190 K=K-1
      if (k .le. 0) goto 260

      I = L(K)
      if(i.gt.k) then
         JQ=N*(K-1)
         JR=N*(I-1)
         do J=1,N
            JK   = JQ+J
            HOLO = A(JK)
            JI   = JR+J
            A(JK)=-A(JI)
            A(JI)= HOLO
         end do
      end if

      J=M(K)
      if (j.gt.k) then
         KI=K-N
         do I=1,N
            KI   = KI+N
            HOLO = A(KI)
            JI   = KI+J-K
            A(KI)=-A(JI)
            A(JI)= HOLO
         end do
      end if
      goto 190

  260 return
      END


      SUBROUTINE qm2_mat_diag(n,m,ndim1,ndim2,a,v,w1,w2,w3,w4,w5,e)
!
! qm2_mat_diag is a diagonalisation routine
!    Reference: Yoshitaka Beppu of Nagoya University, Japan.
!               'Computers & Chemistry' vol.6 1982. page 000
!
! On input
!    a       : matrix to be diagonalised (packed canonical)
!    n       : size of matrix to be diagonalised
!    m       : number of eigenvectors needed
!    e       : array of size at least n
!    v       : array of size at least nmax*m
!    ndim1   : size of matrix 1, same as n, except GHO case
!    ndim2   : size of matrix 2, same as m, except GHO case
!
! On output
!    e       : eigenvalues
!    v       : eigenvectors in array of size nmax*m
!
! This version has been modernised and optimised by
! Ross Walker and David Case (TSRI 2005)
!

  use qmmm_module, only : qm2_params

  use chm_kinds

  use qm2_double
  use qm2_constants

      implicit none

!Passed in
      integer, intent(in)    :: m,n,ndim1,ndim2
      real(chm_real),  intent(inout) :: a(*)             ! Note original matrix will be 
!                                                  ! corrupted by this routine.
      real(chm_real),  intent(out)   :: e(ndim1), v(ndim1,ndim2)
      real(chm_real),  intent(out)   :: w1(ndim1), w2(ndim1), w3(ndim1)
      real(chm_real),  intent(out)   :: w4(ndim1), w5(ndim1)

!Local variables
      real(chm_real)   :: eps, eps3, ssum, r, s, h, summ, ff, ww, hinv, rinv
      real(chm_real)   :: c, gersch, z, sinv, ee, ra, t, rn, vn, del, u
      integer  :: nm1, nm2, irank, jrank, krank, i, j, k, l, kp1, ip1
      integer  :: im1, ii, kk, kpiv, ig, itere, ll

! eps and eps3 are machine-precision dependent
      eps  = 1.0d-8
      eps3 = 1.0d-30
      nm1  = n-1
      if (n .GT. 2) then
         nm2  = n-2
         krank= 0

! Householder transformation
         do k=1,nm2
            kp1   = k+1
            krank = krank+k
            w2(k) = a(krank)
            ssum  = zero
            jrank = krank

            do j=kp1,n
               w2(j) = a(jrank+k)
               jrank = jrank+j
               ssum  = w2(j)*w2(j)+ssum
            end do

            s         = sign(sqrt(ssum),w2(kp1))
            w1(k)     =-s
            w2(kp1)   = w2(kp1)+s
            a(k+krank)= w2(kp1)
            h         = w2(kp1)*s
            if (abs(h) .LT. eps3) then
               a(krank) = h
               cycle
            end if

            hinv  = one/h
            summ  = zero
            irank = krank

            do i=kp1,n
               ssum = zero
               do j=kp1,i
                  ssum = ssum+a(j+irank)*w2(j)
               end do
               if (i .LT. n) then
                  ip1 = i+1
!                                ! jrank=ishft(i*(i+3),-1)
!                                ! jrank=i*(i+3)/2
                  jrank    = qm2_params%pascal_tri2(i)+i
                  do j=ip1,n
                     ssum  = ssum+a(jrank)*w2(j)
                     jrank = jrank+j
                  end do
               end if
               w1(i) = ssum*hinv
               irank = irank+i
               summ  = w1(i)*w2(i)+summ
            end do
            u     = summ*half*hinv
            jrank = krank

            do j=kp1,n
               w1(j) = w2(j)*u-w1(j)
               do i=kp1,j
                  a(i+jrank)=w1(i)*w2(j)+w1(j)*w2(i)+a(i+jrank)
               end do
               jrank = jrank+j
            end do
            a(krank) = h
         end do
      end if

      w2(nm1)= a( qm2_params%pascal_tri2(nm1) )   ! =a( ishft((nm1*(nm1+1)),-1) )
      w2(n)  = a( qm2_params%pascal_tri2(n))      ! =a( ishft((n*(n+1)),-1) )
      w1(nm1)= a( nm1+qm2_params%pascal_tri1(n) ) ! =a( nm1+ishft((n*(n-1)),-1) )
      w1(n)  = zero
      gersch = abs(w2(1))+abs(w1(1))
      do i=1,nm1
         gersch=max(abs(w2(i+1))+abs(w1(i))+abs(w1(i+1)),gersch)
      end do

      del = eps*gersch
      do i=1,n
         w3(i) = w1(i)
         e(i)  = w2(i)
         v(i,m)= e(i)
      end do

      if (abs(del) .GE. eps3) then

! QR-method with origin shift
         k = n
         do 
            l=k
            do
               if (abs(w3(l-1)) .LT. del) exit
               l = l-1
               if (l .EQ. 1)  exit
            end do

            if (l .NE. k) then
               ww     = (e(k-1)+e(k))*half
               r      = e(k)-ww
               z      = sign(sqrt(w3(k-1)**2+r*r),r)+ww
               ee     = e(l)-z
               e(l)   = ee
               ff     = w3(l)
               rinv   = one/sqrt(ee*ee+ff*ff)
               c      = e(l)*rinv
               s      = w3(l)*rinv
               ww     = e(l+1)-z
               e(l)   = (ff*c+ww*s)*s+ee+z
               e(l+1) = c*ww-s*ff

               do j=l+1,k-1
                  rinv   = one/sqrt(e(j)*e(j)+w3(j)*w3(j))
                  r      = one/rinv
                  w3(j-1)= s*r
                  ee     = e(j)*c
                  ff     = w3(j)*c
                  c      = e(j)*rinv
                  s      = w3(j)*rinv
                  ww     = e(j+1)-z
                  e(j)   = (ff*c+ww*s)*s+ee+z
                  e(j+1) = c*ww-s*ff
               end do
               w3(k-1)   = e(k)*s
               e(k)      = e(k)*c+z
               cycle
            end if

            k = k-1
            if(k .EQ. 1) exit
         end do

!At this point the array 'e' contains the un-ordered eigenvalues
!Straight selection sort of eigenvalues:
         j = n
         do
            l = 1
            ii= 1
            ll= 1
            do i=2,j
               if ((e(i)-e(l)) .GE. zero ) then
                  l = i
                  cycle
               end if

               ii = i
               ll = l
            end do

            if(ii .NE. ll) then
               ww    = e(ll)
               e(ll) = e(ii)
               e(ii) = ww
            end if

            j        = ii-1
            if(j .LT. 2) exit
         end do
      end if

      if(m == 0) return
! Ordering of eigenvalues is complete.

! Inverse-iteration for eigenvectors:
      rn = zero
      ra = eps*0.6180339887485d0
!              ! [0.618... is the fibonacci number (-1+sqrt(5))/2.]
      ig = 1
      do i=1,m
         im1 = i-1

         do j=1,n
            w3(j)  = zero
            w4(j)  = w1(j)
            w5(j)  = v(j,m)-e(i)
            rn     = rn+ra
            if(rn.ge.eps) rn = rn-eps
            v(j,i) = rn
         end do

         do j=1,nm1
            if(abs(w5(j)) .LT. abs(w1(j))) then
               w2(j)  =-w5(j)/w1(j)
               w5(j)  = w1(j)
               t      = w5(j+1)
               w5(j+1)= w4(j)
               w4(j)  = t
               w3(j)  = w4(j+1)
               if(abs(w3(j)).lt.eps3) w3(j) = del
               w4(j+1)= zero
            else
               if(abs(w5(j)).lt.eps3) w5(j) = del
               w2(j)  =-w1(j)/w5(j)
            end if

            w4(j+1)   = w3(j)*w2(j)+w4(j+1)
            w5(j+1)   = w4(j)*w2(j)+w5(j+1)
         end do

         if(abs(w5(n)) .LT. eps3) w5(n) = del

         do itere=1,5
            if(itere .NE. 1) then
               do j=1,nm1
                  if(abs(w3(j)).lt.eps3) cycle
                  t       = v(j,i)
                  v(j,i)  = v(j+1,i)
                  v(j+1,i)= t
                  v(j+1,i)= v(j,i)*w2(j)+v(j+1,i)
               end do
            end if

            v(n,i)   = v(n,i)/w5(n)
            v(nm1,i) = (v(nm1,i)-v(n,i)*w4(nm1))/w5(nm1)
            vn       = max(abs(v(n,i)),abs(v(nm1,i)),1.0d-20)

            if(n .NE. 2) then
               k     = nm2
               do
                  v(k,i) = (v(k,i)-v(k+1,i)*w4(k)-v(k+2,i)*w3(k))/w5(k)
                  vn     = max(abs(v(k,i)),vn,1.d-20)
                  k      = k-1
                  if(k .LT. 1) exit
               end do
            end if

            s = TEN_TO_MINUS5/vn                 ! 1.0d-5/vn
            do j=1,n
               v(j,i) = v(j,i)*s
            end do
            if(itere .GT. 1 .and. vn .GT. 1) exit
         end do

! Transformation of eigenvectors
         if (n .NE. 2) then
            krank  = ishft(nm2*(n+1),-1)
            kpiv   = ishft(nm2*nm1,-1)

            do k=nm2,1,-1
               kp1 = k+1

               if(abs(a(kpiv)) .GT. eps3) then
                  ssum = zero
                  do kk=kp1,n
                     ssum = ssum+a(krank)*v(kk,i)
                     krank= krank+kk
                  end do

                  s    =-ssum/a(kpiv)
                  do kk=n,kp1,-1
                     krank   = krank-kk
                     v(kk,i) = a(krank)*s+v(kk,i)
                  end do
               end if

               kpiv  = kpiv-k
               krank = krank-kp1
            end do
         end if

         do j=ig,i-1
            if( abs(e(j)-e(i)) .LT. 0.05d0 ) exit
         end do

         ig=j
         if (ig .NE. i) then

! re-orthogonalisation:
            do k=ig,im1
               ssum = zero
               do j=1,n
                  ssum   = v(j,k)*v(j,i)+ssum
               end do

               s    =-ssum
               do j=1,n
                  v(j,i) = v(j,k)*s+v(j,i)
               end do
            end do
         end if

! Normalisation:
         ssum = 1.0d-24
         do j=1,n
            ssum   = ssum+v(j,i)**2
         end do
         sinv = one/sqrt(ssum)
         do j=1,n
            v(j,i) = v(j,i)*sinv
         end do
      end do

      return
      END SUBROUTINE qm2_mat_diag


      SUBROUTINE qm2_pseudo_diag(ndim,norbs,ndim1,ndim2,noccupied,   &
                              matrix,vectors,eigen,workspace,   &
                              scratch_matrix,   &
                              smallsum)
!
! "FAST" but "APPROXIMATE" matrix diagonalisation.
! See: Stewart, J.J.P., Csaszar, P., Pulay, P., J. Comp. Chem. 
!      3:227,1982
!
! Written by Ross Walker (TSRI, 2005)
!
! qm2_pseudo_diag is a "FAST" but "approximate" matrix 
! diagonalisation routine. The vectors generated by this 
! routine are more likely to be able to block-diagonalise
! the Fock matrix over the molecular orbitals than the
! starting vectors. It is based on the work by Stewart et al.
! and must be considered to be a pseudo diagonaliser because:
!
!  1) Eigenvectors are not generated. Only the occupied-virtual 
!     intersection is diagonalised, not the secular determinant.
!
!  2) When rotation is used to eliminate elements of the secular 
!     determinant the remainder is assumed to remain unchanged.
!     Any elements that are created are thus ignored.
!
!  3) The rotation required to eliminate those elements considered 
!     significant is approximated to using the eigenvalues of the
!     exact diagonalisation throughout the rest of the iterative
!     procedure. In other words the errors on each pseudo-step are
!     propogated to the next SCF step. The errors here are assumed
!     to be smaller than the actual error in the SCF at this point
!     in the SCF.
!
! Variables:
!  MATRIX : On input should contain the matrix to be diagonalised 
!           in the form of a packed lower half triangle. 
!

  use qmmm_module, only : qm2_struct
  use chm_kinds
  use qm2_double
  use qm2_constants

      implicit none

!Passed in
      integer, intent(in)   :: ndim, norbs, ndim1, ndim2, noccupied
      real(chm_real),  intent(in)   :: matrix(qm2_struct%matsize)
      real(chm_real),  intent(inout):: vectors(ndim1,ndim2)
      real(chm_real),  intent(in)   :: eigen(ndim2)
      real(chm_real),  intent(out)  :: workspace(ndim2)
      real(chm_real),intent(out):: scratch_matrix(noccupied*(ndim2-noccupied))
      real(chm_real),  intent(in)   :: smallsum

!Local variables
      real(chm_real)  :: sum1,sum2,a,b,c,d,e,alpha,beta
      real(chm_real)  :: eigeni
      integer :: i,j,k,j1,k2,ij,kk,m,lumo

      lumo = noccupied+1
      ij   = 0
      do i = lumo, norbs
         kk = 0

         do j=1,norbs
            sum1 = zero
            do k=1,j
               kk  = kk+1
               sum1= sum1+matrix(kk)*vectors(k,i)
            end do

            j1 = j+1
            k2 = kk
            do k=j1,norbs
               k2  = k2+k-1
               sum1= sum1+matrix(k2)*vectors(k,i)
            end do

            workspace(j) = sum1
         end do                               ! j=1,norbs

         do j=1,noccupied
            ij   = ij+1
            sum2 = zero
            do k=1,norbs
               sum2=sum2+workspace(k)*vectors(k,j)
            end do
            scratch_matrix(ij) = sum2
         end do                               ! j=1,noccupied
      end do                                  ! i=lumo,norbs

!
! Now we have done the squaring we can do a crude 2 by 2 rotation,
! which we will assume eliminates the significant elements.
!

      ij = 0
      do i=lumo,norbs
         eigeni = eigen(i)
         do j=1,noccupied
            ij  = ij+1
            C   = scratch_matrix(ij)
            D   = eigen(j)-eigeni

!Check the machine precision for whether to do a 2x2 rotation
            if (abs(C) .GE. (smallsum*abs(D))) then
! E=sign(1.0D0/sqrt(4.0D0*c*c+d*d),d)
               E     = one/sqrt(four*c*c+d*d)
               E     = half+abs(half*D*E)
               alpha = sqrt(E)
               beta  = -sign(sqrt(one-E),c)

! Rotate the pseudo eigenvectors
               do m=1,norbs
                  a           = vectors(m,j)
                  b           = vectors(m,i)
                  vectors(m,j)= alpha*a+beta*b
                  vectors(m,i)= alpha*b-beta*a
               end do
            end if                            ! abs(C/D) >= smallsum
         end do                               ! j=1,noccupied
      end do                                  ! i=lumo,norbs

      return
      END SUBROUTINE qm2_pseudo_diag
#else /*   (mainsquatn)*/
      SUBROUTINE qm2_scf_BLANK
!
! dummy routine for compilation
!
      RETURN
      END
#endif /* (mainsquatn)*/

