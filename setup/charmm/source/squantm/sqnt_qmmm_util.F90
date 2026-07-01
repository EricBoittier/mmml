#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1 /*mainsquatn*/
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

      SUBROUTINE validate_qm_atoms(iqmatoms, nquant, natom, Qcheck)
!
! This routine will check the list of atoms numbers stored in 
! iqmatoms and check the following:
!
! 1) All are >= 1 .and. <= natoms
! 2) All are unique integer numbers
!
! Written by Ross Walker, TSRI, 2004
!
 
  use chm_kinds
  use qm2_double
  use stream

      implicit none

!Passed in
      integer, intent(in)    :: nquant, natom
      integer, intent(in)    :: iqmatoms(*)
      logical, intent(inout) :: Qcheck

!Local
      integer icount1, icount2, iatom
 
! Sanity check 1, ensure nquant isn't bigger than natom 
! (it can't be)
      if ((nquant .LT. 1) .OR. (nquant .GT. natom)) then
         if(Prnlev.ge.2) then
           write(6,'(" QM ATOM VALIDATION: nquant has a value of ",i8)')   &
                nquant
           write(6,'(" which is bigger than natom of ",i8,   &
                   ". Need 0 < nquant <= natom.")') natom
           write (6,'(A)')   &
         'validate_qm_atoms: QM atoms overflow, should less NATOM.'
         end if
 
         Qcheck =.FALSE.
         return
      end if
 
! Check 2 - loop over nquant atoms and check it is > 1 
!           and <= natom and it is unique

      do icount1=1,nquant
         iatom = iqmatoms(icount1)
         if ( (iatom .GT. 0) .AND. (iatom .LE. natom)  ) then
!check atom number is legal
!QM atom ID is valid - check it is unique
            do icount2=(icount1+1),nquant
               if ( iatom .EQ. iqmatoms(icount2) ) then
                  if(Prnlev.ge.2) then
                     write (6,'(" QM ATOM VALIDATION: qm atom ",i8,   &
                                " is not unique.")') iatom
                     write (6,'(A)')   &
         'QM atoms specified with iqmatoms do not form a unique set.'
                  end if

                  Qcheck =.FALSE.
                  return
               end if
            end do
         else
! it is not legal
            if(Prnlev.ge.2) then
               write (6,'(" QM ATOM VALIDATION: iqmatom ID number of ", &
                          i8," is not valid.")') iatom
               write (6,'(A)')'Invalid QM atom ID.'
            end if

            Qcheck =.FALSE.
            return
         end if
      end do

      return
      END SUBROUTINE validate_qm_atoms


      SUBROUTINE qm_assign_atom_types
!
! This routine will go through the atomic numbers for each of the 
! QM atoms and work out how many different elements there are.
! It will then assign an atom type to each QM atom. This atom type
! is essentially a rebasing of the atomic number in order to save memory.
!
! Written by Ross Walker (TSRI,2005)
!

  use qmmm_module, only : qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_parameters, only : nelements

      implicit none

!Passed in

!Local
      integer :: ier=0
      integer :: i,j, natqmi
      logical :: assigned

      allocate(qmmm_struct%qm_atom_type(qmmm_struct%nquant_nlink),  &
               stat=ier )
      if(ier.ne.0) call Aass(1,'qm_assign_atom_types','qm_atom_type')
!Deallocated in deallocate_qmmm

      qmmm_struct%qm_atom_type(1:qmmm_struct%nquant_nlink) = 0
      qmmm_struct%qm_type_id(1:nelements) = 0
      qmmm_struct%qm_ntypes = 0

      do i=1,qmmm_struct%nquant_nlink
         natqmi = qmmm_struct%iqm_atomic_numbers(i)
         assigned = .false.

! Loop over the number of types found so far and see if this atomic number
! has already been assigned to a type.
         do j=1,qmmm_struct%qm_ntypes
            if (natqmi .EQ. qmmm_struct%qm_type_id(j)) then
               assigned = .true.
               qmmm_struct%qm_atom_type(i) = j
            end if
         end do

! if assigned is true here then we have already assigned this type, we just
! simply move to the next atom.
         if ( .NOT. assigned) then
            qmmm_struct%qm_ntypes = qmmm_struct%qm_ntypes + 1
            qmmm_struct%qm_type_id(qmmm_struct%qm_ntypes) = natqmi 
            qmmm_struct%qm_atom_type(i) = qmmm_struct%qm_ntypes
         end if
      end do

      return
      END SUBROUTINE qm_assign_atom_types


      SUBROUTINE qm2_guess_density

  use qmmm_module, only : qmmm_nml,qmmm_struct, qm2_struct,  &
                              qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants

      implicit none

! Passed in

! Local variables
      integer   :: i, j
      integer   :: first_orb, last_orb
      real(chm_real)  :: pdiag_guess1, pdiag_guess2


!set up array of lower half triangle indicies (Pascal's triangle)
! lower triangle index: start from pascal_tri1(i)+1
!                                  ~ pascal_tri2(I)
      do I=1,qm2_struct%norbs
         qm2_params%pascal_tri1(I)=ishft((I*(I-1)),-1)
         qm2_params%pascal_tri2(I)=qm2_params%pascal_tri1(I)+I
      end do

!Guess Density Matrix

!Initialize the density matrix on setup
      qm2_struct%den_matrix = zero
      qm2_struct%old_den_matrix = zero

!Fill the diagonal of the density matrix with the first guess:
      pdiag_guess1=dble(qmmm_nml%qmcharge)/(qm2_struct%norbs +   &
                                            TEN_TO_MINUS10)
      do i=1,qmmm_struct%nquant_nlink
         first_orb = qm2_params%orb_loc(1,i)
         last_orb  = qm2_params%orb_loc(2,i)
         pdiag_guess2=(qm2_params%core_chg(i)/   &
                      (qm2_params%natomic_orbs(i)+TEN_TO_MINUS10))   &
                     - pdiag_guess1
         do j=first_orb,last_orb
           qm2_struct%den_matrix(qm2_params%pascal_tri2(j))    =   &
                                                     pdiag_guess2
           qm2_struct%old_den_matrix(qm2_params%pascal_tri2(j))=   &
                                                     pdiag_guess2
         end do
      end do
!END of Density Matrix

      return
      END SUBROUTINE qm2_guess_density

      SUBROUTINE qm2_load_params(qprint,qcheck)
!!
!! Written by: Ross Walker (TSRI, 2005)
!!
!! This routine should be called before running any qm2 routine 
!! calculations.
!! It is responsible for filling the parameter arrays with the 
!! designated parameters for the method chosen. 
!!
!! All parameters are loaded into qm2_params structure.
!!
!! Note, allocation is done in this routine for all pointers in the 
!! qm2_params structure.
!! Deallocation is done by deallocate qmmm routine.
!!
!! Parameter definitions:
!!     Common params for all methods:
!!     core(nquant_nlink)         - The core charge on each atom
!!     natomic_orbs(nquant_nlink) - Number of atomic orbitals on each atom.
!!     orb_loc(2,nquant_nlink)    - (1,x) = position of first orbital on 
!!                                          atom x. 2,x = last orbital on x.
!!     heat_of_form(nquant_nlink) - Gas Phase heat of formation for atom.

  use qmmm_module, only : qmmm_nml,qmmm_struct, qm2_struct,   &
                              qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
! put here.
  use qm2_parameters   ! include the file containing all of the parameter constants.

      implicit none

!Passed in
      logical, intent(in)    :: qprint
      logical, intent(inout) :: qcheck

!Locals
      real(chm_real)  :: pddg_zaf, pddg_zbf
      integer :: i, j, iqm_atomic, n_atomic_orb, first_orb, last_orb
      integer :: nheavy_atoms, nlight_atoms, nelectrons, nopen
      integer :: ier=0
      real(chm_real)  :: onesq, rhalf

! Some values replacing explicit number
      onesq = 1.25D0
      rhalf = 0.50D0

! Now we fill up the data depending on the method we are using
!
!--------------------------------------
!           MNDO PARAMS               *
!--------------------------------------
      if (qmmm_nml%qmtheory .EQ. MNDO) then
        do i = 1,qmmm_struct%nquant_nlink
          iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

! Check that parameters exist for this element in MNDO
          if (.NOT. element_supported_mndo(iqm_atomic)) then
             if(qprint) then
             write(6,'("QMMM: Atom number: ",i6," has atomic number ",   &
                       i4,".")') i, iqm_atomic
             write(6,*)'QMMM: ',   &
                       'There are no MNDO parameters for this element.'
             end if

             Qcheck = .FALSE.
             return
          end if

! Next do the electronic energy - add it to the total heat 
!                                 of formation energy.
          qm2_params%tot_heat_form         = qm2_params%tot_heat_form -  &
                                            (elec_eng_mndo(iqm_atomic)*  &
                                            EV_TO_KCAL)
          qm2_params%onec2elec_params(1,i) = rhalf*GSS_mndo(iqm_atomic)
          qm2_params%onec2elec_params(2,i) = GSP_mndo(iqm_atomic)
          qm2_params%onec2elec_params(3,i) = rhalf*GPP_mndo(iqm_atomic)
          qm2_params%onec2elec_params(4,i) = onesq*GP2_mndo(iqm_atomic)
          qm2_params%onec2elec_params(5,i) = rhalf*HSP_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(1,i) = DD_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(2,i) = QQ_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(3,i) =   &
                                          rhalf/AM_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(4,i) =   &
                                          rhalf/AD_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(5,i) =   &
                                          rhalf/AQ_mndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(6,i) =   &
                               qm2_params%multip_2c_elec_params(3,i)*   &
                               qm2_params%multip_2c_elec_params(3,i)
          qm2_params%multip_2c_elec_params(7,i) =   &
                               qm2_params%multip_2c_elec_params(4,i)*   &
                               qm2_params%multip_2c_elec_params(4,i)
          qm2_params%multip_2c_elec_params(8,i) =   &
                               qm2_params%multip_2c_elec_params(5,i)*   &
                               qm2_params%multip_2c_elec_params(5,i)
          qm2_params%cc_exp_params(i) = alp_mndo(iqm_atomic)
          qm2_params%orb_elec_ke(1,i) = uss_mndo(iqm_atomic)
          qm2_params%orb_elec_ke(2,i) = upp_mndo(iqm_atomic)
        end do

!Loop over elements: Slater orbital expansion coefficient
        do i = 1,nelements
           qm2_params%s_orb_exp_by_type(i) = s_orb_exp_mndo(i)
           qm2_params%p_orb_exp_by_type(i) = p_orb_exp_mndo(i)
        end do

! Precompute some parameters to save time later
! RCW: By rebasing the atomic numbers of each atoms as types we 
! reduce the overall size of the arrays.
! While this does not save us much memory it greatly increases
! the chance of cache hits.
        do i=1,qmmm_struct%qm_ntypes
          do j = 1,qmmm_struct%qm_ntypes
            qm2_params%betasas(i,j) =   &
                              betas_mndo(qmmm_struct%qm_type_id(i))+   &
                              betas_mndo(qmmm_struct%qm_type_id(j))
            qm2_params%betasap(i,j) =   &
                              betas_mndo(qmmm_struct%qm_type_id(i))+   &
                              betap_mndo(qmmm_struct%qm_type_id(j))
            qm2_params%betapap(i,j) =   &
                              betap_mndo(qmmm_struct%qm_type_id(i))+   &
                              betap_mndo(qmmm_struct%qm_type_id(j))
          end do
        end do
!--------------------------------------
!       END  MNDO PARAMS              *
!--------------------------------------

!--------------------------------------
!           AM1 PARAMS                *
!--------------------------------------
      else if (qmmm_nml%qmtheory .EQ. AM1) then
        do i = 1,qmmm_struct%nquant_nlink
          iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

! Check that parameters exist for this element in AM1
          if (.NOT. element_supported_am1(iqm_atomic)) then
             if(qprint) then
             write(6,'("QMMM: Atom number: ",i6," has atomic number ",  &
                       i4,".")') i, iqm_atomic
             write(6,*)'QMMM: ',   &
                       'There are no AM1 parameters for this element.'
             end if

             Qcheck = .FALSE.
             return
          end if

! Next do the electronic energy - add it to the total heat of formation energy.
          qm2_params%tot_heat_form         = qm2_params%tot_heat_form-   &
                                            (elec_eng_am1(iqm_atomic)*   &
                                             EV_TO_KCAL)
          qm2_params%onec2elec_params(1,i) = rhalf*GSS_am1(iqm_atomic)
          qm2_params%onec2elec_params(2,i) = GSP_am1(iqm_atomic)
          qm2_params%onec2elec_params(3,i) = rhalf*GPP_am1(iqm_atomic)
          qm2_params%onec2elec_params(4,i) = onesq*GP2_am1(iqm_atomic)
          qm2_params%onec2elec_params(5,i) = rhalf*HSP_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(1,i) = DD_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(2,i) = QQ_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(3,i) =   &
                                            rhalf/AM_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(4,i) =   &
                                            rhalf/AD_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(5,i) =   &
                                            rhalf/AQ_am1(iqm_atomic)
          qm2_params%multip_2c_elec_params(6,i) =   &
                               qm2_params%multip_2c_elec_params(3,i)*   &
                               qm2_params%multip_2c_elec_params(3,i)
          qm2_params%multip_2c_elec_params(7,i) =   &
                               qm2_params%multip_2c_elec_params(4,i)*   &
                               qm2_params%multip_2c_elec_params(4,i)
          qm2_params%multip_2c_elec_params(8,i) =   &
                               qm2_params%multip_2c_elec_params(5,i)*   &
                               qm2_params%multip_2c_elec_params(5,i)
          qm2_params%cc_exp_params(i) = alp_am1(iqm_atomic)
          qm2_params%orb_elec_ke(1,i) = uss_am1(iqm_atomic)
          qm2_params%orb_elec_ke(2,i) = upp_am1(iqm_atomic)
        end do

!Loop over elements: Slater orbital expansion coefficient
        do i = 1,nelements
           qm2_params%s_orb_exp_by_type(i) = s_orb_exp_am1(i)
           qm2_params%p_orb_exp_by_type(i) = p_orb_exp_am1(i)
        end do

! Precompute some parameters to save time later
! RCW: By rebasing the atomic numbers of each atoms as types 
!      we reduce the overall size of the arrays.
! While this does not save us much memory it greatly increases
! the chance of cache hits.
        do i=1,qmmm_struct%qm_ntypes
          qm2_params%NUM_FN(i)   = NUM_FN_am1(qmmm_struct%qm_type_id(i))
          do j=1,4
             qm2_params%FN1(j,i) = FN1_am1(j,qmmm_struct%qm_type_id(i))
             qm2_params%FN2(j,i) = FN2_am1(j,qmmm_struct%qm_type_id(i))
             qm2_params%FN3(j,i) = FN3_am1(j,qmmm_struct%qm_type_id(i))
          end do
        end do

        do i=1,qmmm_struct%qm_ntypes
         do j = 1,qmmm_struct%qm_ntypes
          qm2_params%betasas(i,j)=betas_am1(qmmm_struct%qm_type_id(i))+  &
                                  betas_am1(qmmm_struct%qm_type_id(j))
          qm2_params%betasap(i,j)=betas_am1(qmmm_struct%qm_type_id(i))+  &
                                  betap_am1(qmmm_struct%qm_type_id(j))
          qm2_params%betapap(i,j)=betap_am1(qmmm_struct%qm_type_id(i))+   &
                                  betap_am1(qmmm_struct%qm_type_id(j))
         end do
        end do
!--------------------------------------
!        END  AM1 PARAMS              *
!--------------------------------------

!--------------------------------------
!      PM3 AND PM3CARB1 PARAMS        *
!--------------------------------------
      else if ( (qmmm_nml%qmtheory .EQ. PM3) .OR.   &
                (qmmm_nml%qmtheory .EQ. PM3CARB1) ) then
        do i = 1,qmmm_struct%nquant_nlink
          iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

! Check that parameters exist for this element in PM3
          if (.NOT. element_supported_pm3(iqm_atomic)) then
             if(qprint) then
             write(6,'("QMMM: Atom number: ",i6," has atomic number ",  &
                       i4,".")') i, iqm_atomic
             write(6,*) 'QMMM: ',   &
                     'There are no PM3 parameters for this element.'
             end if

             Qcheck = .FALSE.
             return
          end if

! Next do the electronic energy - add it to the total heat of 
! formation energy.
          qm2_params%onec2elec_params(1,i) = rhalf*GSS_pm3(iqm_atomic)
          qm2_params%onec2elec_params(2,i) = GSP_pm3(iqm_atomic)
          qm2_params%onec2elec_params(3,i) = rhalf*GPP_pm3(iqm_atomic)
          qm2_params%onec2elec_params(4,i) = onesq*GP2_pm3(iqm_atomic)
          qm2_params%onec2elec_params(5,i) = rhalf*HSP_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(1,i) = DD_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(2,i) = QQ_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(3,i) =   &
                                            rhalf/AM_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(4,i) =   &
                                            rhalf/AD_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(5,i) =   &
                                            rhalf/AQ_pm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(6,i) =   &
                              qm2_params%multip_2c_elec_params(3,i)*   &
                              qm2_params%multip_2c_elec_params(3,i)
          qm2_params%multip_2c_elec_params(7,i) =   &
                              qm2_params%multip_2c_elec_params(4,i)*   &
                              qm2_params%multip_2c_elec_params(4,i)
          qm2_params%multip_2c_elec_params(8,i) =   &
                              qm2_params%multip_2c_elec_params(5,i)*   &
                              qm2_params%multip_2c_elec_params(5,i)

          if ( (qmmm_nml%qmtheory .EQ. PM3CARB1) .AND.   &
               (iqm_atomic.EQ.1 .OR. iqm_atomic.EQ.8) ) then
!Load the PM3CARB1 versions of O and H params in place of 
!the default PM3 params
            qm2_params%tot_heat_form    = qm2_params%tot_heat_form-   &
                                      (elec_eng_pm3carb1(iqm_atomic)*   &
                                          EV_TO_KCAL)
            qm2_params%cc_exp_params(i) = alp_pm3carb1(iqm_atomic)
            qm2_params%orb_elec_ke(1,i) = uss_pm3carb1(iqm_atomic)
            qm2_params%orb_elec_ke(2,i) = upp_pm3carb1(iqm_atomic)
          else
            qm2_params%tot_heat_form    = qm2_params%tot_heat_form-   &
                                         (elec_eng_pm3(iqm_atomic)*   &
                                          EV_TO_KCAL)
            qm2_params%cc_exp_params(i) = alp_pm3(iqm_atomic)
            qm2_params%orb_elec_ke(1,i) = uss_pm3(iqm_atomic)
            qm2_params%orb_elec_ke(2,i) = upp_pm3(iqm_atomic)
          end if
        end do

!Loop over elements: Slater orbital expansion coefficient
        do i = 1,nelements
           qm2_params%s_orb_exp_by_type(i) = s_orb_exp_pm3(i)
           qm2_params%p_orb_exp_by_type(i) = p_orb_exp_pm3(i)
        end do

! Precompute some parameters to save time later
! RCW: By rebasing the atomic numbers of each atoms as types 
! we reduce the overall size of the arrays.
! While this does not save us much memory it greatly increases
! the chance of cache hits.
        do i=1,qmmm_struct%qm_ntypes
          qm2_params%NUM_FN(i)   = NUM_FN_pm3(qmmm_struct%qm_type_id(i))
          do j=1,4
             qm2_params%FN1(j,i) = FN1_pm3(j,qmmm_struct%qm_type_id(i))  
             qm2_params%FN2(j,i) = FN2_pm3(j,qmmm_struct%qm_type_id(i))  
             qm2_params%FN3(j,i) = FN3_pm3(j,qmmm_struct%qm_type_id(i))   
          end do
        end do

!RCW: PM3CARB1 update - we need to make sure we use the correct 
!                       parameters here for hydrogen and oxygen 
!                       atoms. Simple solution is to replace the
!                       data in the betas_pm3 and betap_pm3
!                       arrays with the PM3CARB1 values.
!This avoids having to use complex if statements in the loop 
!below.
        if (qmmm_nml%qmtheory .EQ. PM3CARB1) then
!Replace PM3 betas and betap params for O and H 
!with PM3CARB1 values BEFORE they are copied into
!the working array.
           betas_pm3(1) = betas_pm3carb1(1)
           betas_pm3(8) = betas_pm3carb1(8)
           betap_pm3(1) = betap_pm3carb1(1)
           betap_pm3(8) = betap_pm3carb1(8)
        end if

        do i=1,qmmm_struct%qm_ntypes
          do j = 1,qmmm_struct%qm_ntypes
            qm2_params%betasas(i,j) =   &
                       betas_pm3(qmmm_struct%qm_type_id(i))+   &
                       betas_pm3(qmmm_struct%qm_type_id(j))
            qm2_params%betasap(i,j) =   &
                       betas_pm3(qmmm_struct%qm_type_id(i))+   &
                       betap_pm3(qmmm_struct%qm_type_id(j))
            qm2_params%betapap(i,j) =   &
                       betap_pm3(qmmm_struct%qm_type_id(i))+   &
                       betap_pm3(qmmm_struct%qm_type_id(j))
          end do
        end do
!--------------------------------------
!    END PM3 AND PM3 CARB1 PARAMS     *
!--------------------------------------

!--------------------------------------
!         PDDG/PM3 PARAMS             *
!--------------------------------------
      else if (qmmm_nml%qmtheory .EQ. PDDGPM3) then
        do i = 1,qmmm_struct%nquant_nlink
          iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

! Check that parameters exist for this element in PM3
          if (.NOT. element_supported_pddgpm3(iqm_atomic)) then
             if(qprint) then
             write(6,'("QMMM: Atom number: ",i6," has atomic number ",   &
                       i4,".")') i, iqm_atomic
             write(6,*) 'QMMM: ',   &
                   'There are no PDDG-PM3 parameters for this element.'
             end if

             Qcheck = .FALSE.
             return
          end if

! Next do the electronic energy - add it to the total heat of 
! formation energy.
          qm2_params%tot_heat_form         = qm2_params%tot_heat_form -  &
                                         (elec_eng_pddgpm3(iqm_atomic)*  &
                                             EV_TO_KCAL)
          qm2_params%onec2elec_params(1,i) =   &
                                         rhalf*GSS_pddgpm3(iqm_atomic)
          qm2_params%onec2elec_params(2,i) = GSP_pddgpm3(iqm_atomic)
          qm2_params%onec2elec_params(3,i) =   &
                                         rhalf*GPP_pddgpm3(iqm_atomic)
          qm2_params%onec2elec_params(4,i) =   &
                                         onesq*GP2_pddgpm3(iqm_atomic)
          qm2_params%onec2elec_params(5,i) =   &
                                         rhalf*HSP_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(1,i) = DD_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(2,i) = QQ_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(3,i) =   &
                                         rhalf/AM_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(4,i) =   &
                                         rhalf/AD_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(5,i) =   &
                                         rhalf/AQ_pddgpm3(iqm_atomic)
          qm2_params%multip_2c_elec_params(6,i) =   &
                              qm2_params%multip_2c_elec_params(3,i)*   &
                              qm2_params%multip_2c_elec_params(3,i)
          qm2_params%multip_2c_elec_params(7,i) =   &
                              qm2_params%multip_2c_elec_params(4,i)*   &
                              qm2_params%multip_2c_elec_params(4,i)
          qm2_params%multip_2c_elec_params(8,i) =   &
                              qm2_params%multip_2c_elec_params(5,i)*   &
                              qm2_params%multip_2c_elec_params(5,i)
          qm2_params%cc_exp_params(i) = alp_pddgpm3(iqm_atomic)
          qm2_params%orb_elec_ke(1,i) = uss_pddgpm3(iqm_atomic)
          qm2_params%orb_elec_ke(2,i) = upp_pddgpm3(iqm_atomic)
          qm2_params%pddge1(i)        = pddge1_pm3(iqm_atomic)
          qm2_params%pddge2(i)        = pddge2_pm3(iqm_atomic)
        end do

!Loop over elements: Slater orbital expansion coefficient
        do i = 1,nelements
           qm2_params%s_orb_exp_by_type(i) = s_orb_exp_pddgpm3(i)
           qm2_params%p_orb_exp_by_type(i) = p_orb_exp_pddgpm3(i)
        end do

!Precompute some parameters to save time later
!RCW: By rebasing the atomic numbers of each atoms as types 
!we reduce the overall size of the arrays. While this does
!not save us much memory it greatly increases the chance of 
!cache hits.
        do i=1,qmmm_struct%qm_ntypes
          qm2_params%NUM_FN(i)   =   &
                              NUM_FN_pddgpm3(qmmm_struct%qm_type_id(i))
          do j=1,4
             qm2_params%FN1(j,i) =   &
                              FN1_pddgpm3(j,qmmm_struct%qm_type_id(i))
             qm2_params%FN2(j,i) =   &
                              FN2_pddgpm3(j,qmmm_struct%qm_type_id(i))
             qm2_params%FN3(j,i) =   &
                              FN3_pddgpm3(j,qmmm_struct%qm_type_id(i))
          end do
        end do

        do i=1,qmmm_struct%qm_ntypes
          do j = 1,qmmm_struct%qm_ntypes
            qm2_params%betasas(i,j) =   &
                             betas_pddgpm3(qmmm_struct%qm_type_id(i))   &
                            +betas_pddgpm3(qmmm_struct%qm_type_id(j))
            qm2_params%betasap(i,j) =   &
                             betas_pddgpm3(qmmm_struct%qm_type_id(i))   &
                            +betap_pddgpm3(qmmm_struct%qm_type_id(j))
            qm2_params%betapap(i,j) =   &
                             betap_pddgpm3(qmmm_struct%qm_type_id(i))   &
                            +betap_pddgpm3(qmmm_struct%qm_type_id(j))
            pddg_zaf = dble(core_chg(qmmm_struct%qm_type_id(i)))   &
                      /(dble(core_chg(qmmm_struct%qm_type_id(i)))+   &
                        dble(core_chg(qmmm_struct%qm_type_id(j))))
            pddg_zbf = dble(core_chg(qmmm_struct%qm_type_id(j)))   &
                      /(dble(core_chg(qmmm_struct%qm_type_id(i)))+   &
                        dble(core_chg(qmmm_struct%qm_type_id(j))))
            qm2_params%pddg_term1(i,j) =   &
                       pddg_zaf*pddgc1_pm3(qmmm_struct%qm_type_id(i))   &
                      +pddg_zbf*pddgc1_pm3(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term2(i,j) =   &
                       pddg_zaf*pddgc1_pm3(qmmm_struct%qm_type_id(i))   &
                      +pddg_zbf*pddgc2_pm3(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term3(i,j) =   &
                       pddg_zaf*pddgc2_pm3(qmmm_struct%qm_type_id(i))   &
                      +pddg_zbf*pddgc1_pm3(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term4(i,j) =   &
                       pddg_zaf*pddgc2_pm3(qmmm_struct%qm_type_id(i))   &
                      +pddg_zbf*pddgc2_pm3(qmmm_struct%qm_type_id(j))
          end do
        end do
!--------------------------------------
!          END PDDG/PM3 PARAMS        *
!--------------------------------------
!--------------------------------------
!           PDDG/MNDO PARAMS          *
!--------------------------------------
      elseif (qmmm_nml%qmtheory .EQ. PDDGMNDO) then
        do i = 1,qmmm_struct%nquant_nlink
          iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

! Check that parameters exist for this element in MNDO
          if (.NOT. element_supported_pddgmndo(iqm_atomic)) then
             if(qprint) then
             write(6,'("QMMM: Atom number: ",i6," has atomic number ",   &
                       i4,".")') i, iqm_atomic
             write(6,*) 'QMMM: ',   &
                  'There are no PDDG-MNDO parameters for this element.'
             end if

             Qcheck = .FALSE.
             return
          end if

! Next do the electronic energy - add it to the total heat of 
! formation energy.
          qm2_params%tot_heat_form         = qm2_params%tot_heat_form -  &
                                        (elec_eng_pddgmndo(iqm_atomic)*  &
                                             EV_TO_KCAL)
          qm2_params%onec2elec_params(1,i) =   &
                                        rhalf*GSS_pddgmndo(iqm_atomic)
          qm2_params%onec2elec_params(2,i) = GSP_pddgmndo(iqm_atomic)
          qm2_params%onec2elec_params(3,i) =   &
                                        rhalf*GPP_pddgmndo(iqm_atomic)
          qm2_params%onec2elec_params(4,i) =   &
                                        onesq*GP2_pddgmndo(iqm_atomic)
          qm2_params%onec2elec_params(5,i) =   &
                                        rhalf*HSP_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(1,i) =   &
                                        DD_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(2,i) =   &
                                        QQ_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(3,i) =   &
                                        rhalf/AM_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(4,i) =   &
                                        rhalf/AD_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(5,i) =   &
                                        rhalf/AQ_pddgmndo(iqm_atomic)
          qm2_params%multip_2c_elec_params(6,i) =   &
                               qm2_params%multip_2c_elec_params(3,i)*   &
                               qm2_params%multip_2c_elec_params(3,i)
          qm2_params%multip_2c_elec_params(7,i) =   &
                               qm2_params%multip_2c_elec_params(4,i)*   &
                               qm2_params%multip_2c_elec_params(4,i)
          qm2_params%multip_2c_elec_params(8,i) =   &
                               qm2_params%multip_2c_elec_params(5,i)*   &
                               qm2_params%multip_2c_elec_params(5,i)
          qm2_params%cc_exp_params(i) = alp_pddgmndo(iqm_atomic)
          qm2_params%orb_elec_ke(1,i) = uss_pddgmndo(iqm_atomic)
          qm2_params%orb_elec_ke(2,i) = upp_pddgmndo(iqm_atomic)
          qm2_params%pddge1(i)        = pddge1_mndo(iqm_atomic)
          qm2_params%pddge2(i)        = pddge2_mndo(iqm_atomic)
        end do

!Loop over elements: Slater orbital expansion coefficient
        do i = 1,nelements
           qm2_params%s_orb_exp_by_type(i) = s_orb_exp_pddgmndo(i)
           qm2_params%p_orb_exp_by_type(i) = p_orb_exp_pddgmndo(i)
        end do

!Precompute some parameters to save time later
!RCW: By rebasing the atomic numbers of each atoms as types 
!we reduce the overall size of the arrays. While this does not 
!save us much memory it greatly increases the chance of 
!cache hits.
        do i=1,qmmm_struct%qm_ntypes
          do j = 1,qmmm_struct%qm_ntypes
            qm2_params%betasas(i,j) =   &
                            betas_pddgmndo(qmmm_struct%qm_type_id(i))   &
                           +betas_pddgmndo(qmmm_struct%qm_type_id(j))
            qm2_params%betasap(i,j) =   &
                            betas_pddgmndo(qmmm_struct%qm_type_id(i))   &
                           +betap_pddgmndo(qmmm_struct%qm_type_id(j))
            qm2_params%betapap(i,j) =   &
                            betap_pddgmndo(qmmm_struct%qm_type_id(i))   &
                           +betap_pddgmndo(qmmm_struct%qm_type_id(j))
            pddg_zaf = dble(core_chg(qmmm_struct%qm_type_id(i)))   &
                      /(dble(core_chg(qmmm_struct%qm_type_id(i)))+   &
                        dble(core_chg(qmmm_struct%qm_type_id(j))))
            pddg_zbf = dble(core_chg(qmmm_struct%qm_type_id(j)))   &
                      /(dble(core_chg(qmmm_struct%qm_type_id(i)))+   &
                        dble(core_chg(qmmm_struct%qm_type_id(j))))
            qm2_params%pddg_term1(i,j) =   &
                      pddg_zaf*pddgc1_mndo(qmmm_struct%qm_type_id(i))   &
                     +pddg_zbf*pddgc1_mndo(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term2(i,j) =   &
                      pddg_zaf*pddgc1_mndo(qmmm_struct%qm_type_id(i))   &
                     +pddg_zbf*pddgc2_mndo(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term3(i,j) =   &
                      pddg_zaf*pddgc2_mndo(qmmm_struct%qm_type_id(i))   &
                     +pddg_zbf*pddgc1_mndo(qmmm_struct%qm_type_id(j))
            qm2_params%pddg_term4(i,j) =   &
                      pddg_zaf*pddgc2_mndo(qmmm_struct%qm_type_id(i))   &
                     +pddg_zbf*pddgc2_mndo(qmmm_struct%qm_type_id(j))
          end do
        end do
!--------------------------------------
!     END PDDG/MNDO PARAMS            *
!--------------------------------------
      else

!UNKNOWN method - should never actually get this far.
        if(qprint) then
        write (6,'("QMMM ERROR: Method ID: ",i5,   &
                   " is not supported.")') qmmm_nml%qmtheory
        write (6,'("QMMM ERROR:UNSUPPORTED METHOD")')
        end if

        Qcheck = .FALSE.
        return
      end if

!Now see if user wants an MM peptide torsion correction
      qm2_struct%n_peptide_links = 0
      if(qmmm_nml%peptide_corr) THEN
         if(qprint) then
            write (6,*) 'QMMM: ',   &
         'MOLECULAR MECHANICS CORRECTION APPLIED TO PEPTIDE LINKAGES'
         end if

#if KEY_QTURBO==0 && KEY_G09==0
         call qm2_identify_peptide_links(qm2_struct%n_peptide_links,   &
                                         qmmm_struct%qm_coords)
#endif 
         if (qprint) then
            write (6,'(''QMMM: '',i5,   &
                     '' PEPTIDE LINKAGES HAVE BEEN FOUND:'')')   &
                       qm2_struct%n_peptide_links
            do i=1,qm2_struct%n_peptide_links
              write(6,'(''QMMM:    '',i4,'' - '',i4,'' - '',i4,   &
                        '' - '',i4)')   &
                        qm2_struct%peptide_links(1,i),   &
                        qm2_struct%peptide_links(2,i),   &
                        qm2_struct%peptide_links(3,i),   &
                        qm2_struct%peptide_links(4,i)
            end do
         end if
      end if

      return
      END SUBROUTINE qm2_load_params

      SUBROUTINE CheckSpinState(Nspin,Nelec,Nopen,Nclose,   &
                             Norbs,qprint,QGHO,Qcheck)
!
! Check the spin state and print informations.
!
  use chm_kinds
      implicit none

!Passed in
      integer, intent(in)     :: Nspin, Nelec, Norbs
      integer, intent(inout)  :: Nopen, Nclose
      logical, intent(in)     :: qprint,QGHO
      logical, intent(inout)  :: Qcheck

!Local

! Check the consistency in spin state and number of electrons.
      If (Nspin.EQ.1 .OR. Nspin.EQ.3 .OR. Nspin.EQ.5)THEN

! Make sure we have an even number of electrons
         if ( (Nelec/2)*2 .NE. Nelec ) THEN
           if (qprint) then
             write(6,   &
        '(''QMMM: System specified with odd number of electrons ('',   &
                   i5,'')'')') Nelec
             write(6,'(''QMMM: but odd spin ('',i5,'')'')') Nspin
           end if

           Qcheck=.FALSE.
           return
         end if

      Else if (Nspin.EQ.2 .OR. Nspin.EQ.4 .OR. Nspin.EQ.6) then

! Give warning message to specifying OPEN SHELL SYSTEM. 
!Stop currently here.
         if(qprint) write(6,*)   &
         'Right now, we do not support UHF calculations. Be carefule.'

! Make sure we have an odd number of electrons.`
         if ( (Nelec/2)*2 .EQ. Nelec ) then
           if (qprint) then
             write(6,'(A,A,i5,A)') 'QMMM: ',   &
             'System specified with even number of electrons (',   &
                                    Nelec,')'
             write(6,'(''QMMM: but even spin ('',i3,'')'')') Nspin
           end if

           Qcheck=.FALSE.
           return
         end if
      End if

! Print Spin state
      if (Nspin .EQ. 1) then
         if(qprint) write (6,'(''QMMM: SINGLET STATE CALCULATION'')')
         Nopen=0
      else if (Nspin .EQ. 2) then
         if(qprint) write (6,'(''QMMM: DOUBLET STATE CALCULATION'')')
         Nopen=1
      else if (Nspin .EQ. 3) then
         if(qprint) WRITE(6,'(''QMMM: TRIPLET STATE CALCULATION'')')
         Nopen=2
      else if (Nspin .EQ. 4) then
         if(qprint) WRITE(6,'(''QMMM: QUARTET STATE CALCULATION'')')
         Nopen=3
      else if (Nspin .EQ. 5) then
         if(qprint) WRITE(6,'(''QMMM: QUINTET STATE CALCULATION'')')
         Nopen=4
      else if (Nspin .EQ. 6) then
         if(qprint) WRITE(6,'(''QMMM: SEXTET STATE CALCULATION'')')
         Nopen=5
      ENDIF

      Nclose = Nelec/2
      If (Nopen .GT. 0) Then
          Nclose = Nclose - Nopen/2
          if ( (Nclose+Nopen) .GT. Norbs ) then
             if(qprint) then
                if(nclose.gt.0) write(6,'(A,A,I3,A)') 'QMMM: ',  &
                  'Number of doubly occupied (',Nclose,') orbitals'
                if(nopen .gt.0) write(6,'(A,A,I3,A)') 'QMMM: ',   &
                  'Number of singly occupied (',Nopen,') levels'
                if(norbs .gt.0) write(6,'(A,A,I5,A)') 'QMMM: ',   &
                  'The total number of orbitals (', Norbs,').'   
                write(6,'(''QMMM: Errors in the number of orbitals.'')')
             end if

             Qcheck = .FALSE.
             return
          end if

          if(qprint) then
             write(6,'(''QMMM: ROHF CALCULATION'')')
             write(6,'(''QMMM: THERE ARE'',I3,   &
                       '' DOUBLY OCCUPIED ORBITALS'')') Nclose
             write(6,'(''QMMM: AND'')')
             write(6,'(''QMMM:          '',i3,   &
                       '' SINGLY OCCUPIED ORBITALS'')') Nopen
          end if

! Temporal comment for GHO-ROHF calculation, which aren't tested.
! Maybe later, UHF will be implemented.
          if(QGHO) then
             if(qprint) then
                write(6,'(A)')   &
           '**********************************************************'
                write(6,'(A)')   &
           'QMMM: ROHF CALCULATION ARE NOT IMPLEMENTED WITH GHO METHOD'
                write(6,'(A)')   &
           '**********************************************************'
             end if

             Qcheck = .FALSE.
             return
          end if
      Else
          if(qprint) write(6,'(A,I3)') &
           'QMMM: RHF CALCULATION, NO. OF DOUBLY OCCUPIED ORBITALS: ',   &
             Nclose
      End if

      return
      END SUBROUTINE


      SUBROUTINE qm2_setup_orb_exp

  use qmmm_module, only : qm2_params, qmmm_struct
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions

      implicit none

!Local
      real(chm_real)  :: ALLC(6,6,2),ALLZ(6,6,2)
      integer :: i, ni, nqn, j, k, l
      integer :: ier=0
      real(chm_real)  :: xi

!For pre-computing the overlap equations used in energy and 
! derivative calculation.
      real(chm_real), pointer :: atom_orb_cc_s_by_type(:,:) => NULL()
      real(chm_real), pointer :: atom_orb_zz_s_by_type(:,:) => NULL()
      real(chm_real), pointer :: atom_orb_cc_p_by_type(:,:) => NULL()
      real(chm_real), pointer :: atom_orb_zz_p_by_type(:,:) => NULL()
      real(chm_real)        :: atom_orb_cc_s_x_s, atom_orb_zz_s_x_s,   &
                         atom_orb_cc_s_x_p
      real(chm_real)        :: atom_orb_zz_s_x_p, atom_orb_cc_p_x_p,   &
                         atom_orb_zz_p_x_p
      real(chm_real)        :: atom_orb_zz_one_s_a_s, atom_orb_zz_one_s_a_p,   &
                         atom_orb_zz_one_p_a_p
      real(chm_real)        :: atom_orb_sp_eqn, atom_orb_pp_eqn
      real(chm_real)        :: temp_real

!This routine fills atom_orb_cc_s, atom_orb_cc_p, atom_orb_zz_s, 
!atom_orb_zz_p with the STO-6G orbital expansion data.
!It then de-allocates the no longer needed 
!qm2_params%s_orb_exp and p_orb_exp. 
!For this reason it should be called once and only once per run.

!SET-UP THE STEWART'S STO-6G EXPANSIONS
!                                      1S
      ALLZ(1,1,1) =2.310303149D01
      ALLZ(2,1,1) =4.235915534D00
      ALLZ(3,1,1) =1.185056519D00
      ALLZ(4,1,1) =4.070988982D-01
      ALLZ(5,1,1) =1.580884151D-01
      ALLZ(6,1,1) =6.510953954D-02
      ALLC(1,1,1) =9.163596280D-03
      ALLC(2,1,1) =4.936149294D-02
      ALLC(3,1,1) =1.685383049D-01
      ALLC(4,1,1) =3.705627997D-01
      ALLC(5,1,1) =4.164915298D-01
      ALLC(6,1,1) =1.303340841D-01

!                                      2S 
      ALLZ(1,2,1) =2.768496241D01
      ALLZ(2,2,1) =5.077140627D00
      ALLZ(3,2,1) =1.426786050D00
      ALLZ(4,2,1) =2.040335729D-01
      ALLZ(5,2,1) =9.260298399D-02
      ALLZ(6,2,1) =4.416183978D-02
      ALLC(1,2,1) =-4.151277819D-03
      ALLC(2,2,1) =-2.067024148D-02
      ALLC(3,2,1) =-5.150303337D-02
      ALLC(4,2,1) =3.346271174D-01
      ALLC(5,2,1) =5.621061301D-01
      ALLC(6,2,1) =1.712994697D-01

!                                     2P
      ALLZ(1,2,2) =5.868285913D00
      ALLZ(2,2,2) =1.530329631D00
      ALLZ(3,2,2) =5.475665231D-01
      ALLZ(4,2,2) =2.288932733D-01
      ALLZ(5,2,2) =1.046655969D-01
      ALLZ(6,2,2) =4.948220127D-02
      ALLC(1,2,2) =7.924233646D-03
      ALLC(2,2,2) =5.144104825D-02
      ALLC(3,2,2) =1.898400060D-01
      ALLC(4,2,2) =4.049863191D-01
      ALLC(5,2,2) =4.012362861D-01
      ALLC(6,2,2) =1.051855189D-01

!                                      3S
      ALLZ(1,3,1) =3.273031938D00
      ALLZ(2,3,1) =9.200611311D-01
      ALLZ(3,3,1) =3.593349765D-01
      ALLZ(4,3,1) =8.636686991D-02
      ALLZ(5,3,1) =4.797373812D-02
      ALLZ(6,3,1) =2.724741144D-02
      ALLC(1,3,1) =-6.775596947D-03
      ALLC(2,3,1) =-5.639325779D-02
      ALLC(3,3,1) =-1.587856086D-01
      ALLC(4,3,1) =5.534527651D-01
      ALLC(5,3,1) =5.015351020D-01
      ALLC(6,3,1) =7.223633674D-02

!                                     3P
      ALLZ(1,3,2) =5.077973607D00
      ALLZ(2,3,2) =1.340786940D00
      ALLZ(3,3,2) =2.248434849D-01
      ALLZ(4,3,2) =1.131741848D-01
      ALLZ(5,3,2) =6.076408893D-02
      ALLZ(6,3,2) =3.315424265D-02
      ALLC(1,3,2) =-3.329929840D-03
      ALLC(2,3,2) =-1.419488340D-02
      ALLC(3,3,2) =1.639395770D-01
      ALLC(4,3,2) =4.485358256D-01
      ALLC(5,3,2) =3.908813050D-01
      ALLC(6,3,2) =7.411456232D-02

!                                     4S
      ALLZ(1,4,1) = 1.365346D+00
      ALLZ(2,4,1) = 4.393213D-01
      ALLZ(3,4,1) = 1.877069D-01
      ALLZ(4,4,1) = 9.360270D-02
      ALLZ(5,4,1) = 5.052263D-02
      ALLZ(6,4,1) = 2.809354D-02
      ALLC(1,4,1) = 3.775056D-03
      ALLC(2,4,1) =-5.585965D-02
      ALLC(3,4,1) =-3.192946D-01
      ALLC(4,4,1) =-2.764780D-02
      ALLC(5,4,1) = 9.049199D-01
      ALLC(6,4,1) = 3.406258D-01

!                                     4P
      ALLC(1,4,2) =-7.052075D-03
      ALLC(2,4,2) =-5.259505D-02
      ALLC(3,4,2) =-3.773450D-02
      ALLC(4,4,2) = 3.874773D-01
      ALLC(5,4,2) = 5.791672D-01
      ALLC(6,4,2) = 1.221817D-01
      ALLZ(1,4,2) = 1.365346D+00
      ALLZ(2,4,2) = 4.393213D-01
      ALLZ(3,4,2) = 1.877069D-01
      ALLZ(4,4,2) = 9.360270D-02
      ALLZ(5,4,2) = 5.052263D-02
      ALLZ(6,4,2) = 2.809354D-02

!                                     5S
      ALLZ(1,5,1) = 7.701420258D-01
      ALLZ(2,5,1) = 2.756268915D-01
      ALLZ(3,5,1) = 1.301847480D-01
      ALLZ(4,5,1) = 6.953441940D-02
      ALLZ(5,5,1) = 4.002545502D-02
      ALLZ(6,5,1) = 2.348388309D-02
      ALLC(1,5,1) = 1.267447151D-02
      ALLC(2,5,1) = 3.266734789D-03
      ALLC(3,5,1) =-4.307553999D-01
      ALLC(4,5,1) =-3.231998963D-01
      ALLC(5,5,1) = 1.104322879D+00
      ALLC(6,5,1) = 4.368498703D-01

!                                      5P
      ALLZ(1,5,2) = 7.701420258D-01
      ALLZ(2,5,2) = 2.756268915D-01
      ALLZ(3,5,2) = 1.301847480D-01
      ALLZ(4,5,2) = 6.953441940D-02
      ALLZ(5,5,2) = 4.002545502D-02
      ALLZ(6,5,2) = 2.348388309D-02
      ALLC(1,5,2) =-1.105673292D-03
      ALLC(2,5,2) =-6.243132446D-02
      ALLC(3,5,2) =-1.628476766D-01
      ALLC(4,5,2) = 3.210328714D-01
      ALLC(5,5,2) = 6.964579592D-01
      ALLC(6,5,2) = 1.493146125D-01

!                                      6S
      ALLZ(1,6,1) = 5.800292686D-01
      ALLZ(2,6,1) = 2.718262251D-01
      ALLZ(3,6,1) = 7.938523262D-02
      ALLZ(4,6,1) = 4.975088254D-02
      ALLZ(5,6,1) = 2.983643556D-02
      ALLZ(6,6,1) = 1.886067216D-02
      ALLC(1,6,1) = 4.554359511D-03
      ALLC(2,6,1) = 5.286443143D-02
      ALLC(3,6,1) =-7.561016358D-01
      ALLC(4,6,1) =-2.269803820D-01
      ALLC(5,6,1) = 1.332494651D+00
      ALLC(6,6,1) = 3.622518293D-01

!                                      6P
      ALLZ(1,6,2) = 6.696537714D-01
      ALLZ(2,6,2) = 1.395089793D-01
      ALLZ(3,6,2) = 8.163894960D-02
      ALLZ(4,6,2) = 4.586329272D-02
      ALLZ(5,6,2) = 2.961305556D-02
      ALLZ(6,6,2) = 1.882221321D-02
      ALLC(1,6,2) = 2.782723680D-03
      ALLC(2,6,2) =-1.282887780D-01
      ALLC(3,6,2) =-2.266255943D-01
      ALLC(4,6,2) = 4.682259383D-01
      ALLC(5,6,2) = 6.752048848D-01
      ALLC(6,6,2) = 1.091534212D-01


!Allocate the local arrays
      allocate (atom_orb_cc_s_by_type(6,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_cc_s_by_type')

      allocate (atom_orb_zz_s_by_type(6,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_zz_s_by_type')

      allocate (atom_orb_cc_p_by_type(6,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_cc_p_by_type')

      allocate (atom_orb_zz_p_by_type(6,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_zz_p_by_type')

      do i=1,qmmm_struct%qm_ntypes
         ni = qmmm_struct%qm_type_id(i)

! get the principal quantum number for QM atom
         If (NI.LT.2) then
           NQN=1
         Else if (NI.LT.10) then
           NQN=2
         Else if (NI.LT.18) then
           NQN=3
         Else if (NI.LT.36) then
           NQN=4
         Else if (NI.LT.54) then
           NQN=5
         Else if (NI.LT.85) then
           NQN=6
         Else if (NI.EQ.85 .OR. NI.EQ.86) then ! Special section for 
!                                              ! Adjusted connection atom
           NQN=2                               ! and GHO boundary atom
         End if

!All types have s orbitals
         XI = qm2_params%s_orb_exp_by_type(ni) *   &
              qm2_params%s_orb_exp_by_type(ni)
         do j=1,6
            atom_orb_cc_s_by_type(j,i)=ALLC(j,NQN,1)
            atom_orb_zz_s_by_type(j,i)=ALLZ(j,NQN,1)*XI
         end do
 
!do p orbs even if atom type doesn't have p orbs, 
!they just won't be used.
         XI = qm2_params%p_orb_exp_by_type(ni) *   &
              qm2_params%p_orb_exp_by_type(ni)
         do j=1,6
            atom_orb_cc_p_by_type(j,i)=ALLC(j,NQN,2)
            atom_orb_zz_p_by_type(j,i)=ALLZ(j,NQN,2)*XI
         end do
      end do

!Deallocate the original orbital exponent parameters 
!as they are no longer needed from this point on.
      deallocate (qm2_params%p_orb_exp_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp','p_orb_exp_by_type')

      deallocate (qm2_params%s_orb_exp_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp','s_orb_exp_by_type')

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!Now, lets pre-compute orbital interactions to save time later.
! All of them might be related to overlap integrals

!Allocate the memory required
      allocate (qm2_params%atom_orb_zz_sxs_over_sas(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_zz_sxs_over_sas')

      allocate (qm2_params%atom_orb_zz_sxp_over_sap(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_zz_sxp_over_sap')

      allocate (qm2_params%atom_orb_zz_pxp_over_pap(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_zz_pxp_over_pap')

      allocate (qm2_params%atom_orb_ss_eqn(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp','atom_orb_ss_eqn')

      allocate (qm2_params%atom_orb_sp_ovlp(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp','atom_orb_sp_ovlp')

      allocate (qm2_params%atom_orb_pp_ovlp_inj(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_pp_ovlp_inj')

      allocate (qm2_params%atom_orb_pp_ovlp_ieqj1(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_pp_ovlp_ieqj1')

      allocate (qm2_params%atom_orb_pp_ovlp_ieqj2(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_pp_ovlp_ieqj2')

      allocate (qm2_params%atom_orb_ss_eqn_adb(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_ss_eqn_adb')

      allocate (qm2_params%atom_orb_sp_eqn_xy(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp','atom_orb_sp_eqn_xy')

      allocate (qm2_params%atom_orb_sp_eqn_xx1(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_sp_eqn_xx1')

      allocate (qm2_params%atom_orb_sp_eqn_xx2(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_sp_eqn_xx2')

      allocate (qm2_params%atom_orb_pp_eqn_xxy1(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_pp_eqn_xxy1')

      allocate (qm2_params%atom_orb_pp_eqn_xxy2(6,6,   &
                qmmm_struct%qm_ntypes,qmmm_struct%qm_ntypes),   &
                stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_setup_orb_exp', &
                             'atom_orb_pp_eqn_xxy2')

  
!Ross Walker - The following pre-computation saves a LOT 
!              of time in 
!              calculating the overlap energy and derivatives.
!              This routine should only be called once. 

!Do p orbital expansions even when the system doesn't have 
!p orbitals. This is just to make things simpler,
!the result of the p orbital calculations on systems without
!p orbitals will not be used, but we won't reference the
!value so it shouldn't matter. 
!
!Note we have to put in checks in this situation to avoid 
!divide by zeros.
      do k=1,qmmm_struct%qm_ntypes
        do l=1,qmmm_struct%qm_ntypes
          do i=1,6
            do j=1,6
              atom_orb_cc_s_x_s = atom_orb_cc_s_by_type(i,k)*   &
                                  atom_orb_cc_s_by_type(j,l)
              atom_orb_zz_s_x_s = atom_orb_zz_s_by_type(i,k)*   &
                                  atom_orb_zz_s_by_type(j,l)
              atom_orb_cc_s_x_p = atom_orb_cc_s_by_type(i,k)*   &
                                  atom_orb_cc_p_by_type(j,l)
              atom_orb_zz_s_x_p = atom_orb_zz_s_by_type(i,k)*   &
                                  atom_orb_zz_p_by_type(j,l)
              atom_orb_cc_p_x_p = atom_orb_cc_p_by_type(i,k)*   &
                                  atom_orb_cc_p_by_type(j,l)
              atom_orb_zz_p_x_p = atom_orb_zz_p_by_type(i,k)*   &
                                  atom_orb_zz_p_by_type(j,l)

              temp_real = atom_orb_zz_s_by_type(i,k)+   &
                          atom_orb_zz_s_by_type(j,l)
              if (temp_real .NE. zero) atom_orb_zz_one_s_a_s =   &
                                                   one/temp_real

              temp_real = atom_orb_zz_s_by_type(i,k)+   &
                          atom_orb_zz_p_by_type(j,l)
              if (temp_real .NE. zero) atom_orb_zz_one_s_a_p =   &
                                                   one/temp_real

              temp_real = atom_orb_zz_p_by_type(i,k)+   &
                          atom_orb_zz_p_by_type(j,l)
              if (temp_real .NE. zero) atom_orb_zz_one_p_a_p =   &
                                                   one/temp_real

              qm2_params%atom_orb_zz_sxs_over_sas(i,j,k,l)=   &
                        atom_orb_zz_s_x_s * atom_orb_zz_one_s_a_s
              qm2_params%atom_orb_zz_sxp_over_sap(i,j,k,l)=   &
                        atom_orb_zz_s_x_p * atom_orb_zz_one_s_a_p
              qm2_params%atom_orb_zz_pxp_over_pap(i,j,k,l)=   &
                        atom_orb_zz_p_x_p * atom_orb_zz_one_p_a_p

!SQRT((two*SQRT(APB)*AMB)**3)*qm2_params%atom_orb_cc_s_x_s(i,j,k,l)
              qm2_params%atom_orb_ss_eqn(i,j,k,l)   &
                              = sqrt((two*sqrt(atom_orb_zz_s_x_s)   &
                               *atom_orb_zz_one_s_a_s)**3)   &
                               *atom_orb_cc_s_x_s 

!SQRT((two*SQRT(APB)*AMB)**3)*atom_orb_cc_s_x_p
              atom_orb_sp_eqn   &
                              = sqrt((two*sqrt(atom_orb_zz_s_x_p)   &
                               *atom_orb_zz_one_s_a_p)**3)   &
                               *atom_orb_cc_s_x_p

!SQRT((two*SQRT(APB)*AMB)**3)*atom_orb_cc_p_x_p
              atom_orb_pp_eqn   &
                              = sqrt((two*sqrt(atom_orb_zz_p_x_p)   &
                               *atom_orb_zz_one_p_a_p)**3)   &
                               *atom_orb_cc_p_x_p

!TWO*atom_orb_zz_s_by_type(K,qmitype)*SQRT(atom_orb_zz_p_by_type(L,qmjtype))
!      *atom_orb_zz_one_s_a_p*atom_orb_sp_eqn
!Used in gover for Si-Pj overlap energy and -Sj-Pi overlap.
              qm2_params%atom_orb_sp_ovlp(i,j,k,l)   &
                              = two*atom_orb_zz_s_by_type(i,k)   &
                               *sqrt(atom_orb_zz_p_by_type(j,l))   &
                               *atom_orb_zz_one_s_a_p*atom_orb_sp_eqn

!-FOUR*sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p* 
!      qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)* 
!           atom_orb_pp_eqn
!Used in gover for Pi-Pj overlap energy when i!=j
              qm2_params%atom_orb_pp_ovlp_inj(i,j,k,l)   &
                        =-four*sqrt(atom_orb_zz_p_x_p)   &
                         *atom_orb_zz_one_p_a_p   &
                         *qm2_params%atom_orb_zz_pxp_over_pap(i,j,k,l)   &
                         *atom_orb_pp_eqn

!-FOUR*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p*
!      qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)
!Used in gover for Pi-Pj overlap energy when i==j
              qm2_params%atom_orb_pp_ovlp_ieqj1(i,j,k,l)   &
                        =-four*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)   &
                         *atom_orb_zz_one_p_a_p   &
                         *qm2_params%atom_orb_zz_pxp_over_pap(i,j,k,l)

!TWO*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p 
!Used in gover for Pi-Pj overlap energy when i==j
              qm2_params%atom_orb_pp_ovlp_ieqj2(i,j,k,l)   &
                        = two*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)   &
                         *atom_orb_zz_one_p_a_p 

!---specifics for QM-QM derivatives
!-TWO*A2_TO_BOHRS2*qm2_params%atom_orb_ss_eqn_adb(i,j,qmitype,qmjtype)
!      *qm2_params%atom_orb_zz_sxs_over_sas(i,j,qmitype,qmjtype)
!Used for S-S overlap in QM-QM derivatives
              qm2_params%atom_orb_ss_eqn_adb(i,j,k,l)    &
                        =-two*A2_TO_BOHRS2   &
                         *qm2_params%atom_orb_ss_eqn(i,j,k,l)   &
                         *qm2_params%atom_orb_zz_sxs_over_sas(i,j,k,l)

!-four*A3_TO_BOHRS3*qm2_params%atom_orb_zz_sxp_over_sap(i,j,qmitype,qmjtype)**2* 
!      (one/(SQRT(atom_orb_zz_p_by_type(J,qmjtype))))*atom_orb_sp_eqn 
!Used for S-P overlap in QM-QM derivatives where P... /= axis
                 if (atom_orb_zz_p_by_type(j,l) .NE. zero) then
                     temp_real = one/SQRT(atom_orb_zz_p_by_type(j,l))
                 else
                     temp_real = zero
                 endif 
              qm2_params%atom_orb_sp_eqn_xy(i,j,k,l)   &
                     =-four*A3_TO_BOHRS3   &
                      *qm2_params%atom_orb_zz_sxp_over_sap(i,j,k,l)**2   &
                      *(temp_real*atom_orb_sp_eqn)
            
!Used for S-P overlap in QM-QM derivatives where P... == axis
                 if (atom_orb_zz_p_by_type(j,l) .NE. zero) then
                     temp_real = one/SQRT(atom_orb_zz_p_by_type(j,l))
                 else 
                     temp_real = zero
                 end if
              qm2_params%atom_orb_sp_eqn_xx1(i,j,k,l)    &
                     = two*A_TO_BOHRS   &
                      *qm2_params%atom_orb_zz_sxp_over_sap(i,j,k,l)   &
                      *temp_real*atom_orb_sp_eqn
              qm2_params%atom_orb_sp_eqn_xx2(i,j,k,l)   &
                     = four*A3_TO_BOHRS3   &
                      *qm2_params%atom_orb_zz_sxp_over_sap(i,j,k,l)**2   &
                      *temp_real*atom_orb_sp_eqn

!Used for P-P overlap in QM-QM derivatives 
!     where P... = P... /= axis as 3.0D*xxx when P... = P... == axis
                 if (atom_orb_zz_p_x_p .NE. zero) then
                     temp_real = one/SQRT(atom_orb_zz_p_x_p)
                 else
                     temp_real = zero
                 end if

!-four*A2_TO_BOHRS2*(ADB_array(inner_index)**2)*(one/(SQRT(atom_orb_zz_p_x_p)))*
!      atom_orb_pp_eqn
              qm2_params%atom_orb_pp_eqn_xxy1(i,j,k,l)   &
                  =-four*A2_TO_BOHRS2   &
                   *(qm2_params%atom_orb_zz_pxp_over_pap(i,j,k,l)**2)   &
                   *temp_real*atom_orb_pp_eqn

!eight*A2_TO_BOHRS2*A2_TO_BOHRS2*(ADB_array(inner_index)**3)* 
!     (one/(SQRT(atom_orb_zz_p_x_p)))*atom_orb_pp_eqn
              qm2_params%atom_orb_pp_eqn_xxy2(i,j,k,l)   &
                  = eight*A2_TO_BOHRS2*A2_TO_BOHRS2   &
                   *(qm2_params%atom_orb_zz_pxp_over_pap(i,j,k,l)**3)   &
                   *temp_real*atom_orb_pp_eqn
            end do
          end do
        end do
      end do

!Finally deallocate no longer needed arrays
      deallocate (atom_orb_zz_p_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp', &
                             'atom_orb_zz_p_by_type')

      deallocate (atom_orb_cc_p_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp', &
                             'atom_orb_cc_p_by_type')

      deallocate (atom_orb_zz_s_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp', &
                             'atom_orb_zz_s_by_type')

      deallocate (atom_orb_cc_s_by_type, stat = ier)
      if(ier.ne.0) call Aass(0,'qm2_setup_orb_exp', &
                             'atom_orb_cc_s_by_type')

      RETURN
      END SUBROUTINE qm2_setup_orb_exp


      SUBROUTINE qm_nb_list_allqm(natom,mminb) 
!
! QM_region - MM_Atom based pair list - NON PERIODIC
! Note:       periodic QMMM list is now built in 
!             qm_fill_qm_xcrd_periodic (RCW+MC)
!             This will be changed (namkh)
!
! This routine calculates a qm_mm_pair_list which is of the form
! listtype atom atom atom ...
! for qmmm_struct%nquant atoms (I.e each QM atom excluding any 
! link atoms
!
! The list of MM atoms that interact with each QM atom is based
! on the cut off distance. Each QM atom will use the identical list
! since to be included an MM atom only needs to be within cut 
! of any QM atom
!
! qm_mm_pair_list comes from qmmm_module.

  use qmmm_module, only : qmmm_nml,qmmm_struct
  use chm_kinds
  use qm2_double
  use stream

      implicit none

!Passed in
      integer, intent(in) :: natom
      integer, intent(in) :: mminb(*)

!Local Variables
      integer :: j,m,n1,n2,jj
      integer :: nquant, nfirst
      integer :: mmatm

      n1 = 0                          ! index of qm_mm_pair_list 
!                                     ! to which we will put the 
!                                     ! current MM atom
      nquant=qmmm_struct%nquant_nlink ! Now in CHARMM:
!                                     ! Original in AMBER : 
!                                     !      nquant=qmmm_struct%nquant

!---------- FIRST PASS -------------------------------------------
      nfirst=0
      qmmm_struct%qm_int_scratch(1:natom)=0 ! Used for a mask
      do m=nquant+1,natom
         mmatm=mminb(m)                     ! Map down into Mopac array.
!                                           ! Should be improved later 
!                                           ! on...namkh_08/04/05
         if(mmatm.GT.0) then
            qmmm_struct%qm_int_scratch(mmatm)=1
            n1 = n1 + 1
            qmmm_struct%qm_mm_pair_list( n1 ) = mmatm
         end if
      end do

!Set QM atoms in qm_int_scratch to zero, just to make sure
      do m=1,nquant
         qmmm_struct%qm_int_scratch(qmmm_nml%iqmatoms(m)) = 0
      enddo

!---------- Second PASS ------------------------------------------
! The following has been done in CHARMM-interface already.
!    Need to check if this is a link atom partner since if it is 
!    we need to keep a copy of it's position in the list.
!    This is because our MM coordinate array
!    in the QM calculation is ordered the same as the pair list.
! However, just keep here for future references

      n2 = 0
      If(qmmm_struct%nlink .GT. 0) then
         If(Prnlev.ge.2) write(6,'(A)')  &
         'Warning: It should not have any link atoms. But it has now'
         do m=1,natom         
             if ( qmmm_struct%qm_int_scratch(m) .EQ. 1 ) then
! include this mm atom in the list
                 n2 = n2+1
                 do jj=1,qmmm_struct%nlink
                    if (qmmm_struct%link_pairs(1,jj) .EQ. m)  &
                                       qmmm_struct%link_pairs(3,jj)=n2
                 end do
             end if
         end do
      End if
      qmmm_struct%qm_mm_pairs = n1

      return
      END SUBROUTINE qm_nb_list_allqm

      SUBROUTINE qm_extract_coords(natom,x,y,z)
!
! This routine copies the qm atom coordinates from the x array 
! and places them in the qm_coords array.
! Note: qm_coords must have previously been allocated enough space.
! Principle Authors of current code:
!           Ross Walker
!           Mike Crowley
!
! Please send all comments or
! queries to: ross@rosswalker.co.uk

  use qmmm_module, only : qmmm_nml,qmmm_struct
  use chm_kinds
  use qm2_double

      implicit none

!Passed in
      integer, intent(in) :: natom
      real(chm_real),intent(in) :: x(*), y(*), z(*)

!Local variables 
      integer :: i, m

! Initial qmmm_struct%nquant_nlink elements of mminb always
! point QM atom number in CHARMM coordinates
      do i=1,qmmm_struct%nquant
          m = qmmm_nml%iqmatoms(i)
          qmmm_struct%qm_coords(1,i) = x(m)
          qmmm_struct%qm_coords(2,i) = y(m)
          qmmm_struct%qm_coords(3,i) = z(m)
      end do

      if(qmmm_struct%nlink .gt. 0) then
         do i=qmmm_struct%nquant+1,qmmm_struct%nquant_nlink
            m=qmmm_nml%iqmatoms(i)
            qmmm_struct%qm_coords(1,i) = x(m)
            qmmm_struct%qm_coords(2,i) = y(m)
            qmmm_struct%qm_coords(3,i) = z(m)
         end do
      end if

      return
      END SUBROUTINE qm_extract_coords

      SUBROUTINE qm_fill_mm_coords(Natom,X,Y,Z,mm_chrgs,qm_xcrd)
!
! QM FILL QM_XCRD NON PERIODIC ( does it matter?)
! Adoped from qm_fill_qm_xcrd

  use qmmm_module, only:qmmm_struct, qmmm_switch
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

! Passed in
      integer, intent(in)  :: natom
      real(chm_real),  intent(in)  :: X(*),Y(*),Z(*),mm_chrgs(*)
      real(chm_real),  intent(out) :: qm_xcrd(4,natom)

! Local variables
      integer :: i,j,ncount, mstart,mstop
      integer :: ISTRT_CHECK                   ! for external function
      integer :: ier=0


! initialize
      mstart = 1
      mstop  = qmmm_struct%qm_mm_pairs
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         qm_xcrd = zero

! get mstart and mstop
         mstart=ISTRT_CHECK(mstop,qmmm_struct%qm_mm_pairs)
      end if
#endif 


!------ fill the pairlist atom coords ---------------------------
      If(qmmm_switch%qswitch) then
! Use swtiching function
         ncount = 0
         do i = mstart,mstop         !   1,qmmm_struct%qm_mm_pairs
            j = qmmm_struct%qm_mm_pair_list(i)
            qm_xcrd(1,i) = x(j)
            qm_xcrd(2,i) = y(j)
            qm_xcrd(3,i) = z(j)

! scale charge for switching
            qm_xcrd(4,i) = qmmm_switch%scale(j)*mm_chrgs(j)

!!
!! gradient contribution
!! -pack the array ...
!!
!!           if(i.gt.j) write(6,*)'Error in qm_fill_mm_coords to', 
!!    *              'construct Switching arrays.'
!!           qmmm_switch%scmask(i)     = qmmm_switch%scmask(j)
!!           qmmm_switch%dxqmmm(1:6,i) = qmmm_switch%dxqmmm(1:6,j)

         end do

! allocate/deallocate necessary memory for switching on cutoff
         if(qmmm_switch%qalloc_mem) then
            if(associated(qmmm_switch%dxqmmm_elec))   &
                       Deallocate(qmmm_switch%dxqmmm_elec, stat=ier)
            if(ier.ne.0) call Aass(0,'qm_fill_mm_coords','dxqmmm_elec')

            if(associated(qmmm_switch%dxmm_wrk))      &
                       Deallocate(qmmm_switch%dxmm_wrk, stat=ier)
            if(ier.ne.0) call Aass(0,'qm_fill_mm_coords','dxmm_wrk')

            Allocate(qmmm_switch%dxqmmm_elec(11,   &
                     qmmm_switch%nswitch_atom_old,   &
                     qmmm_struct%nquant_nlink),   &
                     stat=ier)
            if(ier.ne.0) call Aass(1,'qm_fill_mm_coords','dxqmmm_elec')

            Allocate(qmmm_switch%dxmm_wrk(3,   &
                     qmmm_switch%nswitch_atom_old),   &
                     stat=ier)
            if(ier.ne.0) call Aass(1,'qm_fill_mm_coords','dxmm_wrk')
         end if
      Else
         do i = mstart,mstop         ! 1,qmmm_struct%qm_mm_pairs
            j = qmmm_struct%qm_mm_pair_list(i)
            qm_xcrd(1,i) = x(j)
            qm_xcrd(2,i) = y(j)
            qm_xcrd(3,i) = z(j)
            qm_xcrd(4,i) = mm_chrgs(j)
         end do

         ! for dummy memory allocation for dxqmmm_elec and dxmm_wrk
         if(.not.associated(qmmm_switch%dxqmmm_elec)) then
            Allocate(qmmm_switch%dxqmmm_elec(11,1,1),stat=ier)
            if(ier.ne.0) call Aass(1,'qm_fill_mm_coords','dxqmmm_elec')
         end if
         if(.not.associated(qmmm_switch%dxmm_wrk)) then
            Allocate(qmmm_switch%dxmm_wrk(3,1),stat=ier)
            if(ier.ne.0) call Aass(1,'qm_fill_mm_coords','dxmm_wrk')
         end if
      End if

#if KEY_PARALLEL==1
      if(numnod.gt.1) call GCOMBR(qm_xcrd,4*natom)
#endif 

      return
      END SUBROUTINE qm_fill_mm_coords

      SUBROUTINE qm_print_dyn_mem(natom,npairs)
!
! This routine prints a summary of the dynamic
! memory allocated for use in the QM calculation.
! Written by Ross Walker, TSRI, 2004
! Assumes real(chm_real) is real(chm_real) = 8 bytes.
!

  use qmmm_module, only : qmmm_nml,qmmm_struct, qm2_struct,   &
                              qm2_params, qm2_rij_eqns,   &
                              qm2_ghos, qmmm_ewald, qmmm_switch
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations

      implicit none

!Passed in
      integer, intent(in) :: natom, npairs

!Local
      integer           :: total_memory, element_memory
      real(chm_real),parameter:: bytes_to_mb = 1.0D0/(1024D0*1024D0)
      integer           :: bytes_per_int, bytes_per_real,   &
                           bytes_per_logical

      total_memory = zero
      bytes_per_int = bit_size(element_memory)/8 
      bytes_per_real = 8        !Assume size of real(chm_real) is 8 bytes
      bytes_per_logical = 1     !Assume size of logical is a single byte

      write(6,'(/"| QMMM: Estimated QM Dynamic Memory Usage")')
      write(6,   &
      '("| QMMM: ---------------------------------------------------")')

!1) QM atom type memory: qm_atom_type + qm_type_id
      element_memory = size(qmmm_struct%qm_atom_type)*bytes_per_real+size(qmmm_struct%qm_type_id)*bytes_per_real
      write(6,'("| QMMM:              QM Atom Type Info : ",i12," bytes")') element_memory 
      total_memory = total_memory + element_memory

!2) QM resp charge memory: Not used in CHARMM

!3) QM atom number list: iqmatoms
      element_memory = size(qmmm_nml%iqmatoms)*bytes_per_int
      write(6,'("| QMMM:            QM Atom Number List : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!4) QM Link Atom Pairs: link_pairs
      if(qmmm_struct%nlink .gt. 0) then
         element_memory = size(qmmm_struct%link_pairs)*bytes_per_int
         write(6,'("| QMMM:                Link Atom Pairs : ",i12, " bytes")') element_memory
         total_memory = total_memory + element_memory
      end if

!5) Peptide Link identity: peptide_links
      if(qmmm_nml%peptide_corr) then
         element_memory = size(qm2_struct%peptide_links)*bytes_per_int
         write(6,'("| QMMM:       Peptide Linkage Identity : ",i12," bytes")') element_memory
         total_memory = total_memory + element_memory
      end if

!6) QM atomic number list: iqm_atomic_numbers
      element_memory = size(qmmm_struct%iqm_atomic_numbers)*bytes_per_int
      write(6,'("| QMMM:          QM Atomic Number List : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!7) QM-MM pair list: qm_mm_pair_list
!Pair list is always allocated as at least 1 even if natom = nquant.
      element_memory = size(qmmm_struct%qm_mm_pair_list)*bytes_per_int
      write(6,'("| QMMM:                QM-MM Pair List : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!8) QM Atom mask: atom_mask
      element_memory = size(qmmm_struct%atom_mask)*bytes_per_logical
      write(6,'("| QMMM:                   QM Atom Mask : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!9) QM coordinates array: qm_coords x 4
!   include the qm_xcrd(4,natom) array here
      element_memory = size(qmmm_struct%qm_coords)*bytes_per_real+4*natom*bytes_per_real
      write(6,'("| QMMM:           QM Coordinate Arrays : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!10) Scaled MM Charge Array: scaled_mm_charges
      element_memory = size(qmmm_struct%scaled_mm_charges)*bytes_per_real
      write(6,'("| QMMM:         Scaled MM Charge Array : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory
  
!11) QM Force Arrays: dxyzqm + dxyzcl
!    Include the local force matrices, dxyzqm and dxyzcl here as well
      element_memory =3*qmmm_struct%nquant_nlink*bytes_per_real+3*natom*bytes_per_real
      write(6,'("| QMMM:                QM Force Arrays : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!12) Density matrix: den_matrix
      element_memory = size(qm2_struct%den_matrix)*bytes_per_real
      write(6,'("| QMMM:                 Density Matrix : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!13) Density Matrix Copies: old_den_matrix + old2_den_matrix
      element_memory = (size(qm2_struct%old_den_matrix)+size(qm2_struct%old2_den_matrix))*bytes_per_real
      write(6,'("| QMMM:          Density Matrix Copies : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!14) Time-reversible BOMD, auxiliary density + old auxiliary density matrices.
      if(qmmm_nml%tr_bomd) then
         element_memory = (size(qm2_struct%density_p_new)+size(qm2_struct%density_p_old))*bytes_per_real
         write(6,'("| QMMM:    TR-BOMD Auxiliary Densities : ",i12," bytes")') element_memory
         total_memory = total_memory + element_memory
      end if

!15) Fock2 Density Matrix: fock2(nquant_nlink*16)
      element_memory = qmmm_struct%nquant_nlink*16*bytes_per_real
      write(6,'("| QMMM: Fock2 Density Matrix Workspace : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!16) Fock Matrix: fock_matrix
      element_memory = size(qm2_struct%fock_matrix)*bytes_per_real
      write(6,'("| QMMM:                    Fock Matrix : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!17) Eigen Vector Storage: eigen_vectors
      element_memory = size(qm2_struct%eigen_vectors)*bytes_per_real
      write(6,'("| QMMM:           Eigen Vector Storage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!18) QM-MM Elec Repulsion Integrals: qmmm_erep_incore
      if (qmmm_nml%qmmm_erep_incore) then
         element_memory = ( MIN( npairs*qmmm_struct%nquant_nlink+npairs &
                                ,qmmm_struct%nquant_nlink*(natom-qmmm_struct%nquant_nlink) ) )*4*bytes_per_real
      else
         element_memory = 0
      end if
      write(6,'("| QMMM: QM-MM Elec Repulsion Integrals : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!19) QM-QM Elec Repulsion Integrals: qmqm_erep_incore
      if (qmmm_nml%qmqm_erep_incore) then
         element_memory = size(qm2_struct%qm_qm_e_repul)*bytes_per_real
      else
         element_memory = 0
      end if
      write(6,'("| QMMM: QM-QM Elec Repulsion Integrals : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!20) QM 2-Elec Repulsion Integrals: qm_qm_2e_repul
      element_memory = size(qm2_struct%qm_qm_2e_repul)*bytes_per_real
      write(6,'("| QMMM:  QM 2-Elec Repulsion Integrals : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!21) 1-Electron Matrix: hmatrix
      element_memory = size(qm2_struct%hmatrix)*bytes_per_real
      write(6,'("| QMMM:              1-Electron Matrix : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!22) GHO Boundary method: PHO/PBHO/FAOA/FAOB
      if (qm2_ghos%QGHO) then
         element_memory = size(qm2_ghos%PHO ) + size(qm2_ghos%CAHB) + size(qm2_ghos%DAHB) &
                        + size(qm2_ghos%FAHB) + size(qm2_ghos%PAHB)
         if(qm2_ghos%UHFGHO) then
            element_memory = element_memory + size(qm2_ghos%PBHO) + size(qm2_ghos%CBHB) &
                           + size(qm2_ghos%DBHB) + size(qm2_ghos%FBHB) + size(qm2_ghos%PBHB)
         end if
         element_memory = element_memory * bytes_per_real
      else
         element_memory = 0
      end if
      write(6,'("| QMMM:      GHO Gradient memory usage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!23) QM/MM-Ewald summation: scf_mchg/Kvec/Ktable/qmktable/empot/eslf/d_ewald_mm/qmqmerfcx_data
      if (qmmm_ewald%QEwald) then
! Basic memory 
          element_memory =( size(qmmm_ewald%scf_mchg) + size(qmmm_ewald%Kvec) + size(qmmm_ewald%empot) &
                          + size(qmmm_ewald%eslf) )*bytes_per_real
          if (qmmm_ewald%nexl_atm .gt. 0) then
             element_memory = element_memory + (size(qmmm_ewald%exl_xyz) + size(qmmm_ewald%dexl_xyz))*bytes_per_real &
                            + size(qmmm_ewald%nexl_index)*bytes_per_int
          end if
          if (qmmm_ewald%erfcx_incore) element_memory = element_memory + size(qmmm_ewald%qmqmerfcx_data)*bytes_per_real
          write(6,'("! QMMM:QM/MM-Ewald K vector and others : ",i12," bytes")') element_memory
          total_memory = total_memory + element_memory

! K Table memory
          element_memory = (size(qmmm_ewald%Ktable)+size(qmmm_ewald%qmktable))*bytes_per_real
          write(6,'("! QMMM:    QM/MM-Ewald K Tables arrays : ",f12.3," Mb   ")')element_memory * bytes_to_mb
          total_memory = total_memory + element_memory

! Reciprocal gradient contribution
          element_memory = size(qmmm_ewald%d_ewald_mm)*bytes_per_real
          write(6,'("! QMMM:Recip. QM/MM-Ewald Grad. arrays : ",i12," bytes")') element_memory  
          total_memory = total_memory + element_memory
      end if

!24) Switching function for QM-MM interaction at cutoff region: 
!                                             scmask/scale/dxqmmm/dxqmmm_elec
      if(qmmm_switch%qswitch) then
! real memory
         element_memory =( size(qmmm_switch%dxqmmm) + size(qmmm_switch%dxqmmm_elec) + size(qmmm_switch%scale)  +  &
                           size(qmmm_switch%dxmm_wrk) + size(qmmm_switch%dxyz_qm) ) * bytes_per_real
         element_memory = element_memory +(  size(qmmm_switch%iqmatm) + size(qmmm_switch%iqmgrp)   &
                                           + size(qmmm_switch%immatm) + size(qmmm_switch%immgrp) )*bytes_per_int
         element_memory = element_memory + size(qmmm_switch%scmask)*bytes_per_logical
         write(6,'(" QMMM:QM-MM switching at cutoff region: ",i12," bytes")') element_memory  
         total_memory = total_memory + element_memory
      end if

!25) QM parameter space: all
      element_memory = size(qm2_params%core_chg) +   &
                       size(qm2_params%betasas) +   &
                       size(qm2_params%betasap) +   &
                       size(qm2_params%betapap) +   &
                       size(qm2_params%atom_orb_zz_sxs_over_sas) +   &
                       size(qm2_params%atom_orb_zz_sxp_over_sap) +   &
                       size(qm2_params%atom_orb_zz_pxp_over_pap) +   &
                       size(qm2_params%atom_orb_ss_eqn) +   &
                       size(qm2_params%atom_orb_sp_ovlp) +   &
                       size(qm2_params%atom_orb_pp_ovlp_inj) +   &
                       size(qm2_params%atom_orb_pp_ovlp_ieqj1) +   &
                       size(qm2_params%atom_orb_pp_ovlp_ieqj2) +   &
                       size(qm2_params%atom_orb_ss_eqn_adb) +   &
                       size(qm2_params%atom_orb_sp_eqn_xy)+   &
                       size(qm2_params%atom_orb_sp_eqn_xx1) +   &
                       size(qm2_params%atom_orb_sp_eqn_xx2) +   &
                       size(qm2_params%atom_orb_pp_eqn_xxy1) +   &
                       size(qm2_params%atom_orb_pp_eqn_xxy2) +   &
                       size(qm2_params%orb_elec_ke) +   &
                       size(qm2_params%onec2elec_params) +   &
                       size(qm2_params%multip_2c_elec_params)
      if(associated(qm2_params%s_orb_exp_by_type)) &
         element_memory = element_memory + size(qm2_params%s_orb_exp_by_type)
      if(associated(qm2_params%p_orb_exp_by_type)) &
         element_memory = element_memory + size(qm2_params%p_orb_exp_by_type)

      if ( (qmmm_nml%qmtheory.EQ.PDDGPM3) .OR.   &
           (qmmm_nml%qmtheory.EQ.PDDGMNDO) ) then
         element_memory = element_memory + size(qm2_params%pddge1) +   &
                          size(qm2_params%pddge2) +   &
                          size(qm2_params%pddg_term1) +   &
                          size(qm2_params%pddg_term2) +   &
                          size(qm2_params%pddg_term3) +   &
                          size(qm2_params%pddg_term4)
      end if
      if ( (qmmm_nml%qmtheory.EQ.AM1) .OR. (qmmm_nml%qmtheory.EQ.PM3) .OR. &
           (qmmm_nml%qmtheory.EQ.PDDGPM3) .OR. (qmmm_nml%qmtheory.EQ.PM3CARB1) ) then
         element_memory = element_memory + size(qm2_params%FN1) + size(qm2_params%FN2) + size(qm2_params%FN3)
      end if
      element_memory = element_memory * bytes_per_real
      write(6,'("| QMMM:       real(chm_real) parameter storage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!26) Integer parameter storage: all
      element_memory = size(qm2_params%natomic_orbs) +   &
                       size(qm2_params%orb_loc)+   &
                       size(qm2_params%pascal_tri1) +   &
                       size(qm2_params%pascal_tri2) 
      element_memory = element_memory*bytes_per_int
      write(6,'("| QMMM:      integer parameter storage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!27) QM-QM RIJ Eqns storage: qmqmrij_incore
      if (qmmm_nml%qmqmrij_incore) then
         element_memory = ishft(qmmm_struct%nquant_nlink *qmmm_struct%nquant_nlink,-1)
         if ( (qmmm_nml%qmtheory .EQ. PDDGPM3) .OR. (qmmm_nml%qmtheory .EQ. PDDGMNDO) ) then
            element_memory = element_memory*(QMQMNORIJ+QMQMNOPDDG)*bytes_per_real
         else
            element_memory = element_memory*QMQMNORIJ*bytes_per_real
         end if
      else
         element_memory = 0
      end if
      write(6,'("| QMMM:         QM-QM RIJ Eqns storage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!28) QM-MM RIJ Eqns storage: qmmmrij_incore
      if (qmmm_nml%qmmmrij_incore) then
         element_memory = min(npairs*qmmm_struct%nquant_nlink+npairs, &
                              qmmm_struct%nquant_nlink*(natom-qmmm_struct%nquant_nlink))
         element_memory = QMMMNORIJ*element_memory*bytes_per_real
      else
         element_memory = 0
      end if
      write(6,'("| QMMM:         QM-MM RIJ Eqns storage : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory
   
!29) Pseudo-diagonalization Scratch space: 
!    mat_diag_workspace + pseudo_diag_matrix +qm_real_scratch
!    qm2_struct%mat_diag_workspace
      element_memory=size(qm2_struct%mat_diag_workspace)

      if (qmmm_nml%allow_pseudo_diag) then
         element_memory=element_memory + size(qm2_struct%pseudo_diag_matrix)
      end if
      element_memory  = element_memory + size(qmmm_struct%qm_real_scratch)*bytes_per_real
      write(6,'("| QMMM:          real(chm_real) Scratch arrays : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!30) Integer Scratch arrays: qm_int_scratch
      element_memory = size(qmmm_struct%qm_int_scratch)*bytes_per_int
      write(6,'("| QMMM:         Integer Scratch arrays : ",i12," bytes")') element_memory
      total_memory = total_memory + element_memory

!31) Final array
      write(6,'("| QMMM: ---------------------------------------------------")')
      write(6,'("| QMMM:        Total Dynamic Memory Usage: ",f10.3," Mb")') total_memory * bytes_to_mb

      return
      END SUBROUTINE qm_print_dyn_mem


      SUBROUTINE qm_print_coords(qcheck)
! This routine prints the qm region coordinates.
! Including link atoms
!
! Written by Ross Walker, TSRI, 2004
!

  use qmmm_module, only : qmmm_struct, qm2_ghos
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_elements

      implicit none

!Passed in
      logical, intent(in) :: qcheck

!Local
      integer i,j

      write(6,   &
       '(/,''  QMMM: QM Region Cartesian Coordinates (*=GHO atom) '')')
      write(6,  &
       '(''  QMMM:    NO.'',7X,''ATOM'',9X,''X'',9X,''Y'',9X,''Z'')')

      If( (qm2_ghos%QGHO) .and. (qm2_ghos%nqmlnk.GT.0) ) then
         do i=1,qmmm_struct%nquant_nlink-qm2_ghos%nqmlnk
            WRITE(6,'("  QMMM:",I6,8X,A2,4X,3F10.4)') i,   &
                  element_sym(qmmm_struct%iqm_atomic_numbers(i)),   &
                 (qmmm_struct%qm_coords(j,i),j=1,3)
         end do
         do i=qmmm_struct%nquant_nlink-qm2_ghos%nqmlnk+1,   &
                                              qmmm_struct%nquant_nlink
            WRITE(6,'("  QMMM:",I6,7X,"*",A2,4X,3F10.4)') i,  &
                  'C ',(qmmm_struct%qm_coords(j,i),j=1,3)
         end do
      Else
         do i=1,qmmm_struct%nquant
            WRITE(6,'("  QMMM:",I6,8X,A2,4X,3F10.4)') i,   &
                  element_sym(qmmm_struct%iqm_atomic_numbers(i)),  &
                 (qmmm_struct%qm_coords(j,i),j=1,3)
         end do

         if(qmmm_struct%nlink .gt. 0) then
            do i=qmmm_struct%nquant+1,qmmm_struct%nquant_nlink
               WRITE(6,'("  QMMM:",I6,7X,"*",A2,4X,3F10.4)') i,   &
                     element_sym(qmmm_struct%iqm_atomic_numbers(i)),   &
                    (qmmm_struct%qm_coords(j,i),j=1,3)
            end do
         end if
      End if

      return
      END SUBROUTINE qm_print_coords


      SUBROUTINE qm2_calc_rij_and_eqns(coords, nquant_nlink, crdsmm,  &
                                       natom, npairs, indx_for_qm, &
                                       first_call)
!
! Written by Ross Walker (TSRI, 2005)
!
! This routine should be called on each call to QM_MM. It's 
! purpose is to calculate RIJ for each QM-QM pair.
! It stores this is in the qm2_rij_eqns structure.
! It also calculated a number of RIJ related equations
! that involve exponentials, sqrts etc. In this way they only
! need to be calculated once rather than several times during
! the energy calculation and then in the derivative code.
!

  use qmmm_module, only : qm2_allocate_qm2_qmqm_rij_eqns, &
                              qm2_allocate_qm2_qmmm_rij_eqns,  &
                              qmmm_nml, qmmm_struct, qm2_rij_eqns, &
                              qm2_params
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations

#if KEY_PARALLEL==1
  use parallel  
#endif
!
      implicit none

!Passed in
      integer, intent(in) :: nquant_nlink, natom, npairs, indx_for_qm
      real(chm_real),  intent(in) :: coords(3,nquant_nlink), crdsmm(*)
!                            ! crdsmm array is laid out as 
!                            ! x,y,z,chg,x,y,z,chg...
      logical, intent(in) :: first_call

!Local variables
      integer :: i,j, loop_count, iminus, four_npairs, nqmatm
      logical :: heavy_atom, PDDG_IN_USE, PDDG_IN_USE_keep(2)
      real(chm_real):: r2, rr2, rr, vec1, vec2, vec3, onerij, rij,  &
                 qmi_alpa, qmj_alpa
      real(chm_real):: qmi_oneBDD1, qmi_oneBDD2, qmi_oneBDD3
      real(chm_real):: qmj_oneBDD1
      real(chm_real):: ijBDD1, qmi_DD, qmi_QQ, qmi_QQ2
      real(chm_real):: RRADD, RRMDD, RRAQQ, RRMQQ
      real(chm_real):: xqm, yqm, zqm, RIJ_temp
      integer :: ier=0

      integer :: mstart,mstop,mstart2,mstop2,mmynod,mnumnod
      integer :: ISTRT_CHECK                 ! for external function

      save PDDG_IN_USE_keep

      nqmatm = qmmm_struct%nquant_nlink

      if(first_call) then                   ! only do at first_call

! Memory allocation if needed. This array is a static, 
! since the number of QM-QM interactions does not change,
! thus only need to do the allocation on the first call.
         if (qmmm_nml%qmqmrij_incore) then
             PDDG_IN_USE_keep(indx_for_qm) =  &
                           (qmmm_nml%qmtheory .EQ. PDDGPM3 .OR.  &
                            qmmm_nml%qmtheory .EQ. PDDGMNDO)
             call qm2_allocate_qm2_qmqm_rij_eqns( &
                                          PDDG_IN_USE_keep(indx_for_qm))
         end if

! Allocate memory here for the qmmmrijdata array.
! If this is the first call it will be allocated. 
! It will checked later during energy evaluation
! whether the array be changed or not.
         if (qmmm_nml%qmmmrij_incore) then
            call qm2_allocate_qm2_qmmm_rij_eqns(natom,npairs)
         end if

         return
      end if                         ! only do at first_call and return

      PDDG_IN_USE = PDDG_IN_USE_keep(indx_for_qm)
 
      if (qmmm_nml%qmqmrij_incore) then
! for parallel calculation
#if KEY_PARALLEL==1
         mmynod  = mynod
         mnumnod = numnod
         qm2_rij_eqns%qmqmrijdata = zero
#endif 

         loop_count = 0
         do i=2,nquant_nlink
            qmi_alpa    = qm2_params%cc_exp_params(i)
            qmi_oneBDD1 = qm2_params%multip_2c_elec_params(3,i)

            iminus      = i-1
#if KEY_PARALLEL==1
            if(mmynod .eq. mod(i-2,mnumnod)) then
#endif 
               do j=1,iminus
                  qmj_alpa    = qm2_params%cc_exp_params(j)
                  qmj_oneBDD1 = qm2_params%multip_2c_elec_params(3,j)
                  ijBDD1      = qmi_oneBDD1+qmj_oneBDD1
                  ijBDD1      = ijBDD1*ijBDD1
                  loop_count  = loop_count+1
                  vec1        = coords(1,i)-coords(1,j)
                  vec2        = coords(2,i)-coords(2,j)
                  vec3        = coords(3,i)-coords(3,j)
                  r2          = vec1*vec1+vec2*vec2+vec3*vec3
                  rr2         = r2*A2_TO_BOHRS2
                  qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS2,loop_count)=rr2
                  onerij      = one/sqrt(r2)
                  qm2_rij_eqns%qmqmrijdata(QMQMONERIJ,loop_count)   =  &
                                                                 onerij
                  rij         = r2*onerij                ! one/onerij
                  qm2_rij_eqns%qmqmrijdata(QMQMRIJ,loop_count)      =rij
                  qm2_rij_eqns%qmqmrijdata(QMQMRIJBOHRS,loop_count) =  &
                                                         rij*A_TO_BOHRS
                  qm2_rij_eqns%qmqmrijdata(QMQMEXP1I,loop_count)    =  &
                                                     EXP(-qmi_alpa*rij)
                  qm2_rij_eqns%qmqmrijdata(QMQMEXP1J,loop_count)    =  &
                                                     EXP(-qmj_alpa*rij)
                  qm2_rij_eqns%qmqmrijdata(QMQMSQRTAEE,loop_count)  =  &
                                                   one/sqrt(RR2+ijBDD1)
               end do
#if KEY_PARALLEL==1
            else
               loop_count  = loop_count + iminus
            end if
#endif 
         end do
!Fill the PDDG data if PDDG-PM3 or PDDG-MNDO is requested.
         if (PDDG_IN_USE) then
           loop_count = 0
           do i=2,nquant_nlink
              iminus = i-1
#if KEY_PARALLEL==1
              if(mmynod .eq. mod(i-2,mnumnod)) then
#endif 
                do j=1,iminus
                  loop_count=loop_count+1
                  RIJ_temp =qm2_rij_eqns%qmqmrijdata(QMQMRIJ,loop_count)
                  qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP1,loop_count) =  &
                         EXP( -ten*( RIJ_temp - qm2_params%pddge1(i) -  &
                                     qm2_params%pddge1(j) )**2 )
                  qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP2,loop_count) =  &
                         EXP( -ten*( RIJ_temp - qm2_params%pddge1(i) -  &
                                     qm2_params%pddge2(j) )**2 )
                  qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP3,loop_count) =  &
                         EXP( -ten*( RIJ_temp - qm2_params%pddge2(i) - &
                                     qm2_params%pddge1(j) )**2 )
                  qm2_rij_eqns%qmqmrijdata(QMQMPDDG_EXP4,loop_count) =  &
                         EXP( -ten*( RIJ_temp - qm2_params%pddge2(i) -  &
                                     qm2_params%pddge2(j) )**2 )
                end do
#if KEY_PARALLEL==1
              else
               loop_count  = loop_count + iminus
              end if
#endif 
           end do
         end if
#if KEY_PARALLEL==1
         if(numnod.gt.1) then
            if (PDDG_IN_USE) then
               i=(QMQMNORIJ+QMQMNOPDDG)*loop_count 
            else
               i=QMQMNORIJ*loop_count 
            end if
            call GCOMBR(qm2_rij_eqns%qmqmrijdata,i)
         end if
#endif 
      end if


      if (qmmm_nml%qmmmrij_incore) then

! Check whether to allocate memory here again for 
! the qmmmrijdata array. by comparing
!    min(npairs*qmmm_struct%nquant+npairs,
!        qmmm_struct%nquant*(natom - qmmm_struct%nquant)) 
!    and qm2_rij_eqns%qmmmrij_allocated
! Since the number of pairs can change on each call we should 
! check each time to see if we need to re-allocate.
! 
         if (min(npairs*(nqmatm+1),nqmatm*(natom-nqmatm)) .GT.  &
                                  qm2_rij_eqns%qmmmrij_allocated) then
! We need to de-allocate the array and then call the 
! allocation routine again so it gets allocated larger.
            deallocate(qm2_rij_eqns%qmmmrijdata,stat=ier)
            if(ier.ne.0) call Aass(0,'qm2_calc_rij_and_eqns', &
                                   'qmmmrijdata')

            call qm2_allocate_qm2_qmmm_rij_eqns(natom,npairs)
         end if

! do some initialization
         mstart = 1
         mstop  = npairs
         mstart2= 1
         mstop2 = ishft(npairs,2)
#if KEY_PARALLEL==1
         if(numnod.gt.1) then
            mstart = ISTRT_CHECK(mstop,npairs)
            if(mstart.gt.0) mstart2= ishft(mstart-1,2)+1
            mstop2 = ishft(mstop,2)

            qm2_rij_eqns%qmmmrijdata = zero
         end if
#endif 

         loop_count = 0
!!!      four_npairs=ishft(npairs,2) !*4
         do i=1,nqmatm
            heavy_atom = (qm2_params%natomic_orbs(i) .GT. 1)
            qmi_alpa   = qm2_params%cc_exp_params(i)
            qmi_DD     = qm2_params%multip_2c_elec_params(1,i)
            qmi_QQ     = qm2_params%multip_2c_elec_params(2,i)*two
            qmi_oneBDD1= qm2_params%multip_2c_elec_params(6,i)
            qmi_oneBDD2= qm2_params%multip_2c_elec_params(7,i)
            qmi_oneBDD3= qm2_params%multip_2c_elec_params(8,i)
            qmi_QQ2    = qmi_QQ*qmi_QQ+qmi_oneBDD3
            xqm        = coords(1,i)
            yqm        = coords(2,i)
            zqm        = coords(3,i)

! RCW: We have duplication of code below but avoids us 
!      having to have the if statement inside the inner loop
           if (heavy_atom) then
#if KEY_PARALLEL==1
              loop_count = loop_count + (mstart-1)  ! befor loop (for loop_count)
#endif 
              do j=mstart2,mstop2,4                 ! 1,four_npairs,4
                 loop_count = loop_count+1
                 vec1       = xqm-crdsmm(j)
                 vec2       = yqm-crdsmm(j+1)
                 vec3       = zqm-crdsmm(j+2)
                 r2         = vec1*vec1+vec2*vec2+vec3*vec3
                 rr2        = r2*A2_TO_BOHRS2
                 onerij     = one/sqrt(r2)
                 qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count) =   &
                                                                 onerij
                 rij        = r2*onerij                   ! one/onerij
                 qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)    = rij
                 rr         = rij*A_TO_BOHRS
                 qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)   =  &
                                                     exp(-qmi_alpa*rij)
                 qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)   =  &
                                                      exp(-ALPH_MM*rij)
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,loop_count)=  &
                                              one/sqrt(RR2+qmi_oneBDD1)

! Heavy atom specific stuff
                 RRADD     = RR+qmi_DD
                 RRADD     = RRADD*RRADD+qmi_oneBDD2
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRADDADE,  &
                                          loop_count) = one/sqrt(RRADD)
                 RRMDD     = RR-qmi_DD
                 RRMDD     = RRMDD*RRMDD+qmi_oneBDD2
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMDDADE,  &
                                          loop_count) = one/sqrt(RRMDD)
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRR2AQE,   &
                                          loop_count) = one/SQRT(RR2+   &
                                                           qmi_oneBDD3)
                 RRAQQ     = RR+qmi_QQ
                 RRAQQ     = RRAQQ*RRAQQ+qmi_oneBDD3
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRAQQAQE,  &
                                          loop_count) = one/SQRT(RRAQQ)
                 RRMQQ     = RR-qmi_QQ
                 RRMQQ     = RRMQQ*RRMQQ+qmi_oneBDD3
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRMQQAQE,  &
                                          loop_count) = one/SQRT(RRMQQ)
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTRRAQQ2AQE, &
                                          loop_count)= one/SQRT(RR2 +  &
                                                               qmi_QQ2)
! End Heavy atom specific stuff
              end do
#if KEY_PARALLEL==1
              loop_count = loop_count + (npairs-mstop)   ! after loop (for loop_count)
#endif 
           else ! (heavy_atom)
#if KEY_PARALLEL==1
              loop_count = loop_count + (mstart-1)  ! befor loop (for loop_count)
#endif 
              do j=mstart2,mstop2,4                 ! 1,four_npairs,4
                 loop_count = loop_count+1
                 vec1       = xqm-crdsmm(j)
                 vec2       = yqm-crdsmm(j+1)
                 vec3       = zqm-crdsmm(j+2)
                 r2         = vec1*vec1+vec2*vec2+vec3*vec3
                 rr2        = r2*A2_TO_BOHRS2
                 onerij     = one/sqrt(r2)
                 qm2_rij_eqns%qmmmrijdata(QMMMONERIJ,loop_count) =  &
                                                                  onerij
                 rij=r2*onerij                              ! one/onerij
                 qm2_rij_eqns%qmmmrijdata(QMMMRIJ,loop_count)    = rij
                 qm2_rij_eqns%qmmmrijdata(QMMMEXP1,loop_count)   =  &
                                                      exp(-qmi_alpa*rij)
                 qm2_rij_eqns%qmmmrijdata(QMMMEXP2,loop_count)   =  &
                                                      exp(-ALPH_MM*rij)
                 qm2_rij_eqns%qmmmrijdata(QMMMSQRTAEE,loop_count)=  &
                                               one/sqrt(RR2+qmi_oneBDD1)
              end do
#if KEY_PARALLEL==1
              loop_count = loop_count + (npairs-mstop)   ! after loop (for loop_count)
#endif 
           end if ! (heavy_atom)
         end do

#if KEY_PARALLEL==1
         if(numnod.gt.1 .and. loop_count.gt.0)  &
            call GCOMBR(qm2_rij_eqns%qmmmrijdata,QMMMNORIJ*loop_count)
#endif 
      end if

      return
      END SUBROUTINE qm2_calc_rij_and_eqns

!  temporary uncommented MKL part
!!#ifndef MKL
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE vdinvsqrt( n, x, y )
! vectorized inverse square root

  use chm_kinds
  use qm2_double
  use qm2_constants

      implicit none

!Passed in
      integer  :: n
      real(chm_real) :: x(n), y(n)

!Local variables
      integer  :: i
  
!    temporary uncommented MASSLIB part 
!!#  ifdef MASSLIB
!!      call vrsqrt ( y, x, n )
!!#  else
        y(1:n) = one/sqrt( x(1:n) )
!!#  endif
   
      return
      END SUBROUTINE vdinvsqrt 
!--------------------------------------------------------------
!!#endif

      SUBROUTINE Vdcos( n, x, y)
! Vectorize Cosine routine

  use chm_kinds
  use qm2_double

      implicit none

! Passed in
      integer      :: n
      real(chm_real)     :: x(n), y(n)

!    temporary uncommented MASSLIB part
!!#  ifdef MASSLIB
!!      call vcos( y, x, n)
!!#  else
        y(1:n) = cos(x(1:n))
!!#  endif

      return
      END SUBROUTINE Vdcos
!--------------------------------------------------------------

      SUBROUTINE Aass( condition, routine_name, variable_name)
! Assertion failure reporter 

  use chm_kinds
  use stream
!
#if KEY_PARALLEL==1
  use parallel  
#endif

      implicit none

      integer      :: condition
      character(*) :: routine_name
      character(*) :: variable_name

      if(condition.eq.1) then              ! allocation
#if KEY_PARALLEL==1
         write(6,*)'Allocation failed in variable ',variable_name, &
                   ' of routine ', routine_name,' in node',MYNOD,'.'
#else /**/
         write(6,*)'Allocation failed in variable ',variable_name,   &
                   ' of routine ', routine_name,'.'
#endif 
      else if(condition.eq.0) then         ! deallocation
#if KEY_PARALLEL==1
         write(6,*)'Deallocation failed in variable ',variable_name, &
                   ' of routine ', routine_name,' in node',MYNOD,'.'
#else /**/
         write(6,*)'Deallocation failed in variable ',variable_name,   &
                   ' of routine ', routine_name,'.'
#endif 
      else
#if KEY_PARALLEL==1
          write(6,*)'Wrong call of Aass routine from ', &
          routine_name,' in node',MYNOD,'.'
#else /**/
          write(6,*)'Wrong call of Aass routine from ',   &
          routine_name,'.'
#endif 
      end if

      CALL WRNDIE(-5,'<Aass>','Memory failure. CHARMM will stop.')

      return
      END SUBROUTINE Aass
!--------------------------------------------------------------

!--------------------------------------------------------------
      INTEGER FUNCTION ISTRT_CHECK(mstop,NCNT)
!!
!!    for parallel jobs.
!!    compute mstart and mstop for each node.
!!
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel  
#endif

      implicit none

      integer:: mstop,ncnt

      integer:: mstart,ncnt2

!
#if KEY_PARALLEL==1 /*paramain*/
      if(QMPI) then                  ! protect for MPI-pi calc.
         mstart = 1
         mstop  = ncnt
      else
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/

      if(ncnt.ge.numnod) then 
         ncnt2 = ncnt/NUMNOD
         mstart =  mynod*ncnt2 + 1
         mstop  = (mynod+1)*ncnt2

!!!      if(mstart.gt.ncnt) mstart=ncnt  ! for start
!!!      if(mstop .gt.ncnt) mstop =ncnt  ! for end
         if(MYNOD.EQ.(NUMNOD-1)) mstop =ncnt
      else
         if(MYNOD.eq.0) then
            mstart = 1
            mstop  = ncnt
         else
            mstart = 0
            mstop  = 0
         end if
      end if

#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
      mstart = 1
      mstop  = ncnt

#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
      end if                         ! protect for MPI-pi calc.

#else /*   (paramain)*/

      mstart = 1
      mstop  = ncnt

#endif /*  (paramain)*/

      ISTRT_CHECK = mstart

      RETURN
      END 
!--------------------------------------------------------------
#endif /* (mainsquatn)*/
      SUBROUTINE validate_qm_atoms_BLANK
!
! dummy routine for compilation
!
      RETURN
      END

