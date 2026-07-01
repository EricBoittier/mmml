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

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_INITIALIZE(Natom,Nqmatm,QMMethod,Ncharge,Nspin,   &
                              NZUNC,igmsel,   &
                              Qminb,rscfc,CUT,QGHOL,QMEwd,NOPMEwald,QMSWITch, &
                              QMMM_NoGauss,QMMM_NoDiis,Qcheck, &
                              qmlay_local,Qdual_check, &
                              q_tr_bomd,number_scf_cycle)
!
! Reads the qmmm namelist and calls the qmmm memory allocation 
! routines
! based on SUBROUTINE read_qmmm_namelist_and_allocate( neb, ih, 
! ix, x, cut )
!

  use qmmm_module, only : Get_Type_qmmm_nml,Get_Type_qmmm_struct, &
                              Get_Type_qmmm_switch,Get_Type_qm2_ghos, &
                              Get_Type_qmmm_ewald, &
                              qmsort,allocate_qmmm, &
                              qmmm_nml,qmmm_nml_p,qmmm_nml_r, &
                              qmmm_struct,qmmm_struct_p,qmmm_struct_r, &
                              qmmm_switch,qmmm_switch_p,qmmm_switch_r, &
                              qm2_ghos,qm2_ghos_p,qm2_ghos_r, &
                              qmmm_ewald,qmmm_ewald_p,qmmm_ewald_r
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_parameters
  use stream
      implicit none
      INTEGER, PARAMETER :: max_quantum_atoms=500
                                  ! maxmimum qm atoms

!Passed in
      integer,intent(in)    :: Natom,Nqmatm,QMMethod, &
                               Ncharge,Nspin,number_scf_cycle
      integer,intent(in)    :: nzunc(*),igmsel(*),Qminb(*)
      logical,intent(inout) :: Qcheck
      logical,intent(in)    :: QGHOL,QMEwd,NOPMEwald,QMSWITch, &
                               QMMM_NoGauss,QMMM_NoDiis, &
                               qmlay_local,Qdual_check(2), &
                               q_tr_bomd
      real(chm_real), intent(in)  :: rscfc, cut

!local
!1) something needed
      real(chm_real) :: qmcut            ! local copied to qmmm_nml%qmcut
                                   ! - specified cutoff to use for QM-MM 
                                   !   electrostatics.
                                   ! Default = same as regular MM cutoff.
      integer :: qmtheory          ! local copied to qmmm_nml%qmtheory
                                   ! - flag for level of theory to use 
                                   !        for QM region
                                   !   = 2 (AM1 : default)
                                   !   = 1 (PM3)  / 3 (MNDO)  / 
                                   !     4 (PDDG-PM3) / 5 (PDDG-MNDO)
      integer :: qmcharge          ! local copied to qmmm_nml%qmcharge
                                   ! - value of charge on QM system
      integer :: spin              ! local copied to qmmm_nml%spin
                                   ! - spin state of system
                                   !   = 1 (singlet, default)
      integer :: qmqmdx            ! local copied to qmmm_nml%qmqm_analyt
                                   !   = 1 (analytical, default)
                                   !   = 2 (numerical QM-QM derivatives 
                                   !        in qm2)
      integer :: verbosity         ! local copied to qmmm_nml%verbosity
                                   ! - Controls amount of info about QM 
                                   !   part of calc that is printed
                                   !   = 0 (dafault)
      integer :: tight_p_conv      ! local copied to 
                                   !  qmmm_nml%tight_p_conv
                                   ! - Controls convergence of density 
                                   !   matrix.
                                   !   = 1 (use 0.05*sqrt(SCFCRT),default)
                                   !   = 2 (use SCFCRT value)
      real(chm_real) :: scfconv          ! local copied to qmmm_nml%scfconv
                                   ! - Convergence criteria for SCF 
                                   !   routine.
                                   !   = 1.0D-8 (default), Minimum 
                                   !     (tightest criteria) = 1.0D-16
      integer :: itrmax            ! Local copied to qmmm_nml%itrmax
                                   ! - Maximum number of scf cycles to run
                                   !   = 1000 (default)
      integer :: qmqmrij_incore    ! Flag to store rij between qm-qm 
                                   ! pairs and related equations in 
                                   ! memory.
                                   !   = 1  (store in memory, default)
                                   !   = 0  (calc on fly)
      integer :: qmmmrij_incore    ! Flag to store rij between qm-mm 
                                   ! pairs and related equations in 
                                   ! memory.
                                   !   = 1 (store in memory, default)
                                   !   = 0 (calc on fly)
      integer :: qmmm_erep_incore  ! Flag to store QM-MM 1 electron 
                                   ! repulsion integrals in memory.
                                   !   = 1 (store in memory, default)
                                   !   = 0 (calc on fly)
      integer :: qmqm_erep_incore  ! Flag to store QM-QM 1 electron 
                                   ! repulsion integrals in memory.
                                   ! Only available with QM-QM analytical
                                   !  derivatives.
                                   !   = 1 (store in memory, default)
                                   !   = 0 (calc on fly)
      integer :: pseudo_diag       ! Whether to allow pseudo 
                                   ! diagonalisations to be done when 
                                   ! possible in SCF.
                                   !   = 0 (do full diag only, default)
                                   !   = 1 (do pseudo diagonalisations 
                                   !        when possible)
      real(chm_real) :: pseudo_diag_criteria ! Maximum change in density matrix 
                                     ! between successive SCF iterations
                                     ! in which to allow pseudo 
                                     ! diagonalisation.
                                     !  = 0.05 (default)

!2) something not sure
      integer :: printcharges      ! Local copied to
                                   !  qmmm_nml%printcharges as a logical.
                                   !  =1 print mulliken and cm1a and cm2a
                                   !     charges on evety step
                                   !  =0 don't print charges, default
      integer :: peptide_corr      ! Local copied to the logical 
                                   ! qmmm_nml%peptide_corr
                                   ! Add MM correction to peptide 
                                   ! linkages.
                                   !  = 0 No, Default
                                   !  = 1 Yes
      integer :: qmshake           ! Local copied to qmmm_nml%qmshake 
                                   ! - shake QM atoms if ntc>1?

!3) something will be removed
      logical :: mdin_qmmm=.true.  ! Can be deleted this later
      real(chm_real):: lnk_dis           ! Distance from the QM atom to place 
                                   ! link atom.
      integer :: lnk_atomic_no     ! Atomic number of link atom
      integer :: qmgb              ! local copied to qmmm_nml%qmgb
                                   ! - flag for type of GB do with 
                                   !   QM region

! real local variables
      integer :: i,j               ! temporary counter for local loops
      integer :: ifind
      integer :: ier=0

!We need a local scratch array that will be big enough that
!the iqmatoms list never exceeds it
      integer :: iqmatoms( max_quantum_atoms )

! for printing control
      logical :: qprint, qlocal_react

      qprint=.false.
      If(Prnlev.ge.2) qprint=.true.

! 1st is 1st. LOAD paramters into their own array.
      if(.not.q_parm_loaded) call qm2_initialize_elements


! for dual quantum region
! copy from ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_p)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_p)
            call Get_Type_qmmm_switch(qmmm_switch,qmmm_switch_p)
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_p)
            call Get_Type_qmmm_ewald(qmmm_ewald,qmmm_ewald_p)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_r)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_r)
            call Get_Type_qmmm_switch(qmmm_switch,qmmm_switch_r)
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_r)
            call Get_Type_qmmm_ewald(qmmm_ewald,qmmm_ewald_r)
         end if
      end if

!Setup defaults
      qmcut        = cut
      qmtheory     = 2      !(AM1)
      qmcharge     = 0
      spin         = 1
      qmqmdx       = 1
      verbosity    = 0
      tight_p_conv = 0
      scfconv      = 1.0D-8
      printcharges = 0
      peptide_corr = 0
      itrmax       = 1000

      qmshake      = 1

!!     qmmask=''

      iqmatoms(1:max_quantum_atoms) = 0
      qmqmrij_incore       = 1
      qmmmrij_incore       = 1
      qmmm_erep_incore     = 1
      qmqm_erep_incore     = 1
      pseudo_diag          = 1
      pseudo_diag_criteria = 0.05d0

! will be cleaned up
      lnk_dis      = 1.09d0  ! Methyl C-H distance
      lnk_atomic_no= 1       ! Hydrogen
      qmgb         = 0
      mdin_qmmm    =.true.

!****************************************************
!now fill the values passed from CHARMM
      qmtheory = QMMethod
      qmcharge = Ncharge
      spin     = Nspin
      scfconv  = rscfc
!****************************************************

      j=0

! single quantum region or 1st qm region. 
! (for dual quantum region, it is also same.)
      do i=1,natom
         if(igmsel(i).gt.0.and.igmsel(i).lt.5) then
            j=j+1
            iqmatoms(j)=i
         end if
      end do

      qmmm_struct%nquant=j

      if(j.eq.0) then
          if(qprint) write(6,'(A)') 'No qm atoms selected'
          Qcheck=.FALSE.
          RETURN
      end if
      if(j.gt.max_quantum_atoms) then
          if(qprint) write(6,'(A,I5,2X,I5)')  &
                                 'QM atoms should be less than ',  &
                                  max_quantum_atoms
          Qcheck=.FALSE.
          RETURN
      end if
      if(j.ne.Nqmatm) then
          if(qprint) write(6,'(A,I4,2X,I4)')    &
                 'The number of qm atoms are not matched.',j,Nqmatm
          QCHECK=.FALSE.
          RETURN
      end if

!Test to see if QM atom selection is legal.
      call validate_qm_atoms(iqmatoms,qmmm_struct%nquant,natom,Qcheck)
      if(.not. Qcheck) then
        if(qprint) write(6,'(A)')'Error in validate_qm_atoms.'
        return
      end if

!check we don't bust our statically allocated max_quantum_atoms
!ensure the list of qm atoms is sorted numerically
      call qmsort(iqmatoms)     

!***Note***
! GHO atom case: modify iqmatoms array, since GHO atoms should 
! be last atoms
! 
! namkh: I need to make sure it doesn't affect the calculations.
!                       since it override qmsort routine..
      If (QGHOL.and.(qm2_ghos%nqmlnk.GT.0)) then
       do i=1,qmmm_struct%nquant
          j=iabs(Qminb(i))
          iqmatoms(i) = j
       end do
      End if
!***

!will be removed
      qmmm_nml%lnk_dis       = lnk_dis
      qmmm_nml%lnk_atomic_no = lnk_atomic_no
      qmmm_nml%qmgb          = qmgb
      qmmm_nml%qmshake       = qmshake
      qmmm_nml%printcharges  =.true.
      qmmm_nml%peptide_corr = .false.

      if ( printcharges .NE. 1) qmmm_nml%printcharges =.false.
      if ( peptide_corr .NE. 0) qmmm_nml%peptide_corr =.true.

      qmmm_nml%ifqnt                =.TRUE.
      qm2_ghos%QGHO                 = QGHOL           ! For GHO atoms
 
! QM/MM-Ewald case: activate QM/MM-Ewald routine
      qmmm_ewald%QEwald             = QMEwd           ! For QM/MM-Ewald
      qmmm_ewald%QNoPMEwald         = NOPMEwald       ! For QM/MM-Ewald or QM/MM-PMEwald
      ! fix for scf_mchg array, which has to be allocated.
      if( .not.qmmm_ewald%QEwald ) then
         If(associated(qmmm_ewald%scf_mchg)) Deallocate(qmmm_ewald%scf_mchg)
         Allocate(qmmm_ewald%scf_mchg(qmmm_struct%nquant), stat=ier)
         if(ier.ne.0) call Aass(1,'QMMM_INITIALIZE','scf_mchg')
      end if

      qmmm_nml%qmcut                = qmcut
      qmmm_nml%qmcut2               = qmcut*qmcut
      qmmm_nml%qmtheory             = qmtheory
      qmmm_nml%qmcharge             = qmcharge
      qmmm_nml%spin                 = spin
      qmmm_nml%verbosity            = verbosity
      qmmm_nml%itrmax               = itrmax
      qmmm_nml%pseudo_diag_criteria = pseudo_diag_criteria

! QM/MM switching function.
      if(.not.QMSWITch) qmmm_switch%qswitch = QMSWITch  ! for false case... 
                                                        ! for true case, it is 
                                                        ! done elsewhere. 

! QM-MM Gaussian-Gaussian core-core terms.
! Turn on AM1/PM3/PM3CARB1, but can be turned off in the user input.
! PDDGPM3 should be done differently.
      qmmm_nml%QMMM_Gauss =.false.
!     if(.not. QMMM_NoGauss) then
!        if(qmmm_nml%qmtheory.eq.AM1 .or. qmmm_nml%qmtheory.eq.PM3 .or.
!    *      qmmm_nml%qmtheory.eq.PDDGPM3 .or.
!    *      qmmm_nml%qmtheory.eq.PM3CARB1) then
!           qmmm_nml%QMMM_Gauss =.true. 
!        end if
!     end if
      if(.not. QMMM_NoGauss) then
         if(qmmm_nml%qmtheory.eq.AM1 .or. qmmm_nml%qmtheory.eq.PM3 .or. &
            qmmm_nml%qmtheory.eq.PM3CARB1) then
            qmmm_nml%QMMM_Gauss =.true.
         end if
      end if

      qmmm_nml%qmqm_analyt = .true.   ! analytical QM-QM dericatives in qm2
      if (qmqmdx .NE. 1) qmmm_nml%qmqm_analyt = .false.
                                      ! numerical QM-QM derivatives in qm2

      qmmm_nml%tight_p_conv = .false. ! Loose density matrix convergence
                                      ! (0.05*sqrt(SCFCRT))
      if (tight_p_conv .EQ. 1) qmmm_nml%tight_p_conv = .true.
                                      ! Tight density matrix convergence 
                                      ! (SCFCRT)

!Write a warning about excessively tight convergence requests.
      if ( scfconv .LT. 1.0D-12 .and. qprint ) then
       write(6,'(" QMMM: WARNING - SCF Conv = ",G8.2)') scfconv
       write(6,*) "QMMM:           ",    &
                  "There is a risk of convergence problems when the"
       write(6,*) "QMMM:           ",    &
                  "requested convergence is less that 1.0D-12 kcal/mol."
      end if
      qmmm_nml%scfconv = scfconv

      qmmm_nml%qmqmrij_incore = .true.  ! only available with analytical
                                        ! derivatives.
      if ( qmqmrij_incore .EQ. 0 .OR. qmqmdx .EQ. 2)  &
           qmmm_nml%qmqmrij_incore = .false.

      qmmm_nml%qmmmrij_incore = .true.
      if ( qmmmrij_incore .EQ. 0 ) qmmm_nml%qmmmrij_incore = .false.

      qmmm_nml%qmmm_erep_incore = .true.
      if ( qmmm_erep_incore .EQ. 0) qmmm_nml%qmmm_erep_incore = .false.

      qmmm_nml%qmqm_erep_incore = .true. ! only available with analytical
                                         ! derivatives.
      if ( qmqm_erep_incore .EQ. 0 .OR. qmqmdx .EQ. 2)  &
           qmmm_nml%qmqm_erep_incore = .false.

      qmmm_nml%allow_pseudo_diag = .false.
      if ( pseudo_diag .EQ. 1 ) qmmm_nml%allow_pseudo_diag = .true.

      qmmm_nml%Q_Diis = .true.                   ! use this as default.
      if(QMMM_NoDiis) qmmm_nml%Q_Diis = .false.  ! turn off when asked.

      qmmm_nml%tr_bomd=.false.                   ! default is off
      qmmm_nml%num_scf_cycle=qmmm_nml%itrmax+1   ! should be larget than itrmax
      if(q_tr_bomd) then
         qmmm_nml%tr_bomd=q_tr_bomd
         qmmm_nml%num_scf_cycle=number_scf_cycle
      end if

!At this point we know nquant and natom so we can allocate our 
!arrays that depend on nquant or natom
!Note non master mpi threads need to call this allocation routine 
!manually themselves.
      call allocate_qmmm( natom )

!Copy the iqmatoms from our static local array to our allocated array.
      qmmm_nml%iqmatoms(1:qmmm_struct%nquant) =   &
                                      iqmatoms(1:qmmm_struct%nquant)

!Get the atomic numbers
      qmmm_struct%iqm_atomic_numbers(1:qmmm_struct%nquant) =   &
                                      nzunc(1:qmmm_struct%nquant)

!Now we have a list of atom numbers for QM atoms we can build a 
!true false (natom long) list specifying what the quantum atoms
!are. Useful for doing quick .OR. operations against other lists.
!
      qmmm_struct%atom_mask = .false. ! Sets natom long array to false
      do i = 1, qmmm_struct%nquant
       qmmm_struct%atom_mask(qmmm_nml%iqmatoms(i)) = .true.
      end do

!some other crazy initializations, since the link atom has been 
!handled within CHARMM, not new Mopac part, that has been done in
!"identify_link_atoms" and "qm_assign_atom_types."
!
      qmmm_struct%nlink        = 0
      qmmm_struct%nquant_nlink = qmmm_struct%nquant + qmmm_struct%nlink

!Now we should work out the type of each quantum atom present.
!This is used for our arrays of pre-computed parameters. It is
!essentially a re-basing of the atomic numbers and is done to save
!memory. Note: qm_assign_atom_types will allocate the qm_atom_type
!array for us. Only the master calls this routine. All other
!threads get this allocated and broadcast to them by the mpi setup
!routine.
      call qm_assign_atom_types


! for dual quantum region
! copy back into ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml_p,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_p,qmmm_struct)
            call Get_Type_qmmm_switch(qmmm_switch_p,qmmm_switch)
            call Get_Type_qm2_ghos(qm2_ghos_p,qm2_ghos)
            call Get_Type_qmmm_ewald(qmmm_ewald_p,qmmm_ewald)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml_r,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_r,qmmm_struct)
            call Get_Type_qmmm_switch(qmmm_switch_r,qmmm_switch)
            call Get_Type_qm2_ghos(qm2_ghos_r,qm2_ghos)
            call Get_Type_qmmm_ewald(qmmm_ewald_r,qmmm_ewald)
         end if
      end if

      return
      END SUBROUTINE QMMM_INITIALIZE
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_SETUP(MaxAtom,Natom,Indx_mminb, &
                            Mminb,Mminb2,Lperiod,prnlev, &
                            X,Y,Z,Cgm,  &
                            Qcheck,Qfirst)

  use qmmm_module, only : qmmm_nml,qmmm_struct,qm2_ghos
  use chm_kinds
  use qm2_double
      implicit none
!Passed in
      integer, intent(in)    :: MaxAtom,Natom,Indx_mminb
      integer, intent(inout) :: Mminb(MaxAtom,2),Mminb2(MaxAtom,2)
      real(chm_real),intent(in)    :: X(*), Y(*), Z(*), Cgm(*)
      integer, intent(in)    :: Lperiod           ! periodic boundary condition option
                                                  ! 0 : no periodic boundary condition
                                                  ! 1 : charmm IMAGE facility
                                                  ! 2 : charmm PBOUND facility
      integer, intent(in)    :: prnlev            ! print control.
      logical, intent(in)    :: Qfirst            ! logical for first qm/mm setup
      logical, intent(inout) :: Qcheck

!Local
      integer :: i,j
      integer :: ier=0

!
! Only do this once when QM/MM initial setup. Called from routine SQMINI.
!
      If (Qfirst) Then

! allocate necessary memories / Deallocated in deallocate_qmmm
        allocate (qmmm_struct%dxyzqm(3,qmmm_struct%nquant +   &
                                     qmmm_struct%nlink),stat=ier)
        if(ier.ne.0) call Aass(1,'QMMM_SETUP','dxyzqm')

!dxyzcl array only actually needs to be 3,qm_mm_pairs long...
! / Deallocated in deallocate qmmm
        allocate (qmmm_struct%dxyzcl(3,natom),stat=ier)
        if(ier.ne.0) call Aass(1,'QMMM_SETUP','dxyzcl')

!qm_xcrd only actually needs to be 4,qm_mm_pairs long...
! / Dealocated in deallocate qmmm
        allocate (qmmm_struct%qm_xcrd(4,natom),stat=ier)
        if(ier.ne.0) call Aass(1,'QMMM_SETUP','qm_xcrd')

! Stores the REAL and link atom qm coordinates
        allocate (qmmm_struct%qm_coords(3,qmmm_struct%nquant +   &
                                        qmmm_struct%nlink),stat=ier)
        if(ier.ne.0) call Aass(1,'QMMM_SETUP','qm_coords')

! Do the work of allocate_qmmm_pair_list(natom), no need to call it.
! Note: we are overestimating the memory needed here.
        allocate (qmmm_struct%qm_mm_pair_list( (natom -   &
                              qmmm_struct%nquant + 1) ),stat=ier)
        if(ier.ne.0) call Aass(1,'QMMM_SETUP','qm_mm_pair_list')

! Copying Cgm into "scaled_mm_charges" array, to be used in QM/MM-Ewald sum
        qmmm_struct%scaled_mm_charges(1:natom)=Cgm(1:natom)

      End if
!
!


!!
!! Note: for amber qm_mm.f file, there is routine do some periodic 
!! boundary check,
!!       which will not be used in CHARMM, since it is checked in 
!!       parent routines.
!!       Also, non-bonded list generation will only be used when 
!!       atom-based cutoffs are used. Otherwise, it will used
!!       passed Mminb(flags) for that.
!!
!!       Some routines will be kept (commented out) for future 
!!       references.
!!
!!**Adopted from routine qm_mm (qm_mm.f)
!!!    if(periodic ==1) then
!!!  !---- check periodic status and run some checks on the system
!!!      call qm_check_periodic(a,b,c,alpha,beta,gamma 
!!!   *                         ,x,natom,bxbnd,iqmatoms)
!!!      !----If we got this far, we can assume that the QM region is
!!!      !    contiguous and an OK size. Set the origin in the center
!!!      !    of this region.
!!!      !    Move all coords according to the new origin
!!!      !       and wrap all coord into new box.
!!!   !Note also builds list and starts / stops relevant timers.
!!!      call qm_fill_qm_xcrd_periodic(x,natom,a,b,c,bxbnd, 
!!!   *                    qmmm_struct%qm_xcrd,scaled_mm_chrgs, 
!!!   *                    iqmatoms)
!!!    else
!!!  !---------------Not Periodic ----------------------------------
!!!   !---- cutoff based on distance from whole QM region
!!!    call qm_get_boundbox(x,natom,bxbnd,iqmatoms)
!!!    call qm_nb_list_allqm( x, natom, bxbnd)
!!!   !=============================================================
!!!   !               FILL QM Coordinates from crd array
!!!   !=============================================================
!!!   ! extract qm atom coordinates from x
!!!   ! and place in qmmm_struct%qm_coords
!!!    call qm_extract_coords(x)
!!!    call qm_fill_qm_xcrd(x,natom,qmmm_struct%qm_xcrd, 
!!!   *                     scaled_mm_chrgs)
!!!    endif
!!!
!! Fill qm_nb_list generations. 



! The coordinates here already taken cared in CHARMM in case
! we use IMAGE facility. Use image wrap-uped coordinates.
!
! Or,
! do not use Periodic boundary condition
!
      If ((Lperiod .EQ. 1) .or. (Lperiod .EQ. 0)) Then
! Only copy Mminb array relavant to NewMopac code
         call qm_nb_list_allqm(Natom, &
                               Mminb(1:MaxAtom,Indx_mminb:Indx_mminb))

! Get the QM coordinates ready
         call qm_extract_coords(Natom,X,Y,Z)

! Get the MM coordinates ready
         call qm_fill_mm_coords(Natom,X,Y,Z,Cgm,qmmm_struct%qm_xcrd)


      Else If (Lperiod .EQ. 2) Then
!
! Using Charmm Simple PBOUND facility, when computing distances
!  needs to do
! PBCHECK.
!

!***Note
! Only temporary warning.
! -namkh
         If(Prnlev.ge.2) write(6,*)'Warning>",   &
         "Not supported PBOUND facility with this QM/MM yet.'
         
         Qcheck = .FALSE.
         return

! Only copy Mminb array relavant to NewMopac code
         call qm_nb_list_allqm(Natom, &
                               Mminb(1:MaxAtom,Indx_mminb:Indx_mminb))

! Get the QM coordinates ready
         call qm_extract_coords(Natom,X,Y,Z)

! Get the MM coordinates ready
         call qm_fill_mm_coords(Natom,X,Y,Z,Cgm,qmmm_struct%qm_xcrd)

      Else
         If(Prnlev.ge.2) write(6,*)'Bombing> No correct boundary setup.'

         Qcheck = .FALSE.
         return
      End if

! GHO specific setup
! Define hybrid orbital transformation matrix, 0912PJ05
      If ((.not.Qfirst) .and. (qm2_ghos%QGHO) .and.  &
          (qm2_ghos%nqmlnk.gt.0) ) then
       Call HBDEF(X,Y,Z,qm2_ghos%BT,qm2_ghos%BTM,qm2_ghos%DBTMMM,   &
                  qm2_ghos%nqmlnk,   &
                  qm2_ghos%IQLINK,qm2_ghos%JQLINK,qm2_ghos%KQLINK,   &
                  Qcheck)
      End if

!If verbosity is on print the number of QMMM pairs
      if (qmmm_nml%verbosity .GT. 1 .and. prnlev.ge.2)  &
        write(6,*) 'QMMM: No. QMMM Pairs per QM atom: ',   &
                    qmmm_struct%qm_mm_pairs

      return
      END SUBROUTINE QMMM_SETUP
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_PARM_LOAD(qprint,Qcheck,Qdual_check)

  use qmmm_module, only : Get_Type_qmmm_nml,Get_Type_qmmm_struct, &
                              Get_Type_qm2_struct,Get_Type_qm2_params, &
                              Get_Type_qm2_ghos, &
                              qm2_allocate_params1,qm2_allocate_params2, &
                              qm2_allocate_qmqm_e_repul, &
                              qmmm_nml,qmmm_nml_p,qmmm_nml_r, &
                              qmmm_struct,qmmm_struct_p,qmmm_struct_r, &
                              qm2_struct,qm2_struct_p,qm2_struct_r, &
                              qm2_params,qm2_params_p,qm2_params_r, &
                              qm2_ghos,qm2_ghos_p,qm2_ghos_r
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use stream
  use qm2_parameters
      implicit none
!Passed in
      logical, intent(in)    :: qprint
      logical, intent(inout) :: Qcheck
      logical, intent(in)    :: Qdual_check(2)

!Locals
      integer :: i,iqm_atomic
      integer :: n_atomic_orb, first_orb, last_orb
      integer :: nheavy_atoms, nlight_atoms, nelectrons, nopen

! for printing control
!     logical :: qprint

!     qprint=.false.
!     If(Prnlev.ge.2) qprint=.true.

! for dual quantum region
! copy from ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_p)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_p)
            call Get_Type_qm2_struct(qm2_struct,qm2_struct_p)
            call Get_Type_qm2_params(qm2_params,qm2_params_p)
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_p)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_r)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_r)
            call Get_Type_qm2_struct(qm2_struct,qm2_struct_r)
            call Get_Type_qm2_params(qm2_params,qm2_params_r)
            call Get_Type_qm2_ghos(qm2_ghos,qm2_ghos_r)
         end if
      end if


! allocate necessary memories
      call qm2_allocate_params1

!---------- COMMON PARAMS ---------------
! First, load parameters common to all Semi-empirical methods
      qm2_struct%norbs=0
      nheavy_atoms=0
      nelectrons=-qmmm_nml%qmcharge
      qm2_params%tot_heat_form = zero    ! Initialize total heat of formation

      Do i=1,qmmm_struct%nquant_nlink
        iqm_atomic=qmmm_struct%iqm_atomic_numbers(i)

        qm2_params%core_chg(i)=dble(core_chg(iqm_atomic))
        nelectrons            =nelectrons+core_chg(iqm_atomic)
        n_atomic_orb          =natomic_orbs(iqm_atomic)

        if (n_atomic_orb .GT. 1) nheavy_atoms=nheavy_atoms+1

! Check we don't bust any static arrays
        if (n_atomic_orb .GT. MAX_VALENCE_ORBITALS) then
           if(qprint) write (6,*) 'n_atomic_orb of ',n_atomic_orb,   &
               ' exceeds max_valence_orbitals of MAX_VALENCE_ORBITALS'
           CALL WRNDIE(-5,'<SQMINI>',   &
               'qm2_load_params: exceed max valence orbitals.')
        end if

        qm2_params%natomic_orbs(i)=n_atomic_orb
        qm2_params%orb_loc(1,i)   =qm2_struct%norbs+1            !orbital index: start orb_loc(1,i)
        qm2_params%orb_loc(2,i)   =qm2_struct%norbs+n_atomic_orb !               end   orb_loc(2,i)
        qm2_struct%norbs          =qm2_struct%norbs+n_atomic_orb
        qm2_params%tot_heat_form  =qm2_params%tot_heat_form +   &
                                   heat_of_form(iqm_atomic)
      End do

! For GHO boundary atom, substract the 3 electrons from 
! total number of electrons per each GHO atoms
!***Note***
! We don't need this, as far as we use Core charge to be 1 
! for GHO boundary atom
      If(qm2_ghos%QGHO.and.(qm2_ghos%nqmlnk.GT.0)) then
        nelectrons = nelectrons - 3*qm2_ghos%nqmlnk
      End if

!Work out how many 2 electron integrals there will be
      nlight_atoms   =qmmm_struct%nquant_nlink-nheavy_atoms
      qm2_struct%n2el=50*nheavy_atoms*(nheavy_atoms-1) +   &
                     10*nheavy_atoms*nlight_atoms+   &
                     ishft((nlight_atoms*(nlight_atoms-1)),-1)

!QMMM e-repul memory depends on QM-MM pair list size so is
!allocated later on and checked on every call.
! allocate : qm2_struct%qm_qm_2e_repul(:)
!            qm2_struct%qm_qm_e_repul(:,:)
      call qm2_allocate_qmqm_e_repul(qm2_struct%n2el)

!Protect DUMB users from STUPID errors
      if (nelectrons .GT. 2*qm2_struct%norbs) then
       if (qprint) then
        write(6,'(''QMMM: ERROR-number of electrons: '',i5,   &
                  '' is more'')') nelectrons
        write(6,'(''QMMM: than 2xnorbs of: '',i5)') qm2_struct%norbs
        write(6,'(''QMMM: Check qmcharge in qmmm namelist and rerun'')')
        write(6,'(''QMMM: the calculation.'')')
       end if

       Qcheck=.FALSE.
       return
      end if

!Now, we know the number of electrons: nelectrons
!Compute the number of closed/open shells and print Spin state
      Call CheckSpinState(qmmm_nml%spin,nelectrons,nopen,   &
                         qm2_struct%nclosed,   &
                         qm2_struct%norbs,qprint,qm2_ghos%QGHO,Qcheck)

      If(.NOT.Qcheck) then
        if(qprint)  &
           write(6,'(''Check the spin state: CHARMM will stop.'')')
        return
      End if
! number of open and closed orbitals.
      qm2_struct%nopenclosed=nopen+qm2_struct%nclosed

! Allocate full space for setup/energy calculations
      call qm2_allocate_params2

! Guess density
      call qm2_guess_density

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! actual routine loads all parameters
      Qcheck = .TRUE.
      call qm2_load_params(qprint,Qcheck)

      If(.NOT.Qcheck) then
        if(qprint)  &
           write(6,'(''Check the input line: CHARMM will stop.'')')
        return
      End if

!Finally setup the STO-6G orbital expansions and allocate 
!the memory required. 
!Setup the STO-6G orbital expansions and pre-calculate overlaps 
!by type as we can and store these in memory.
!This will help a lot with speed in the energy and derivative code.
      call qm2_setup_orb_exp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! In case of using GHO method, allocate memory 
      If(qm2_ghos%QGHO) Call gho_allocate(qm2_struct%norbs)


! for dual quantum region
! copy back into ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml_p,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_p,qmmm_struct)
            call Get_Type_qm2_struct(qm2_struct_p,qm2_struct)
            call Get_Type_qm2_params(qm2_params_p,qm2_params)
            call Get_Type_qm2_ghos(qm2_ghos_p,qm2_ghos)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml_r,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_r,qmmm_struct)
            call Get_Type_qm2_struct(qm2_struct_r,qm2_struct)
            call Get_Type_qm2_params(qm2_params_r,qm2_params)
            call Get_Type_qm2_ghos(qm2_ghos_r,qm2_ghos)
         end if
      end if

      return
      END SUBROUTINE QMMM_PARM_LOAD
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Distance_setup(Natom,Qfirst,Qdual_check)
! 
! Call qm2_calc_rij_and_eqns in this routine and setup relavant
! equations
!

  use qmmm_module, only: qmmm_nml,qmmm_struct, &
                             qm2_rij_eqns,qm2_params
  use chm_kinds
  use qm2_double
      implicit none
!Passed in
      integer, intent(in)    :: Natom
      logical, intent(in)    :: Qfirst,Qdual_check(2)

!Local variables
      integer                :: indx_for_qm

! Calculate rij and related equations, and store them in memory.
! Also, allocate memory required.
      indx_for_qm = 1
      if(Qdual_check(1).and.Qdual_check(2)) indx_for_qm = 2
      Call qm2_calc_rij_and_eqns(qmmm_struct%qm_coords,   &
                                 qmmm_struct%nquant_nlink,   &
                                 qmmm_struct%qm_xcrd,natom,   &
                                 qmmm_struct%qm_mm_pairs, &
                                 indx_for_qm,   &
                                 Qfirst)

      return
      END SUBROUTINE QMMM_Distance_setup
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_PRNT_MEM(Natom,qprint,Qcheck,Qdual_check)
!
! Print summary of memory usage and QM coordinates
!

  use qmmm_module
  use chm_kinds
      implicit none
!Passed in
      integer, intent(in)    :: Natom
      logical, intent(in)    :: qprint
      logical, intent(inout) :: Qcheck
      logical, intent(in)    :: Qdual_check(2)

! for dual quantum region      
! copy from ... (all)
      if(Qdual_check(1)) call GetQMMMType(Qdual_check(2)) ! get lists.

! Print a summary of memory usage
      if(qprint) call qm_print_dyn_mem(Natom,qmmm_struct%qm_mm_pairs)

! Print the initial QM region coordinates
      if(qprint) call qm_print_coords(Qcheck)

! for dual quantum region
! copy back into ... (all)
      if(Qdual_check(1)) call PutQMMMType(Qdual_check(2))  ! put lists.

      return
      END SUBROUTINE QMMM_PRNT_MEM
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_EWALD_INITIALIZE(Natom,Natqm_1st,Ewmode,Erfmod, &
                                    Igmsel,   &
                                    KmaxX,KmaxY,KmaxZ,KSQmax,      &
                                    Kappa,cginb_pka,   &
                                    QpKa_on,Qcheck,Qdual_check)
!
! Initial setup for QM/MM-Ewald calculations
!

  use qmmm_module, only : Get_Type_qmmm_nml,Get_Type_qmmm_struct, &
                              Get_Type_qmmm_ewald, &
                              qmmm_nml,qmmm_nml_p,qmmm_nml_r,  &
                              qmmm_struct,qmmm_struct_p,qmmm_struct_r,  &
                              qmmm_ewald,qmmm_ewald_p,qmmm_ewald_r
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
! Passed in
      integer, intent(in)    :: Natom,Natqm_1st,Ewmode,Erfmod
      integer, intent(in)    :: Igmsel(*)
      integer, intent(in)    :: KmaxX,KmaxY,KmaxZ,KSQmax
      real(chm_real),  intent(in)  :: Kappa,cginb_pka(*)
      logical, intent(inout) :: Qcheck
      logical, intent(in)    :: QpKa_on,Qdual_check(2)

! Local variables
      integer                :: i, nexcl
      integer                :: ier=0
#if KEY_PARALLEL==1
      integer                :: ntmp_iatotl(numnod)
      integer  :: ISTRT_CHECK                 ! for external function
#endif 

! for dual quantum region
! copy from ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_p)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_p)
            call Get_Type_qmmm_ewald(qmmm_ewald,qmmm_ewald_p)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml,qmmm_nml_r)
            call Get_Type_qmmm_struct(qmmm_struct,qmmm_struct_r)
            call Get_Type_qmmm_ewald(qmmm_ewald,qmmm_ewald_r)
         end if
      end if

! Local check
      If (.NOT. qmmm_ewald%QEwald) then
! It shouldn't happen. Just double check
        write(6,*)'QMMM_EWALD_SETUP> Error in setup QM/Mm-Ewald.'
        write(6,*)'QMMM_EWALD_SETUP> Mostly coding error.'

        Qcheck=.FALSE.
        return
      End if


! Setup for QM/MM-Ewald summation
      qmmm_ewald%Ewmode = Ewmode
      qmmm_ewald%natom  = Natom
      qmmm_ewald%kmaxqx = KmaxX
      qmmm_ewald%kmaxqy = KmaxY
      qmmm_ewald%kmaxqz = KmaxZ
      qmmm_ewald%ksqmaxq= KSQmax
      qmmm_ewald%Erfmod = Erfmod
      qmmm_ewald%kappa  = Kappa

! 
      qmmm_ewald%qpka_calc_on = QpKa_on
      qmmm_ewald%natqm_fix    = Natqm_1st

! for parallel prepare...: it doesn't need to recompute at each loop.
      qmmm_ewald%iastrt = 1          ! starting of do i=1,natom loop
      qmmm_ewald%iafinl = natom      ! ending of the loop
      qmmm_ewald%iatotl = natom      ! total term in do-loop
#if KEY_PARALLEL==1 /*paramain*/

!!#if KEY_PARAFULL==1 /* (parfmain) */
!!#if KEY_PARASCAL==1 /* (parstest) */
!!##ERROR 'Illegal parallel compile options'
!!#endif  /* (parstest) */

!!  this is not true when quantum is called before any
!!  mm energy call earlier.
!!      qmmm_ewald%iastrt=1+IPARPT(MYNOD)
!!      qmmm_ewald%iafinl=IPARPT(MYNODP)

      if(numnod.gt.1)  &
         qmmm_ewald%iastrt=ISTRT_CHECK(qmmm_ewald%iafinl,natom)
      qmmm_ewald%iatotl=qmmm_ewald%iafinl-qmmm_ewald%iastrt+1

! check and iatotl is maximum over the entire node.
! (so, memory allocation over the node are same, but 
!  smaller than numnod=1.)
      if(numnod.gt.1) then
         ntmp_iatotl = 0
         ntmp_iatotl(mynod+1)=qmmm_ewald%iatotl
         call IGCOMB(ntmp_iatotl,numnod)
         do i=1,numnod
            if(ntmp_iatotl(i).gt.qmmm_ewald%iatotl) then
               qmmm_ewald%iatotl=ntmp_iatotl(i)
            end if
         end do
      end if

!!#if KEY_PARASCAL==1 || KEY_SPACDEC==1 /* (parfmain) */
!!      qmmm_ewald%iastrt=1
!!      qmmm_ewald%iafinl=NATOM
!!      qmmm_ewald%iatotl=natom
!!#else /* (parfmain) */
!!##ERROR 'Illegal parallel compile options'
!!#endif /* (parfmain) */
#endif /* (paramain)      */

 
! Whether incore calculation for "(one-erfcx)/rij"
! Set always fault, since it is differ from AMBER code in 
! computing drfc.
! So, inactivate using this one
!
!!!   If (qmmm_nml%qmmm_erep_incore) then
!!!      qmmm_ewald%erfcx_incore=.TRUE.
!!!   Else
 
          qmmm_ewald%erfcx_incore=.FALSE.

!!!   End if
!

! Check how many MM atoms are excluded from QM-MM non-bonded 
! interactions
      nexcl = 0
      Do i=1,natom
        if(Igmsel(i).eq.5) nexcl=nexcl+1
      End do

      qmmm_ewald%nexl_atm =nexcl

      If(qmmm_ewald%nexl_atm .gt. 0) then
! Allocate integer array here
        If(associated(qmmm_ewald%nexl_index))   &
                                     Deallocate(qmmm_ewald%nexl_index)
        Allocate(qmmm_ewald%nexl_index(qmmm_ewald%nexl_atm))
        if(ier.ne.0) call Aass(1,'QMMM_EWALD_INITIALIZE','nexl_index')

! Fill qmmm_ewald%nexl_index array
        nexcl = 0
        do i=1,natom
           if(Igmsel(i).eq.5) then
              nexcl = nexcl + 1
              qmmm_ewald%nexl_index(nexcl)=i
           end if
        end do
      End if

! Now setup for rest part and memory allocations
      Call qm_ewald_setup(qmmm_ewald%kmaxqx,qmmm_ewald%kmaxqy,   &
                         qmmm_ewald%kmaxqz,   &
                         qmmm_ewald%ksqmaxq,qmmm_ewald%totkq,Qcheck)

      If(.not.Qcheck) return

! Now setup for memory allocation
      Call qmmm_ewald_allocate(qmmm_struct%nquant_nlink,Qcheck)

      If(.not.Qcheck) return

! Do initializations.
      qmmm_ewald%scf_mchg   = zero
      qmmm_ewald%Kvec       = zero
      qmmm_ewald%Ktable     = zero
      qmmm_ewald%qmktable   = zero
      qmmm_ewald%empot      = zero
      qmmm_ewald%eslf       = zero
      qmmm_ewald%d_ewald_mm = zero
      if(associated(qmmm_ewald%dexl_xyz)) qmmm_ewald%dexl_xyz   = zero

! for pKa FEP-QM/MM.
      if(qmmm_ewald%qpka_calc_on)  &
         qmmm_ewald%mm_chg_fix_qm(1:qmmm_ewald%natqm_fix) = &
                                   cginb_pka(1:qmmm_ewald%natqm_fix)

! for dual quantum region
! copy back into ...
      if(Qdual_check(1)) then
         if(Qdual_check(2)) then                           ! for 2nd qm region
            call Get_Type_qmmm_nml(qmmm_nml_p,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_p,qmmm_struct)
            call Get_Type_qmmm_ewald(qmmm_ewald_p,qmmm_ewald)
         else                                              ! for 1st qm region
            call Get_Type_qmmm_nml(qmmm_nml_r,qmmm_nml)
            call Get_Type_qmmm_struct(qmmm_struct_r,qmmm_struct)
            call Get_Type_qmmm_ewald(qmmm_ewald_r,qmmm_ewald)
         end if
      end if

      return
      END SUBROUTINE QMMM_EWALD_INITIALIZE
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Ewald_Setup_And_Pot(Indx_mminb,Volume,Recip,X,Y,Z, &
                                          ener_bg_cg_corr,tot_cg_fep,virial, &
                                          QCG_corr,Qcheck)  
!
! Do the Ewald setup: 1) Prepare K-vector and K-tables
!                     2) Compute the Ewald potential on QM atom 
!                        sites
!

  use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_rij_eqns,  &
                              qmmm_ewald
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
! Passed in
      integer, intent(in)    :: Indx_mminb
      real(chm_real),  intent(in)  :: Volume, Recip(6)
      real(chm_real),  intent(in)  :: X(*), Y(*), Z(*)
      real(chm_real)               :: ener_bg_cg_corr(*),tot_cg_fep(*),virial(9)
      logical, intent(inout) :: QCG_corr(*),Qcheck

! Local variables
      integer                :: i, nexcl, itotal,itotkq
      integer                :: ier=0
      real(chm_real)               :: PotBGRND

!!! very important.
! do special with QMPI: if you once turn on this, cannot turn off later!!!!
#if KEY_PARALLEL==1
      if (QMPI) then
         if(qmmm_ewald%iatotl .ne. qmmm_ewald%natom) then
            qmmm_ewald%iastrt = 1
            qmmm_ewald%iafinl = qmmm_ewald%natom
            qmmm_ewald%iatotl = qmmm_ewald%natom
! now, memory allocation.
            If(associated(qmmm_ewald%Ktable)) Deallocate(qmmm_ewald%Ktable)
            if(qmmm_ewald%QNoPMEwald) then
               Allocate(qmmm_ewald%Ktable(6,qmmm_ewald%iatotl, &
                        qmmm_ewald%totkq),stat=ier)              ! iatotl=natom
            else
               Allocate(qmmm_ewald%Ktable(6,1,1),stat=ier)       ! dummy allocation
            end if
            if(ier.ne.0) call Aass(1,'QMMM_Ewald_Setup_And_Pot', &
                                   'Ktable')
         end if
      end if
#endif 

! do initialization
      qmmm_ewald%Ktable     = zero
      qmmm_ewald%qmktable   = zero
      qmmm_ewald%d_ewald_mm = zero

! Copy volume, and reciprocal space lattice vector
! They should be copied every step, since during dynamics 
! volume and lattice vector will change.
      qmmm_ewald%Volume     = volume
      qmmm_ewald%Recip(1:6) = Recip(1:6)

! Copy x,y,z,cgm into exl_xyz array
      If(qmmm_ewald%nexl_atm .gt. 0) then
        do i=1, qmmm_ewald%nexl_atm
           nexcl = qmmm_ewald%nexl_index(i)

           qmmm_ewald%exl_xyz(1,i) = X(nexcl)
           qmmm_ewald%exl_xyz(2,i) = Y(nexcl)
           qmmm_ewald%exl_xyz(3,i) = Z(nexcl)
           qmmm_ewald%exl_xyz(4,i) =    &
                      qmmm_struct%scaled_mm_charges(nexcl)
        end do
      End if

! background charge correction
      If(QCG_corr(Indx_mminb)) then
         PotBGRND=-PIi/(qmmm_ewald%kappa*qmmm_ewald%kappa*Volume)
         ener_bg_cg_corr(Indx_mminb)=half*tot_cg_fep(Indx_mminb)* &
                                     PotBGRND*CCELEC
      End if


! 1) Kvector setup
      Call qm_ewald_calc_kvec(qmmm_ewald%kappa,qmmm_ewald%Volume,   &
                             qmmm_ewald%Recip,   &
                             qmmm_ewald%Kvec,qmmm_ewald%totkq,   &
                             qmmm_ewald%ksqmaxq,   &
                             qmmm_ewald%kmaxqx,qmmm_ewald%kmaxqy,   &
                             qmmm_ewald%kmaxqz,   &
                             Qcheck)
      If(.not.Qcheck) return  ! Check: if wrong, CHARMM should die.


! 2) Ktables setup
      if(qmmm_ewald%QNoPMEwald) then
         itotal=qmmm_ewald%iatotl
         itotkq=qmmm_ewald%totkq
      else
         itotal=1
         itotkq=1
      end if
      Call qm_ewald_calc_ktable(qmmm_ewald%natom,   &
                               qmmm_struct%nquant_nlink,   &
                               qmmm_nml%iqmatoms,   &
                               qmmm_ewald%iastrt,qmmm_ewald%iafinl, &
                               qmmm_ewald%iatotl,itotal,itotkq, &
                               qmmm_ewald%totkq,qmmm_ewald%ksqmaxq,   &
                               qmmm_ewald%kmaxqx,qmmm_ewald%kmaxqy,   &
                               qmmm_ewald%kmaxqz,   &
                               X,Y,Z,qmmm_ewald%Recip,   &
                               qmmm_ewald%Ktable,qmmm_ewald%qmktable,   &
                               qmmm_ewald%QNoPMEwald,Qcheck)
      If(.not.Qcheck) return  ! Check: if wrong, CHARMM should die.

! 3) Ewald potential setup
! Now Compute the Ewald potential on QM atom from all MM atoms.
! Self interaction / Real space contribution / Reciprocal space 
! contribution from "pure" MM atoms
! 
! Note: in case of parallel run, empot is combined over node from
!       the subroutine. (So, each node has same information already.)
      Call qm_ewald_mm_pot(qmmm_ewald%natom,qmmm_struct%nquant_nlink,   &
                           qmmm_ewald%totkq,qmmm_ewald%iatotl,   &
                           qmmm_ewald%iastrt,qmmm_ewald%iafinl,itotal,itotkq, &
                           qmmm_ewald%ksqmaxq,qmmm_ewald%kmaxqx,   &
                           qmmm_ewald%kmaxqy,qmmm_ewald%kmaxqz,   &
                           qmmm_struct%qm_mm_pairs,qmmm_ewald%Erfmod, &
                           qmmm_struct%qm_xcrd,qmmm_struct%qm_coords,   &
                           qmmm_struct%scaled_mm_charges,   &
                           qm2_rij_eqns%qmmmrijdata,   &
                           qmmm_ewald%Kvec,qmmm_ewald%Ktable,   &
                           qmmm_ewald%qmktable,qmmm_ewald%structfac_mm, &
                           qmmm_ewald%empot,qmmm_ewald%kappa,   &
                           qmmm_ewald%QNoPMEwald,qmmm_nml%qmmmrij_incore)

! for the MM excluded list
      If(qmmm_ewald%nexl_atm.gt.0) then
          call qm_ewald_mm_pot_exl(qmmm_struct%nquant_nlink, &
                                   qmmm_ewald%nexl_atm,   &
                                   qmmm_ewald%nexl_index,   &
                                   qmmm_ewald%Erfmod,   &
                                   qmmm_struct%qm_coords,     &
                                   qmmm_ewald%exl_xyz,   &
                                   qmmm_ewald%empot,         &
                                   qmmm_ewald%kappa) 
      End if
                            
      If(.not.Qcheck) return

! From QM image atoms
      Call qm_ewald_qm_pot(qmmm_ewald%natom,qmmm_struct%nquant_nlink,   &
                           qmmm_ewald%totkq,   &
                           qmmm_ewald%ksqmaxq,qmmm_ewald%kmaxqx,   &
                           qmmm_ewald%kmaxqy,qmmm_ewald%kmaxqz,   &
                           qmmm_ewald%Erfmod,qmmm_struct%qm_coords,   &
                           qm2_rij_eqns%qmqmrijdata,   &
                           qmmm_ewald%Kvec,qmmm_ewald%qmktable,   &
                           qmmm_ewald%eslf,   &
                           qmmm_ewald%qmqmerfcx_data,   &
                           qmmm_ewald%kappa,   &
                           qmmm_nml%qmqmrij_incore,   &
                           qmmm_ewald%erfcx_incore,   &
                           qm2_rij_eqns%n1_qmqm,qm2_rij_eqns%n2_qmqm)

! for pme version
      if(.not.qmmm_ewald%QNoPMEwald) then
         call qm_pme_mm_pot(qmmm_ewald%natom,qmmm_struct%nquant_nlink,   &
                            x,y,z,qmmm_struct%scaled_mm_charges,qmmm_ewald%scf_mchg_2,   &
                            qmmm_ewald%Recip,qmmm_ewald%Volume, &
                            qmmm_ewald%empot,   &
                            qmmm_ewald%kappa)
      end if

! debug
!     do i=1,qmmm_struct%nquant_nlink
!        write(6,'(A,I3,2(A,F12.5))')'i=',i,       &
!                  ' eslf=',qmmm_ewald%eslf(i,i),  &
!                  ' empot=',qmmm_ewald%empot(i)
!     end do

      return
      END SUBROUTINE QMMM_Ewald_Setup_And_Pot
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Energy(Escf,Natom,Qfirst,Qcheck,Qdual_check, &
                             q_apply_tr_bomd,i_md_step)
!
! Do the job of routine qm2_energy and it's preceding jobs
! At first call from SQMINI, it only do initial setup part.
! At call from SQMMME, it do energy evaluation.

  use qmmm_module, only : qm2_allocate_qmmm_e_repul,   &
                              qmmm_nml,qmmm_struct,qm2_struct,   &
                              qm2_params,qmmm_ewald
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      real(chm_real),  intent(inout) :: Escf
      integer, intent(in)      :: Natom,i_md_step
      logical, intent(in)      :: Qfirst
      logical, intent(inout)   :: Qcheck
      logical, intent(in)      :: Qdual_check(2)
      logical, intent(in)      :: q_apply_tr_bomd

!Local variables
      integer   :: i, indx_for_qm
      real(chm_real)  :: ANGLE, HTYPE
      integer, save :: i_local_cnt=0

! Check this every time on energy evaluations. Also, 
! do this QM/MM initial setup.
! when Qfirst, allocate QM-MM replusion integrals array 
! big enough such
! that qm_mm_e_repul has a dimension (4,qm_mm_pairs*(nquant)). 
! However, it will check later to avoid blowing up the array size.
      if (qmmm_nml%qmmm_erep_incore)   &
         call qm2_allocate_qmmm_e_repul(Natom,qmmm_struct%qm_mm_pairs,   &
                                        Qfirst)

!***Note
! This part has been moved into "QMMM_Distance_setup" routine.
! The reason is qmqmrijdata and qmmmrijdata are used in 
! routine "QMMM_Ewald_Setup_And_Pot", so it should be done before 
! calling QMMM_Ewald_Setup_And_Pot
!
! Calculate rij and related equations, and store them in memory.
! Also, allocate memory required.
!     call qm2_calc_rij_and_eqns(qmmm_struct%qm_coords,  
!    *                           qmmm_struct%nquant_nlink,  
!    *                           qmmm_struct%qm_xcrd,natom,  
!    *                           qmmm_struct%qm_mm_pairs,  
!    *                           Qfirst)
!***end

! for DIIS converger: initialize the matrices..
      if (qmmm_nml%Q_Diis) then
          qm2_struct%diis_fppf  = zero
          qm2_struct%diis_fock  = zero
          qm2_struct%diis_emat  = zero
          qm2_struct%diis_start =.TRUE.
          qm2_struct%diis_lfock = 0
          qm2_struct%diis_nfock = 0
      end if

! for TR-BOMD: only apply this during MD simulations.
      qm2_struct%q_apply_tr_bomd = q_apply_tr_bomd
      if(qmmm_nml%tr_bomd .and. q_apply_tr_bomd) then
         if(i_md_step.eq.1) then                       ! 1-st MD steps.
            qm2_struct%q_apply_tr_bomd = .false.
         else if(i_md_step.gt.1) then                  ! during MD steps.
!           rho_tilde(n+1)= 2*rho_scf(n) - rho_tilde(n-1)
            qm2_struct%den_matrix=two*qm2_struct%den_matrix &      ! r_curr <= 2*r_curr - r_old
                                  -qm2_struct%density_p_old        !
            qm2_struct%density_p_old=qm2_struct%density_p_new      ! r_old  <= r_new
            qm2_struct%density_p_new=qm2_struct%den_matrix         ! r_new  <= r_curr
            if(i_md_step.eq.2) qm2_struct%q_apply_tr_bomd = .false.
         else
            qm2_struct%q_apply_tr_bomd = .false.
         end if
      end if

! if first call, don't need to do scf calculation.
! Call routine doing compute QM/MM SCF energy 
! Explicit do qm2_energy work here.

      if(.not. Qfirst) then
! Initialisations required on every call...
         escf                    = zero
         qm2_struct%hmatrix      = zero
         qmmm_struct%enuclr_qmqm = zero
         qmmm_struct%enuclr_qmmm = zero

! Get the QM-QM interactions.
!        qm2_struct%hmatrix        = 1-electron matrix
!        qm2_struct%qm_qm_2e_repul = 2-electron repulsion integrals
!        ENUCL                     = core-core repulsion

         CALL qm2_hcore_qmqm(qmmm_struct%qm_coords,qm2_struct%hmatrix, &
                             qm2_struct%qm_qm_2e_repul,  &
                             qmmm_struct%enuclr_qmqm)

! Get the QM-MM interactions.
! add to qm2_struct%hmatrix and qmmm_enuclr:
!
! Updated hmatrix includes interactions between QM electrons
! and partial charges on MM atoms (1-electron terms)
!
! Updated ENUCLR includes interactions between core QM charges 
! and partial atomic MM charges (QM-MM core-core interactions)
! 
         Call qm2_hcore_qmmm(qmmm_struct%qm_coords,qm2_struct%hmatrix,   &
                             qmmm_struct%enuclr_qmmm,   &
                             qmmm_struct%qm_xcrd)

#if KEY_PARALLEL==1
         if(numnod.gt.1) then
            call GCOMB(qm2_struct%hmatrix,qm2_struct%matsize)
            call GCOMB(qm2_struct%qm_qm_2e_repul,qm2_struct%n2el)
            call GCOMB(qmmm_struct%enuclr_qmqm,1)
            call GCOMB(qmmm_struct%enuclr_qmmm,1)
         end if
#endif 

! Compute the SCF calculations
         indx_for_qm = 1
         if(Qdual_check(1).and.Qdual_check(2)) indx_for_qm = 2
         CALL qm2_scf(qm2_struct%fock_matrix, qm2_struct%hmatrix,   &
                      qm2_struct%qm_qm_2e_repul, escf,   &
                      qm2_struct%den_matrix, qm2_struct%old_den_matrix,  &
                      qmmm_struct%enuclr_qmqm+qmmm_struct%enuclr_qmmm,   &
                      qmmm_ewald%scf_mchg,indx_for_qm)

! conversion of unit: It's done in qm2_scf routine
!!!                   escf = escf*EV_TO_KCAL


! The periodic interactions from QM core charges with MM Ewald potential.
         If (qmmm_ewald%QEwald) then
            Call qm_ewald_core(qmmm_struct%nquant,   &
                               qmmm_ewald%ewald_core,   &
                               qm2_params%core_chg,   &
                               qmmm_ewald%scf_mchg,   &
                               qmmm_ewald%empot,qmmm_ewald%eslf)

! Adjust the SCF and Enucle_qmmm energies by the value qm_ewald_core returns.
            qmmm_struct%enuclr_qmmm = qmmm_struct%enuclr_qmmm +   &
                                      qmmm_ewald%ewald_core
            escf                    = escf +   &
                                      qmmm_ewald%ewald_core*EV_TO_KCAL
         End if


! MM correction to peptide linkages: doing this for every node.
!                                    but, only main node in the gradient.
         If (qmmm_nml%peptide_corr) then
            If ( (qmmm_nml%qmtheory .EQ. PM3) .OR.    &
                 (qmmm_nml%qmtheory .EQ. PDDGPM3) .OR.   &
                 (qmmm_nml%qmtheory .EQ. PM3CARB1) ) then
               htype = 7.1853D0
            Else if (qmmm_nml%qmtheory .EQ. AM1) then
               htype = 3.3191D0
            Else !Assume MNDO
               htype = 6.1737D0
            End if

            do I=1,qm2_struct%n_peptide_links
               CALL qm2_dihed(qmmm_struct%qm_coords,   &
                              qm2_struct%peptide_links(1,I),   &
                              qm2_struct%peptide_links(2,I),   &
                              qm2_struct%peptide_links(3,I),   &
                              qm2_struct%peptide_links(4,I),   &
                              ANGLE)
               escf=escf + HTYPE*SIN(ANGLE)**2
            end do
         End if
      end if

! for TR-BOMD: update density for next scf call.
      i_local_cnt=i_local_cnt+1
! debug
!      write(6,*)i_local_cnt, escf, '= escf'
      if(qmmm_nml%tr_bomd .and. q_apply_tr_bomd .and. i_md_step.eq.1) then
         qm2_struct%density_p_old=qm2_struct%den_matrix
         qm2_struct%density_p_new=qm2_struct%den_matrix
      end if

      return
      END SUBROUTINE QMMM_Energy
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Gradient(Natom,Dx,Dy,Dz,X,Y,Z,CG,Eclass,grad_fct, &
                               Qdual_check)
!
! Do the job of computing QM-QM and QM-MM (Coulombic)
! gradients, and passed into CHARMM gradient array.
!

  use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_struct,   &
                              qm2_ghos
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      integer, intent(in)      :: Natom
      real(chm_real),  intent(inout) :: Dx(*), Dy(*), Dz(*)
      real(chm_real),  intent(in)    :: X(*),  Y(*),  Z(*),  CG(*)
      real(chm_real),  intent(inout) :: Eclass
      real(chm_real),  intent(in)    :: grad_fct
      logical, intent(in)      :: Qdual_check(2)
!Local variables
      integer    :: i, m
      real(chm_real)   :: Ecltmp
      integer    :: mstart,mstop
      integer    :: indx_for_qm

! initial setup.
      mstart=1
      mstop=NATOM
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         mstart=1+IPARPT(MYNOD)
         mstop=IPARPT(MYNODP)
      end if
#endif 
!#if KEY_PARALLEL==1 /* (paramain) */
!#if KEY_PARAFULL==1 /* (parfmain) */
!#if KEY_PARASCAL==1 /* (parstest) */
!##ERROR 'Illegal parallel compile options'
!#endif /* (parstest) */
!      mstart=1+IPARPT(MYNOD)
!      mstop=IPARPT(MYNODP)
!#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /* (parfmain) */
!      mstart=1
!      mstop=NATOM
!#else /* (parfmain) */
!##ERROR 'Illegal parallel compile options'
!#endif /* (parfmain) */
!#endif /* (paramain) */

! initialization of gradient array
      qmmm_struct%dxyzqm = zero
      qmmm_struct%dxyzcl = zero

      indx_for_qm = 1
      if(Qdual_check(1).and.Qdual_check(2)) indx_for_qm = 2

! Call routine computing qm-qm gradients
      call qm2_get_qm_forces(indx_for_qm,qmmm_struct%dxyzqm)

! Call routine computing qm-mm (Coulombic) gradients
      call qm2_get_qmmm_forces(qmmm_struct%dxyzqm,   &
                               qmmm_struct%qm_xcrd,   &
                               qmmm_struct%dxyzcl)

! Note for parallel: until now, the gradients are not synchronized.
!                    so, it should be done.
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         call GCOMBR(qmmm_struct%dxyzqm,3*qmmm_struct%nquant_nlink)
         call GCOMBR(qmmm_struct%dxyzcl,3*qmmm_struct%qm_mm_pairs)
      end if
#endif 

! Copyting gradients into CHARMM gradient Dx,Dy,Dz arrays
! 1) QM atom first
      do i=1,qmmm_struct%nquant_nlink
         m = qmmm_nml%iqmatoms(i)
#if KEY_PARALLEL==1
         if(m.ge.mstart .and. m.le.mstop) then
#endif 
         Dx(m) = Dx(m) + qmmm_struct%dxyzqm(1,i)*grad_fct
         Dy(m) = Dy(m) + qmmm_struct%dxyzqm(2,i)*grad_fct
         Dz(m) = Dz(m) + qmmm_struct%dxyzqm(3,i)*grad_fct
#if KEY_PARALLEL==1
         end if
#endif 
      end do

! 2) MM atom 
      do i=1,qmmm_struct%qm_mm_pairs
         m = qmmm_struct%qm_mm_pair_list(i)
#if KEY_PARALLEL==1
         if(m.ge.mstart .and. m.le.mstop) then
#endif 
         Dx(m) = Dx(m) + qmmm_struct%dxyzcl(1,i)*grad_fct
         Dy(m) = Dy(m) + qmmm_struct%dxyzcl(2,i)*grad_fct
         Dz(m) = Dz(m) + qmmm_struct%dxyzcl(3,i)*grad_fct
#if KEY_PARALLEL==1
         end if
#endif 
      end do

! GHO boundary atom method
! Include the gradient correction term for alpha and beta FOCK
! atrices seperately. The repulsion energy between MM atoms
! linked to the GHO boundary is included when the alpha correction
! term is included
      If ( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
         Ecltmp = zero
#if KEY_PARALLEL==1
         if(mynod.eq.0) then          ! only do mynod.eq.0
#endif 
! Treat RHF and UHF differently
         If(qm2_ghos%UHFGHO) then  
!
!! Currently commented out, since it isn't supported yet
!!       Call DQLINK4(Dx,Dy,Dz,X,Y,Z,qm2_struct%fock_matrix,  
!!   *               .false.,Ecltmp,grad_fct)
!!       Eclass = Eclass + Ecltmp
!!       Call DQLINK4(Dx,Dy,Dz,X,Y,Z,qm2_struct%fock_matrixB,  
!!   *                qm2_ghos%UHFGHO,Ecltmp,grad_fct)
!
         Else
            Call DQLINK4(Dx,Dy,Dz,X,Y,Z,CG,qm2_struct%fock_matrix,   &
                         qm2_ghos%UHFGHO,Ecltmp,grad_fct)
            Eclass = Ecltmp
         End if
#if KEY_PARALLEL==1
         else
            Eclass = zero
         end if
         if(numnod.gt.1) call psync()
#endif /* */
      End if

      return
      END SUBROUTINE QMMM_Gradient
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Ewald_Grad(Volume,Recip,X,Y,Z,DX,DY,DZ,grad_fct, &
                                 Virial)
!
! This routine computes gradients from QM/MM-Ewald correction.
! The gradient from reciprocal sum is divided from real space 
! contribution, since the virial needs to be taken cared for
! reciprocal contributions.
!
! The gradient needs to be added after CALL VIRIAL (in energy.src)
!

  use qmmm_module, only : qmmm_nml, qmmm_struct, qm2_rij_eqns,   &
                              qmmm_ewald
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
! Passed in
      real(chm_real),  intent(in)    :: Volume, Recip(6)
      real(chm_real),  intent(inout) :: X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),Virial(9)
      real(chm_real),  intent(in)    :: grad_fct

! Local variables
      integer    :: i,m, nexcl, itotal, itotkq
      integer    :: mstart,mstop
      integer    :: ISTRT_CHECK               ! for external function

! initialization of gradient array
      qmmm_struct%dxyzqm = zero
      qmmm_struct%dxyzcl = zero
      qmmm_ewald%d_ewald_mm = zero
      If(qmmm_ewald%nexl_atm.gt.0) qmmm_ewald%dexl_xyz   = zero

! do some initialization
      mstart = 1
      mstop  = qmmm_struct%qm_mm_pairs
#if KEY_PARALLEL==1
      if(numnod.gt.1)  &
         mstart = ISTRT_CHECK(mstop,qmmm_struct%qm_mm_pairs)
#endif 


! Now compute Gradients from QM/MM-Ewald correction
!1) Real space contribution
!   Since the real space contribution can be added into 
!   Dx,Dy,Dz array directly
!   here, the gradients are added in "dxyzqm" and "dxyzcl", 
!   and copied into main gradient array. 
!   However, the contribution from QM-MM interaction 
!   excluded list is added into dexl_xyz array.
      Call qm_ewald_real_space_gradient(qmmm_ewald%natom,   &
                                        qmmm_struct%nquant_nlink,   &
                                        qmmm_struct%qm_mm_pairs,   &
                                        qmmm_ewald%Erfmod,   &
                                        qmmm_struct%qm_xcrd,   &
                                        qmmm_struct%qm_coords,   &
                                        qmmm_ewald%scf_mchg,   &
                                        qmmm_ewald%scf_mchg_2, &
                                        qm2_rij_eqns%qmmmrijdata,   &
                                        qm2_rij_eqns%qmqmrijdata,  &
                                        qmmm_struct%dxyzqm,   &
                                        qmmm_struct%dxyzcl,   &
                                        qmmm_ewald%kappa,   &
                                        qmmm_nml%qmmmrij_incore,   &
                                        qmmm_nml%qmqmrij_incore,   &
                                        qm2_rij_eqns%n1_qmqm,qm2_rij_eqns%n2_qmqm , &
                                        qm2_rij_eqns%n1_qmmm,qm2_rij_eqns%n2_qmmm) 
      If(qmmm_ewald%nexl_atm.gt.0) then
         call qm_ewald_real_space_gradient_exl(qmmm_ewald%natom,   &
                                        qmmm_struct%nquant_nlink,   &
                                        qmmm_ewald%nexl_atm,   &
                                        qmmm_ewald%nexl_index,   &
                                        qmmm_ewald%Erfmod,   &
                                        qmmm_struct%qm_coords,   &
                                        qmmm_ewald%scf_mchg,  &
                                        qmmm_ewald%scf_mchg_2,  &
                                        qmmm_ewald%exl_xyz,   &
                                        qmmm_struct%dxyzqm,   &
                                        qmmm_ewald%dexl_xyz,   &
                                        qmmm_ewald%kappa) 
      End if


!2) Reciprocal space contribution
!   Due to the need of computing virial contribution from 
!   reciprocal space, the gradients are saved into "d_ewald_mm",
!   and added into Dx,Dy,Dz array later
      if(qmmm_ewald%QNoPMEwald) then
         itotal=qmmm_ewald%iatotl
         itotkq=qmmm_ewald%totkq
      else
         itotal=1
         itotkq=1
      end if
      Call qm_ewald_recip_space_gradient(qmmm_ewald%natom,   &
                                         qmmm_struct%nquant_nlink,   &
                                         qmmm_ewald%iastrt, &
                                         qmmm_ewald%iafinl, &
                                         qmmm_ewald%iatotl, &
                                         itotal,itotkq,      &
                                         qmmm_nml%iqmatoms,   &
                                         qmmm_ewald%totkq,   &
                                         qmmm_ewald%ksqmaxq,   &
                                         qmmm_ewald%kmaxqx,   &
                                         qmmm_ewald%kmaxqy,   &
                                         qmmm_ewald%kmaxqz,   &
                                         qmmm_struct%scaled_mm_charges,  &
                                         qmmm_ewald%scf_mchg, &
                                         qmmm_ewald%scf_mchg_2,   &
                                         qmmm_ewald%Kvec,   &
                                         qmmm_ewald%Ktable,   &
                                         qmmm_ewald%qmktable,   &
                                         qmmm_ewald%structfac_mm, &
                                         qmmm_ewald%kappa, &
                                         recip,Virial,  &
                                         qmmm_ewald%d_ewald_mm, &
                                         qmmm_ewald%QNoPMEwald)

!3) Reciprocal space contribution, for PME version.
      if(.not.qmmm_ewald%QNoPMEwald) then
         Call qm_pme_mm_grad(qmmm_ewald%natom,qmmm_struct%nquant_nlink,   &
                             qmmm_nml%iqmatoms,   &
                             x,y,z,qmmm_ewald%d_ewald_mm,qmmm_struct%scaled_mm_charges,   &
                             qmmm_ewald%scf_mchg_2,recip,volume,   &
                             qmmm_ewald%kappa,virial)
      end if

! for d_ewald_mm/dxyzqm/dxyzcl/dexl_xyz  arrays.
      qmmm_ewald%d_ewald_mm = qmmm_ewald%d_ewald_mm*grad_fct
      qmmm_struct%dxyzqm    = qmmm_struct%dxyzqm*grad_fct
      qmmm_struct%dxyzcl    = qmmm_struct%dxyzcl*grad_fct
      If (qmmm_ewald%nexl_atm .gt. 0)  &
         qmmm_ewald%dexl_xyz=qmmm_ewald%dexl_xyz*grad_fct

! combine all forces and virial here.
#if KEY_PARALLEL==1
      if(numnod.gt.1) then
         call GCOMBR(qmmm_struct%dxyzqm,3*qmmm_struct%nquant_nlink)
         call GCOMBR(qmmm_struct%dxyzcl,3*qmmm_struct%qm_mm_pairs)
         call GCOMBR(qmmm_ewald%d_ewald_mm,3*qmmm_ewald%natom)
         call GCOMB(Virial,9)
         if(mynod.ne.0) Virial=zero
      end if
#endif 


! Copyting gradients into CHARMM gradient Dx,Dy,Dz arrays
! 1) QM atom first
#if KEY_PARALLEL==1
      if(mynod.eq.0) then
#endif 
      do i=1,qmmm_struct%nquant_nlink
         m = qmmm_nml%iqmatoms(i)
         Dx(m) = Dx(m) + qmmm_struct%dxyzqm(1,i) 
         Dy(m) = Dy(m) + qmmm_struct%dxyzqm(2,i) 
         Dz(m) = Dz(m) + qmmm_struct%dxyzqm(3,i) 
      end do
#if KEY_PARALLEL==1
      end if
#endif 

! 2) MM atom
      do i=mstart,mstop                    ! 1,qmmm_struct%qm_mm_pairs
         m = qmmm_struct%qm_mm_pair_list(i)
         Dx(m) = Dx(m) + qmmm_struct%dxyzcl(1,i) 
         Dy(m) = Dy(m) + qmmm_struct%dxyzcl(2,i) 
         Dz(m) = Dz(m) + qmmm_struct%dxyzcl(3,i) 
      end do

! 3) Excluded MM atoms from QM-MM non-bonded interactions 
!    (IGMSEL(i)=5 case)
#if KEY_PARALLEL==1
      if(mynod.eq.0) then
#endif 
      If (qmmm_ewald%nexl_atm .gt. 0) then
         do i=1, qmmm_ewald%nexl_atm
            nexcl = qmmm_ewald%nexl_index(i)
            Dx(nexcl) = Dx(nexcl) + qmmm_ewald%dexl_xyz(1,i) 
            Dy(nexcl) = Dy(nexcl) + qmmm_ewald%dexl_xyz(2,i) 
            Dz(nexcl) = Dz(nexcl) + qmmm_ewald%dexl_xyz(3,i) 
         end do
      End if
#if KEY_PARALLEL==1
      end if
#endif 

      return
      END SUBROUTINE QMMM_EWALD_Grad
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE GETGRDQ(Natom,Dx,Dy,Dz,QMFEP)
!
! Add kspace gradient contribution of QM/MM into gradient array.
!

  use qmmm_module, only : qmmm_ewald,qmmm_ewald_p,qmmm_ewald_r
  use chm_kinds
  use qm2_double
      implicit none
! Passed in
      integer, intent(in)    :: Natom
      real(chm_real),intent(inout) :: Dx(*),Dy(*),Dz(*)
      logical, intent(in)    :: QMFEP

! Local variables
      integer                :: i

!
      if(QMFEP) then
         Do i = qmmm_ewald_r%iastrt,qmmm_ewald_r%iafinl       ! 1, Natom
            Dx(i) = Dx(i) + qmmm_ewald_r%d_ewald_mm(1,i)
            Dy(i) = Dy(i) + qmmm_ewald_r%d_ewald_mm(2,i)
            Dz(i) = Dz(i) + qmmm_ewald_r%d_ewald_mm(3,i)
         End do
         Do i = qmmm_ewald_p%iastrt,qmmm_ewald_p%iafinl       ! 1, Natom
            Dx(i) = Dx(i) + qmmm_ewald_p%d_ewald_mm(1,i)
            Dy(i) = Dy(i) + qmmm_ewald_p%d_ewald_mm(2,i)
            Dz(i) = Dz(i) + qmmm_ewald_p%d_ewald_mm(3,i)
         End do
      else
         Do i = qmmm_ewald%iastrt,qmmm_ewald%iafinl       ! 1, Natom
            Dx(i) = Dx(i) + qmmm_ewald%d_ewald_mm(1,i)
            Dy(i) = Dy(i) + qmmm_ewald%d_ewald_mm(2,i)
            Dz(i) = Dz(i) + qmmm_ewald%d_ewald_mm(3,i)
         End do
      end if

      return
      END SUBROUTINE GETGRDQ
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_Nbnd_Switch(Natom,Ngrp,Natqm,Qminb,Igpbs,Igrpg2,   &
                               Nqmgrp,   &
                               C2OFNB,C2ONNB,XYZG,   &
                               qswitch,qpbound,qfirst,Qcheck, &
                               qdual_check)

  use qmmm_module, only : Get_Type_qmmm_switch, &
                              qmmm_switch,qmmm_switch_r,qmmm_switch_p
  use chm_kinds
  use qm2_double
  use qm2_constants
  use stream
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
! Passed in
      integer,intent(in)    :: Ngrp,natom,natqm,Qminb(*),igrpg2(*),   &
                               igpbs(*),Nqmgrp(*)
      logical,intent(inout) :: qcheck
      real(chm_real), intent(in)  :: C2OFNB,C2ONNB
      real(chm_real), intent(in)  :: XYZG(3,Ngrp)
      logical,intent(in)    :: qpbound,qfirst,qswitch,qdual_check(2)

! local
      integer               :: i,j,iqr,imr,mnum,qnum,ncount,is,iq,js,jq
      real(chm_real)              :: delr(3),drsfm(3),drsfq(3)
      real(chm_real)              :: rul3, rul12, rijl, riju, funct,   &
                               dfn, dfm, dfq
      real(chm_real)              :: scent

      integer               :: ier=0
      real(chm_real), POINTER     :: dxqmmm(:,:) => NULL()

! for printing control
      logical :: qprint

      qprint=.false.
      If(Prnlev.ge.2) qprint=.true.

! for dual quantum region
! copy from ...
      if (qdual_check(1)) then         
         if(qdual_check(2)) then          ! for 2nd qm region.
            call Get_Type_qmmm_switch(qmmm_switch,qmmm_switch_p)
         else                             ! for 1st qm region.
            call Get_Type_qmmm_switch(qmmm_switch,qmmm_switch_r)
         end if
      end if

! initial check
      qmmm_switch%qswitch = qswitch

      If(qmmm_switch%qswitch) then

! memory check first
! first call, allocate memory
        if(qfirst) then
           qmmm_switch%natom  = natom
           qmmm_switch%natqm  = natqm
           qmmm_switch%nqmgrp = Nqmgrp(1)
           qmmm_switch%nmmgrp = ngrp
        end if
        if(qfirst .or. (natom.ne.qmmm_switch%natom)) then
           if(natom.ne.qmmm_switch%natom) then
              qmmm_switch%natom  = natom
              qmmm_switch%natqm  = natqm
              qmmm_switch%nqmgrp = Nqmgrp(1)
              qmmm_switch%nmmgrp = ngrp

! deallocate memory
              deallocate(qmmm_switch%scale, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','scale')

              deallocate(qmmm_switch%scmask, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','scmask')

              deallocate(qmmm_switch%dxqmmm, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','dxqmmm')

              deallocate(qmmm_switch%dxyz_qm, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','dxyz_qm')

              deallocate(qmmm_switch%immgrp, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','immgrp')

              deallocate(qmmm_switch%immatm, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','immatm')

              deallocate(qmmm_switch%iqmgrp, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','iqmgrp')

              deallocate(qmmm_switch%iqmatm, stat=ier)
              if(ier.ne.0) call Aass(0,'QMMM_Nbnd_Switch','iqmatm')
           end if

! allocate memory
           allocate(qmmm_switch%scale(natom), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','scale')

           allocate(qmmm_switch%scmask(natom), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','scmask')

           allocate(qmmm_switch%dxqmmm(6,natom), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','dxqmmm')

           allocate(qmmm_switch%dxyz_qm(3,qmmm_switch%nqmgrp), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','dxyz_qm')


! for mm atom/group
           allocate(qmmm_switch%immgrp(3,ngrp), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','immgrp')

           allocate(qmmm_switch%immatm(natom), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','immatm')

! for qm atom/group
           allocate(qmmm_switch%iqmgrp(qmmm_switch%nqmgrp), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','iqmgrp')

           allocate(qmmm_switch%iqmatm(qmmm_switch%natqm), stat=ier)
           if(ier.ne.0) call Aass(1,'QMMM_Nbnd_Switch','iqmatm')
        end if

! pre-assign qm group and mm group index
        if(qfirst .or. (natom.ne.qmmm_switch%natom)) then
! for qm group
           qmmm_switch%iqmgrp(1:qmmm_switch%nqmgrp) =   &
                                    Nqmgrp(2:qmmm_switch%nqmgrp+1)

! for mm group
           qmmm_switch%immatm(1:natom)    = -1000
           qmmm_switch%immgrp(2:3,1:ngrp) = -1000

           do imr = 1, ngrp
              js  = igpbs(imr)+1
              jq  = igpbs(imr+1)

!  fill immgrp array
              qmmm_switch%immgrp(2,imr) = js
              qmmm_switch%immgrp(3,imr) = jq

!  fill immatm array
              do j = js,jq
                 qmmm_switch%immatm(j) = imr
              end do
           end do

! for qm atom
           do i= 1,qmmm_switch%natqm
              j= iabs(Qminb(i))
              imr = qmmm_switch%immatm(j)
              do iqr=1,qmmm_switch%nqmgrp
                 if(imr.eq.Nqmgrp(iqr+1)) qmmm_switch%iqmatm(i)=iqr
              end do
           end do
        end if

! pre-compute
        rul3 = ONE/ (c2ofnb - c2onnb)**3
        rul12= twelve * rul3

        qmmm_switch%scale(1:natom)      = one
        qmmm_switch%dxqmmm(1:6,1:natom) = zero
        qmmm_switch%scmask(1:natom)     =.false.
        qmmm_switch%immgrp(1,1:ngrp)    = -1000

        ncount = 0
        Do imr=1,ngrp
           If(igrpg2(imr).gt.0) then
              iqr = igrpg2(imr)
              If(iqr.gt.ngrp .or. iqr.le.0) then
                if(qprint) write(6,'(A,1X,I5)')  &
                           'It is not a QM group:', iqr
                qcheck=.false.
                return
              end if
              if(qmmm_switch%immgrp(1,imr).gt.0) then
                 if(qprint) write(6,'(A,1X,I5)')   &
        'MM group included more than two QM groups for switching:',imr
                 qcheck=.false.
                 goto 100              ! return
              end if

              is=igpbs(iqr)+1
              iq=igpbs(iqr+1)
              qnum=iq-is+1

              js=igpbs(imr)+1
              jq=igpbs(imr+1)
              mnum=jq-js+1

              delr(1:3)=XYZG(1:3,iqr)-XYZG(1:3,imr)
!
! in case of using PBOUNDary condition
              If(qpbound) CALL PBCHECK(delr(1),delr(2),delr(3))
!
              scent  =delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)
              If(scent.le.c2onnb .or.scent.ge.c2ofnb) then
                 if(qprint) write(6,'(A,1X,I5)')   &
                 'MM group are not within switching range:', imr
                 qcheck=.false.
                 goto 100              ! return
              End if

              rijl  = c2onnb - scent
              riju  = c2ofnb - scent
              funct = riju*riju*(riju-three*rijl)*rul3
              dfn   = rijl*riju*rul12
              dfm   = dfn /( dble(mnum) * funct )
              dfq   = dfn /( dble(qnum) * funct )

              drsfm(1:3) = delr(1:3)*dfm
              drsfq(1:3) = delr(1:3)*dfq

              do i=js,jq
                 ncount = ncount + 1
                 qmmm_switch%scmask(i)=.true.
                 qmmm_switch%scale(i) = funct
                 qmmm_switch%dxqmmm(1:3,i) = drsfm(1:3)
                 qmmm_switch%dxqmmm(4:6,i) = drsfq(1:3)
              end do

              do i=1,qmmm_switch%nqmgrp
                 if(qmmm_switch%iqmgrp(i).eq.iqr) then
                    qmmm_switch%immgrp(1,imr)=i
                 end if
              end do
           End if
        End do

        qmmm_switch%nswitch_atom = ncount

        If(qmmm_switch%nswitch_atom.le.0 .and. qprint) then
           write(6,*)'QMMM_Nbnd_Switch: no atoms to be switched.',ncount
        End if

! for later memory allocation (see routine qm_fill_mm_coords)
        qmmm_switch%qalloc_mem = .false.
        if(qfirst) then
           qmmm_switch%nswitch_atom_old = qmmm_switch%nswitch_atom+50
           qmmm_switch%qalloc_mem = .true.
        end if
! extra layer
        if (qmmm_switch%nswitch_atom_old .lt.   &
                             qmmm_switch%nswitch_atom) then
           qmmm_switch%nswitch_atom_old = qmmm_switch%nswitch_atom+50
           qmmm_switch%qalloc_mem = .true.
        end if
      Else
        if(qprint) write(6,*)'QMMM_Nbnd_Switch is called at QMSWTCH=', &
                                                 qmmm_switch%qswitch
        qcheck=.false.
        goto 100                   ! return
      End if

100   continue
! for dual quantum region
! copy back into...
      if (qdual_check(1)) then
         if(qdual_check(2)) then          ! for 2nd qm region.
            call Get_Type_qmmm_switch(qmmm_switch_p,qmmm_switch)
         else                             ! for 1st qm region.
            call Get_Type_qmmm_switch(qmmm_switch_r,qmmm_switch)
         end if
      end if

      return
      END SUBROUTINE QMMM_Nbnd_Switch
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE PushPull_all_array(Qdual_check,qget)
!
! Get or put all array for dual quantum region.
!
  use qmmm_module, only: GetQMMMType, PutQMMMType, &
                             qmmm_nml,qmmm_nml_p,qmmm_nml_r, &
                             qmmm_struct,qmmm_struct_p,qmmm_struct_r, &
                             qm2_struct,qm2_struct_p,qm2_struct_r, &
                             qm2_params,qm2_params_p,qm2_params_r, &
                             qm2_rij_eqns,qm2_rij_eqns_p,qm2_rij_eqns_r, &
                             qmmm_switch,qmmm_switch_p,qmmm_switch_r, &
                             qm2_ghos,qm2_ghos_p,qm2_ghos_r, &
                             qmmm_ewald,qmmm_ewald_p,qmmm_ewald_r
  use chm_kinds
      implicit none
!Passed in
      logical, intent(in) :: Qdual_check(2)
      logical, intent(in) :: qget

!Local variables

      if(Qdual_check(1)) then

! copy from ... (all)
         if(qget) then
            call GetQMMMType(Qdual_check(2)) ! get lists.

! copy back into ... (all)
         else
            call PutQMMMType(Qdual_check(2))  ! put lists.
         end if

      end if

      Return
      End SUBROUTINE PushPull_all_array
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE Get_gho_info(NQMAX,NQM_GHO,IQLINK_2,QGHO_SQ)
!
! Fill GHO information to temporay array.
!
  use qmmm_module, only : qm2_ghos
  use chm_kinds
  use qm2_double
  use qm2_constants
      implicit none
!Passed in
      integer  :: NQMAX,NQM_GHO,IQLINK_2(NQMAX)
      logical  :: QGHO_SQ

!Local variables
      integer  :: i

      IQLINK_2 = 0
      QGHO_SQ  = qm2_ghos%QGHO
      NQM_GHO  = qm2_ghos%nqmlnk
      IQLINK_2(1:NQM_GHO) = qm2_ghos%IQLINK(1:NQM_GHO)

      Return
      End SUBROUTINE Get_gho_info
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++           

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE QMMM_GHO_Eclass(Natom,X,Y,Z,CG,Eclass)
!
! Compute Eclass energy from Eclass from GHO atoms.
! This routine is only for Path Integral calculation.
!
  use qmmm_module, only : qm2_struct, qm2_ghos
  use chm_kinds
  use qm2_double
  use qm2_constants
#if KEY_PARALLEL==1
  use parallel  
#endif
      implicit none
!Passed in
      integer, intent(in)    :: Natom
      real(chm_real),intent(in)    :: X(*), Y(*), Z(*), CG(*)
      real(chm_real),intent(inout) :: Eclass

!Local variables
      real(chm_real)   :: Ecltmp

! GHO boundary atom method
! Include the gradient correction term for alpha and beta FOCK
! atrices seperately. The repulsion energy between MM atoms
! linked to the GHO boundary is included when the alpha correction
! term is included
      If ( (qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0) ) then
         Ecltmp = zero
#if KEY_PARALLEL==1
         if(mynod.eq.0) then          ! only do mynod.eq.0
#endif 
! Treat RHF and UHF differently
         If(qm2_ghos%UHFGHO) then
!
!! Currently commented out, since it isn't supported yet
!!       Call DQLINK4_Ener(X,Y,Z,qm2_struct%fock_matrix,
!!   *               .false.,Ecltmp)
!!       Eclass = Eclass + Ecltmp
!!       Call DQLINK4_Ener(X,Y,Z,qm2_struct%fock_matrixB,
!!   *                qm2_ghos%UHFGHO,Ecltmp)
!
         Else
            Call DQLINK4_Ener(X,Y,Z,CG,qm2_struct%fock_matrix, &
                         qm2_ghos%UHFGHO,Ecltmp)
            Eclass = Ecltmp
         End if
#if KEY_PARALLEL==1
         end if
         Eclass = Ecltmp
#endif 
      End if

      return
      END SUBROUTINE QMMM_GHO_Eclass
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#endif /* (mainsquatn)*/
      SUBROUTINE qmmm_initialize_BLANK
!
! dummy routine for compilation
!
      RETURN
      END

