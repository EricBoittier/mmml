!
!  The top half of this file is from helPME's helpme.F90 file
!
module helpme
    use iso_c_binding
    implicit none

    ! LatticeType enum
    enum, bind(c)
        enumerator :: Undefined = 0
        enumerator :: ShapeMatrix = 1
        enumerator :: XAligned = 2
    end enum

    ! NodeOrder enum
    enum, bind(c)
        ! These are not scoepd, so Undefined comes from the above
        enumerator :: ZYX = 1
    end enum

#if KEY_LJPME == 1
    public Undefined, ShapeMatrix, XAligned, ZYX

    interface

        function helpme_createD() bind(C, name="helpme_createD")
            use iso_c_binding
            type(c_ptr) :: helpme_createD
        end function

        function helpme_createF() bind(C, name="helpme_createF")
            use iso_c_binding
            type(c_ptr) :: helpme_createF
        end function

        subroutine helpme_destroyD(pme) bind(C, name="helpme_destroyD")
            use iso_c_binding
            type(c_ptr), value :: pme
        end subroutine

        subroutine helpme_destroyF(pme) bind(C, name="helpme_destroyF")
            use iso_c_binding
            type(c_ptr), value :: pme
        end subroutine

        subroutine helpme_setupD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)&
                            bind(C, name="helpme_setupD")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            real(c_double), value :: kappa, scaleFactor
        end subroutine

        subroutine helpme_setupF(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)&
                            bind(C, name="helpme_setupF")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            real(c_float), value :: kappa, scaleFactor
        end subroutine

        subroutine helpme_setup_compressedD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC,&
                                            scaleFactor, nThreads)&
                            bind(C, name="helpme_setup_compressedD")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            real(c_double), value :: kappa, scaleFactor
        end subroutine

        subroutine helpme_setup_compressedF(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC,&
                                            scaleFactor, nThreads)&
                            bind(C, name="helpme_setup_compressedF")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            real(c_float), value :: kappa, scaleFactor
        end subroutine

#if HAVE_MPI == 1
        subroutine helpme_setup_parallelD_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                          nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)&
                            bind(C, name="helpme_setup_parallelD")
            use iso_c_binding
            type(c_ptr), value :: pme, communicator
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC
            real(c_double), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder
        end subroutine

        subroutine helpme_setup_parallelF_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                          nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)&
                            bind(C, name="helpme_setup_parallelF")
            use iso_c_binding
            type(c_ptr), value :: pme, communicator
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC
            real(c_float), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder
        end subroutine

        subroutine helpme_setup_compressed_parallelD_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                          maxKA, maxKB, maxKC, scaleFactor, nThreads,&
                                                          communicator, nodeOrder, numNodesA, numNodesB, numNodesC)&
                            bind(C, name="helpme_setup_compressed_parallelD")
            use iso_c_binding
            type(c_ptr), value :: pme, communicator
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC
            real(c_double), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder
        end subroutine

        subroutine helpme_setup_compressed_parallelF_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                          maxKA, maxKB, maxKC, scaleFactor, nThreads,&
                                                          communicator, nodeOrder, numNodesA, numNodesB, numNodesC)&
                            bind(C, name="helpme_setup_compressed_parallelF")
            use iso_c_binding
            type(c_ptr), value :: pme, communicator
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC
            real(c_float), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder
        end subroutine

        function MPI_Comm_f2c_wrapper(f_handle) bind(C, name="f_MPI_Comm_f2c")
            use iso_c_binding
            integer, value :: f_handle
            type(c_ptr) :: MPI_Comm_f2c_wrapper
        end function

#endif


        subroutine helpme_set_lattice_vectorsD(pme, A, B, C, alpha, beta, gamma, lattice)&
                            bind(C, name="helpme_set_lattice_vectorsD")
            use iso_c_binding
            type(c_ptr), value :: pme
            real(c_double), value :: A, B, C, alpha, beta, gamma
            integer(kind(ShapeMatrix)), value :: lattice
        end subroutine

        subroutine helpme_set_lattice_vectorsF(pme, A, B, C, alpha, beta, gamma, lattice)&
                            bind(C, name="helpme_set_lattice_vectorsF")
            use iso_c_binding
            type(c_ptr), value :: pme
            real(c_float), value :: A, B, C, alpha, beta, gamma
            integer(kind(ShapeMatrix)), value :: lattice
        end subroutine

        function helpme_compute_E_recD(pme, nAtoms, parameterAngMom, parameters, coordinates)&
                            bind(C, name="helpme_compute_E_recD")
            use iso_c_binding
            real(c_double) helpme_compute_E_recD
            type(c_ptr), value :: pme, parameters, coordinates
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_E_recF(pme, nAtoms, parameterAngMom, parameters, coordinates)&
                            bind(C, name="helpme_compute_E_recF")
            use iso_c_binding
            real(c_float) helpme_compute_E_recF
            type(c_ptr), value :: pme, parameters, coordinates
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EF_recD(pme, nAtoms, parameterAngMom, parameters, coordinates, forces)&
                            bind(C, name="helpme_compute_EF_recD")
            use iso_c_binding
            real(c_double) helpme_compute_EF_recD
            type(c_ptr), value :: pme, parameters, coordinates, forces
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EF_recF(pme, nAtoms, parameterAngMom, parameters, coordinates, forces)&
                            bind(C, name="helpme_compute_EF_recF")
            use iso_c_binding
            real(c_float) helpme_compute_EF_recF
            type(c_ptr), value :: pme, parameters, coordinates, forces
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EFV_recD(pme, nAtoms, parameterAngMom, parameters, coordinates, forces, virial)&
                            bind(C, name="helpme_compute_EFV_recD")
            use iso_c_binding
            real(c_double) helpme_compute_EFV_recD
            type(c_ptr), value :: pme, parameters, coordinates, forces, virial
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EFV_recF(pme, nAtoms, parameterAngMom, parameters, coordinates, forces, virial)&
                            bind(C, name="helpme_compute_EFV_recF")
            use iso_c_binding
            real(c_float) helpme_compute_EFV_recF
            type(c_ptr), value :: pme, parameters, coordinates, forces, virial
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        subroutine helpme_compute_P_recD(pme, nAtoms, parameterAngMom, parameters, coordinates, nGridPoints,&
                                         gridPoints, derivativeLevel, potential)&
                            bind(C, name="helpme_compute_P_recD")
            use iso_c_binding
            type(c_ptr), value :: pme, parameters, coordinates, gridPoints, potential
            integer(c_int), value :: parameterAngMom, derivativeLevel
            integer(c_size_t), value :: nAtoms, nGridPoints
        end subroutine

        subroutine helpme_compute_P_recF(pme, nAtoms, parameterAngMom, parameters, coordinates, nGridPoints,&
                                         gridPoints, derivativeLevel, potential)&
                            bind(C, name="helpme_compute_P_recF")
            use iso_c_binding
            type(c_ptr), value :: pme, parameters, coordinates, gridPoints, potential
            integer(c_int), value :: parameterAngMom, derivativeLevel
            integer(c_size_t), value :: nAtoms, nGridPoints
        end subroutine

    end interface

    contains

#if HAVE_MPI == 1

        ! The routines below wrap the call to MPI functionality.  We have to take the Fortran (integer)
        ! representation of the communicator, convert it to a C object pointer and pass the pointer through.

        subroutine helpme_setup_parallelD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                          nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC, communicator
            real(c_double), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder

            type(c_ptr) :: mpiCommunicator

            mpiCommunicator = MPI_Comm_f2c_wrapper(communicator)
            call helpme_setup_parallelD_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                             nThreads, mpiCommunicator, nodeOrder, numNodesA, numNodesB, numNodesC)
        end subroutine

        subroutine helpme_setup_parallelF(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                          nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC, communicator
            real(c_float), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder

            type(c_ptr) :: mpiCommunicator

            mpiCommunicator = MPI_Comm_f2c_wrapper(communicator)
            call helpme_setup_parallelF_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor,&
                                             nThreads, mpiCommunicator, nodeOrder, numNodesA, numNodesB, numNodesC)
        end subroutine

        subroutine helpme_setup_compressed_parallelD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                     maxKA, maxKB, maxKC, scaleFactor,&
                                                     nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC, communicator
            real(c_double), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder

            type(c_ptr) :: mpiCommunicator

            mpiCommunicator = MPI_Comm_f2c_wrapper(communicator)
            call helpme_setup_compressed_parallelD_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                        maxKA, maxKB, maxKC,&
                                                        scaleFactor, nThreads, mpiCommunicator, nodeOrder,&
                                                        numNodesA, numNodesB, numNodesC)
        end subroutine

        subroutine helpme_setup_compressed_parallelF(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                     maxKA, maxKB, maxKC, scaleFactor,&
                                                     nThreads, communicator, nodeOrder, numNodesA, numNodesB, numNodesC)
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
            integer(c_int), value :: numNodesA, numNodesB, numNodesC, communicator
            real(c_float), value :: kappa, scaleFactor
            integer(kind(ZYX)), value :: nodeOrder

            type(c_ptr) :: mpiCommunicator

            mpiCommunicator = MPI_Comm_f2c_wrapper(communicator)
            call helpme_setup_compressed_parallelF_impl(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                                        maxKA, maxKB, maxKC,&
                                                        scaleFactor, nThreads, mpiCommunicator, nodeOrder,&
                                                        numNodesA, numNodesB, numNodesC)
        end subroutine
#endif /* HAVE_MPI */
#endif /* KEY_LJPME */

end module helpme

!
!
! Everything above here should come straight from helPME's helpme.F90 file.
! Everything below is just some glue to make things work with CHARMM.  The
! preprocessor would be a better way to do this, but I ran into problems
! related to include paths not being handles correctly.
!
!



module helpme_wrapper
   ! This simple wrapper just figures out the precision and parallel modes, calling the appropriate
   ! routines from the helPME library.
   use chm_kinds
   use colfft_util, only: ny_box, nz_box
#if KEY_DOMDEC==1
   use domdec_common, only: q_use_single, q_domdec
   use domdec_dr_common,only:q_recip_node
#endif
#if KEY_LJPME == 1
   use helpme, only: helpme_createF, helpme_createD,&
                     helpme_destroyF, helpme_destroyD,&
                     helpme_setupF, helpme_setupD,&
                     helpme_setup_compressedF, helpme_setup_compressedD,&
                     helpme_set_lattice_vectorsF, helpme_set_lattice_vectorsD,&
                     helpme_compute_EFV_recF, helpme_compute_EFV_recD,&
#if HAVE_MPI==1
                     helpme_setup_parallelD, helpme_setup_parallelF,&
                     helpme_setup_compressed_parallelD, helpme_setup_compressed_parallelF,&
#endif
                     ZYX, ShapeMatrix
#endif /* KEY_LJPME */
   use image, only:  xucell
   use iso_c_binding, only: c_ptr, c_size_t, c_null_ptr, c_associated, c_loc
   use memory
#if KEY_PARALLEL==1
   use parallel, only: COMM_CHARMM
#endif

   implicit none

   real(chm_real4), allocatable, target :: tmpCrdF(:,:), tmpFrcF(:,:), paramsF(:)
   real(chm_real), allocatable, target :: tmpCrdD(:,:), tmpFrcD(:,:), paramsD(:)
   real(chm_real4), target :: virialF(6)
   real(chm_real), target :: virialD(6)
   character :: current_type = ' '
   integer :: current_size = 0

   contains

   subroutine dealloc_tmp
       if(allocated(tmpCrdD)) then
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','tmpcrdF', 3, current_size, cr4=tmpCrdF)
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','tmpcrdD', 3, current_size, crl=tmpCrdD)
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','tmpfrcF', 3, current_size, cr4=tmpFrcF)
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','tmpfrcD', 3, current_size, crl=tmpFrcD)
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','paramsF', current_size, cr4=paramsF)
           call chmdealloc('helpme_wrappers.src', 'helpme_wrapper','paramsD', current_size, crl=paramsD)
       endif
   end subroutine dealloc_tmp

   subroutine realloc_tmp(natoms)
       integer, intent(in) :: natoms

       if(.not. allocated(tmpCrdD) .or. natoms > current_size ) then
           call dealloc_tmp
           current_size = 1.2*natoms
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','tmpcrdF', 3, current_size, cr4=tmpCrdF)
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','tmpcrdD', 3, current_size, crl=tmpCrdD)
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','tmpfrcF', 3, current_size, cr4=tmpFrcF)
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','tmpfrcD', 3, current_size, crl=tmpFrcD)
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','paramsF', current_size, cr4=paramsF)
           call chmalloc('helpme_wrappers.src', 'helpme_wrapper','paramsD', current_size, crl=paramsD)
       endif
   end subroutine realloc_tmp

#if KEY_LJPME == 1
   subroutine helpme_setup(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)
       type(c_ptr), intent (in out) :: pme

       integer, intent(in) :: rPower, splineOrder, aDim, bDim, cDim, nThreads
       real(chm_real), intent(in) :: kappa, scaleFactor
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
       if ((.not.q_domdec) .or. (q_domdec .and. q_recip_node)) then
           if(q_use_single()) then
               if ( current_type .eq. 'D' ) call helpme_destroy(pme)
               if (.not. c_associated(pme)) pme = helpme_createF()
               call helpme_setup_parallelF(pme, rPower, REAL(kappa, chm_real4), splineOrder, aDim, bDim, cDim,&
                                           REAL(scaleFactor, chm_real4), nThreads, COMM_CHARMM, ZYX, 1, ny_box, nz_box )
               current_type = 'F'
           else
#endif /* DOMDEC */
               if ( current_type .eq. 'F' ) call helpme_destroy(pme)
               if (.not. c_associated(pme)) pme = helpme_createD()
               call helpme_setup_parallelD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads,&
                                           COMM_CHARMM, ZYX, 1, ny_box, nz_box )
               current_type = 'D'
#if KEY_DOMDEC==1
           endif
       endif
#endif /* DOMDEC */
#else /* PARALLEL */
       if ( current_type .eq. 'F' ) call helpme_destroy(pme)
       if (.not. c_associated(pme)) pme = helpme_createD()
       call helpme_setupD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)
       current_type = 'D'
#endif /* PARALLEL */
   end subroutine helpme_setup


   subroutine helpme_setup_compressed(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                      maxKA, maxKB, maxKC, scaleFactor, nThreads)
       type(c_ptr), intent (in out) :: pme

       integer, intent(in) :: rPower, splineOrder, aDim, bDim, cDim, maxKA, maxKB, maxKC, nThreads
       real(chm_real), intent(in) :: kappa, scaleFactor
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
       if ((.not.q_domdec) .or. (q_domdec .and. q_recip_node)) then
           if(q_use_single()) then
               if ( current_type .eq. 'D' ) call helpme_destroy(pme)
               if (.not. c_associated(pme)) pme = helpme_createF()
               call helpme_setup_compressed_parallelF(pme, rPower, REAL(kappa, chm_real4), splineOrder,&
                                                      aDim, bDim, cDim, maxKA, maxKB, maxKC,&
                                                      REAL(scaleFactor, chm_real4), nThreads,&
                                                      COMM_CHARMM, ZYX, 1, ny_box, nz_box )
               current_type = 'F'
           else
#endif /* DOMDEC */
               if ( current_type .eq. 'F' ) call helpme_destroy(pme)
               if (.not. c_associated(pme)) pme = helpme_createD()
               call helpme_setup_compressed_parallelD(pme, rPower, kappa, splineOrder,&
                                                      aDim, bDim, cDim, maxKA, maxKB, maxKC,&
                                                      scaleFactor, nThreads, COMM_CHARMM, ZYX,&
                                                      1, ny_box, nz_box )
               current_type = 'D'
#if KEY_DOMDEC==1
           endif
       endif
#endif /* DOMDEC */
#else /* PARALLEL */
       if ( current_type .eq. 'F' ) call helpme_destroy(pme)
       if (.not. c_associated(pme)) pme = helpme_createD()
       call helpme_setup_compressedD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim,&
                                     maxKA, maxKB, maxKC, scaleFactor, nThreads)
       current_type = 'D'
#endif
   end subroutine helpme_setup_compressed


   subroutine helpme_compute_EFV_rec(pme, energy, nAtoms, atomList, parameters, X, Y, Z, DX, DY, DZ, virial)
            type(c_ptr), intent(in out) :: pme
            real(chm_real), target, dimension(*), intent(in) :: parameters
            real(chm_real), target, dimension(*), intent(in out) :: virial
            real(chm_real), intent(in out) :: energy
            real(chm_real), intent(in) :: X(*), Y(*), Z(*)
            real(chm_real), intent(in out) :: DX(*), DY(*), DZ(*)
            real(chm_real) :: deltaD(3)
            real(chm_real4) :: deltaF(3)
            integer, intent(in) :: nAtoms, atomList(:)
            integer :: error, atom, fullAtom

       call realloc_tmp(natoms)
#if KEY_DOMDEC==1
       if(q_use_single()) then
           deltaF = 0.5*REAL(xucell(1:3), chm_real4) + 1e-6
           call helpme_set_lattice_vectorsF(pme, REAL(xucell(1), chm_real4), REAL(xucell(2), chm_real4),&
                                                 REAL(xucell(3), chm_real4), REAL(xucell(4), chm_real4),&
                                                 REAL(xucell(5), chm_real4), REAL(xucell(6), chm_real4),&
                                                 ShapeMatrix)
           do atom = 1,nAtoms
               fullAtom = atomList(atom)
               tmpCrdF(1, atom) = REAL(X(fullAtom), chm_real4) - deltaF(1)
               tmpCrdF(2, atom) = REAL(Y(fullAtom), chm_real4) - deltaF(2)
               tmpCrdF(3, atom) = REAL(Z(fullAtom), chm_real4) - deltaF(3)
               paramsF(atom) = REAL(parameters(fullAtom), chm_real4)
           enddo
           virialF = 0d0
           tmpFrcF = 0d0
           energy = energy + helpme_compute_EFV_recF(pme, int(nAtoms, c_size_t), 0, c_loc(paramsF),&
                                                     c_loc(tmpCrdF), c_loc(tmpFrcF), c_loc(virialF))
           do atom = 1,nAtoms
               fullAtom = atomList(atom)
               DX(fullAtom) = DX(fullAtom) - tmpFrcF(1, atom)
               DY(fullAtom) = DY(fullAtom) - tmpFrcF(2, atom)
               DZ(fullAtom) = DZ(fullAtom) - tmpFrcF(3, atom)
           enddo
           virial(1:6)  = virial(1:6) + virialF(1:6)
       else
#endif /* DOMDEC */
           deltaD = 0.5d0*xucell(1:3) + 1d-6
           call helpme_set_lattice_vectorsD(pme, xucell(1), xucell(2), xucell(3),&
                                                 xucell(4), xucell(5), xucell(6), ShapeMatrix)
           do atom = 1, natoms
               fullAtom = atomList(atom)
               tmpCrdD(1,atom) = X(fullAtom) - deltaD(1)
               tmpCrdD(2,atom) = Y(fullAtom) - deltaD(2)
               tmpCrdD(3,atom) = Z(fullAtom) - deltaD(3)
               paramsD(atom) = parameters(fullAtom)
           enddo
           virialD = 0d0
           tmpFrcD = 0d0
           energy = energy + helpme_compute_EFV_recD(pme, int(nAtoms, c_size_t), 0, c_loc(paramsD),&
                                                     c_loc(tmpCrdD), c_loc(tmpFrcD), c_loc(virialD))
           do atom = 1,nAtoms
               fullAtom = atomList(atom)
               DX(fullAtom) = DX(fullAtom) - tmpFrcD(1, atom)
               DY(fullAtom) = DY(fullAtom) - tmpFrcD(2, atom)
               DZ(fullAtom) = DZ(fullAtom) - tmpFrcD(3, atom)
           enddo
           virial(1:6)  = virial(1:6) + virialD(1:6)
#if KEY_DOMDEC==1
       endif
#endif /* DOMDEC */
   end subroutine helpme_compute_EFV_rec


   subroutine helpme_destroy(pme)
       type(c_ptr), intent (in out) :: pme
       if (c_associated(pme)) then
           if(current_type .eq. 'F') then
               call helpme_destroyF(pme)
           else if(current_type .eq. 'D') then
               call helpme_destroyD(pme)
           else
               write(*,*) "Unexpected precision mode used in helpme_wrapper.src"
               stop
           endif
           pme = c_null_ptr
           current_type = ' '
       endif
   end subroutine helpme_destroy
#endif /* KEY_LJMPE */

 end module helpme_wrapper
