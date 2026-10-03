! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
module multicom_aux
#if (KEY_STRINGM==1) /*  automatically protect all code */
#if (KEY_MULTICOM==1)
! Basic communicator scheme
!
! MPI_COMM_LOCAL - main communicator ; in general,
! not the same as MPI_COMM_WORLD
! SIZE_LOCAL - number of processors in corresponding communicator
! ME_LOCAL - rank in this communicator
!
! MPI_COMM_GLOBAL - global communicator (involves all nodes)
! SIZE_GLOBAL - number of processors in corresponding communicator
! ME_GLOBAL - rank in communicator
!

  use mpi_f08
!

!



!
!

 TYPE(MPI_Comm), save :: MPI_COMM_LOCAL=MPI_COMM_NULL
 integer*4, save :: ME_LOCAL=MPI_UNDEFINED, SIZE_LOCAL=MPI_UNDEFINED
 TYPE(MPI_Comm), save :: MPI_COMM_GLOBAL=MPI_COMM_NULL
 integer*4, save :: ME_GLOBAL=MPI_UNDEFINED, SIZE_GLOBAL=MPI_UNDEFINED

! Additional communicators

#if (KEY_ENSEMBLE==1)
  TYPE(MPI_Comm), save :: MPI_COMM_ENSBL=MPI_COMM_NULL
  integer*4, save :: ME_ENSBL=MPI_UNDEFINED, SIZE_ENSBL=MPI_UNDEFINED
#endif


#if (KEY_STRINGM==1)
  TYPE(MPI_Comm), save :: MPI_COMM_STRNG=MPI_COMM_NULL
  integer*4, save :: ME_STRNG=MPI_UNDEFINED, SIZE_STRNG=MPI_UNDEFINED
#endif
!
! communicator for reading and parsing input file
  TYPE(MPI_Comm), save :: MPI_COMM_PARSER=MPI_COMM_NULL
  integer*4, save :: ME_PARSER=MPI_UNDEFINED, SIZE_PARSER=MPI_UNDEFINED
!
#endif
!
#endif /* automatically protect all code */
end module multicom_aux
