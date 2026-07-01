module deallocdat
  !     definitions for successful deallocations
  !     dfilnamar      array of array filenames
  !     dprocnamar     array of array procedure names
  !     darnamar       array of array names
  !     dartypar       array of array types
  !     darrankar      array of array ranks
  !     darsizear       array of actual array sizes
  !     diarsizear     array of intended (passed) array sizes
  !     darkindar      array of kind values
  !     dqprndeallocf  true if printing deallocations on the fly
  !     qaccumdeallocdb   true if accumulating deallocations database
  !     qnodealldie    true if never dying on dealloc errors 
  !     dealldbarsz    size of deallocation data arrays
  character(len=8),allocatable,dimension(:) :: dfilnamar, &
       darnamar,dprocnamar,dartypar
  integer,allocatable,dimension(:) :: darrankar,darsizear, &
       diarsizear,darkindar
  integer :: ndeallocentr=0
  integer :: dealldbarsz
  logical :: qprndeallocf=.false.,qaccumdeallocdb=.false., &
       qnodealldie=.false.

  !     definitions for failed deallocations
  !     fdfilnamar      array of array filenames
  !     fdprocnamar     array of array procedure names
  !     fdarnamar       array of array names
  !     fdartypar       array of array types
  !     fdarrankar      array of array ranks
  !     fdarsizear       array of actual array sizes
  !     fdiarsizear     array of intended (passed) array sizes
  !     fdarkindar      array of kind values
  !     fdealldbarsz    size of failed deallocation data arrays
  character(len=8),allocatable,dimension(:) :: fdfilnamar, &
       fdarnamar,fdprocnamar,fdartypar
  integer,allocatable,dimension(:) :: fdarrankar,fdarsizear, &
       fdiarsizear,fdarkindar
  integer :: nfdeallocentr=0
  integer :: fdealldbarsz
  !      logical :: qprndeallocf=.false.,qaccumdeallocdb=.false.
  !     
end module deallocdat

