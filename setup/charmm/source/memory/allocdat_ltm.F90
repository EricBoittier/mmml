      module allocdat
!     definitions
!     filnamar      array of array filenames
!     procnamar     array of array procedure names
!     arnamar       array of array names
!     artypar       array of array types
!     arrankar      array of array ranks
!     arsizear       array of array sizes
!     arkindar      array of kind values
!     qprnallocf    true if printing allocations on the fly
!     qaccumallocdb   true if accumulating database
!     qnoallocdie   true if never dying on alloc errors
!     allocdbarsz   size of allocation database arrays
      character(len=8),allocatable,dimension(:) :: filnamar, &
       arnamar,procnamar,artypar
      integer,allocatable,dimension(:) :: arrankar,arsizear, &
       arkindar
      integer :: nallocentr=0
      integer :: allocdbarsz
      logical :: qprnallocf=.false.,qaccumallocdb=.false., &
       qnoallocdie=.false.
     
! equivalents for failed allocations
!     definitions
!     ffilnamar      array of array filenames
!     fprocnamar     array of array procedure names
!     farnamar       array of array names
!     fartypar       array of array types
!     farrankar      array of array ranks
!     farsizear       array of array sizes
!     farkindar      array of kind values
!     fqprnallocf    true if printing allocations on the fly
!     fqaccumallocdb   true if accumulating database
!     fallocdbarsz   size of allocation database arrays
      character(len=8),allocatable,dimension(:) :: ffilnamar, &
       farnamar,fprocnamar,fartypar
      integer,allocatable,dimension(:) :: farrankar,farsizear, &
       farkindar
      integer :: nfallocentr=0
      integer :: fallocdbarsz
      logical :: qprnfallocf=.false.
!,qaccumallocdb=.false.
      end module allocdat

