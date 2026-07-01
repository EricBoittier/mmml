module keywords
  implicit none
#include "keywords.inc"
  
contains
  subroutine print_keys()
    use stream, only: outu, prnlev
    implicit none

    integer :: i, endl_countdown
    integer, parameter :: words_per_line = 10

    if(prnlev <= 0) return

    write(outu, '(A, X, I3)') 'KEYWORD LIST: Number of keys:', num_pref_keys
    do i = 1, num_pref_keys
      write(outu, '(A)', advance = 'no') trim(pref_keys(i))
      endl_countdown = modulo(i, words_per_line)
      if (endl_countdown .eq. 0) then
        write(outu, '()')
      else
        write(outu, '(X)', advance = 'no')
      end if
    end do
  end subroutine print_keys
end module keywords
