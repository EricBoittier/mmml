! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! implements a stack of bools that is used to keep track
! of conditionals inside a script in a way that permits conditional execution
! that depends on a state of a node in a parallel computation (interfaced with CHARMM in miscom.src)
      module ifstack
#if (KEY_STRINGM==1) /*  automatically protect all code */
       use ivector
       implicit none
!
       type (int_vector), private, save :: stack
!
       public push_if
       public peek_if
       public pop_if
!
       contains
        subroutine push_if(l)
        logical :: l
        integer :: i
        if (l) then
         i=int_vector_add(stack,1)
        else
         i=int_vector_add(stack,0)
        endif
        end subroutine push_if
!
        function peek_if()
        logical :: l, peek_if
        integer :: i
        i=int_vector_getlast(stack)
        if (i.eq.0) then ! note that if the "if stack" is empty (returns -1), the result is "true" -- meaning that command evaluation proceeds
         l=.false.
        else
         l=.true.
        endif
        peek_if=l
        end function peek_if
!
        function pop_if() ! tells whether the "pop" was successful, but does not give its value (given by peek above)
        logical :: pop_if
        if (stack%initialized.and.stack%last.ge.1) then
         stack%last=stack%last-1
         pop_if=.true.
        else
         pop_if=.false.
        endif
        end function pop_if
!
#endif /* automatically protect all code */
      end module ifstack
