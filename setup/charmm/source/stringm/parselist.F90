! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
      module parselist
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      contains
       subroutine ilist_parse(list, str)
! parses a _unique_ list of ints stored in a character string
        use ivector
        use string
        use stream
!
        implicit none
!
        character(len=*) :: str
        type (int_vector) :: list
! locals
        integer :: i, j, k, strl, inode, jnode, kstep
        integer, parameter :: missing=-99999999
!
        character(len=len("ILIST_PARSE>") ),parameter::whoami="ILIST_PARSE>";!macro
!
        call int_vector_init(list)
!
        inode=missing; jnode=missing;
        strl=len(str)
!
        do
         call trima(str, strl)
         i=indx(str, strl, 'THRU', 4)
         if (i.gt.1.or.i.eq.0) then
! first, add previous node number (no error if invalid)
            if (inode.ne.missing) j=int_vector_uadd(list,inode)
         elseif (i.eq.1) then ! have to deal with THRU
            jnode=gtrmi(str, strl, 'THRU', missing)
! check for 'STEP' keyword
            call trima(str, strl)
            i=indx(str, strl, 'STEP', 4)
            if (i.eq.1) then ! STEP keyword present
             kstep=gtrmi(str, strl, 'STEP', missing)
!! allow (almost) any value of STEP for flexibility
             if (kstep.eq.missing) then
              call wrndie(0,whoami,trim(' INVALID STEP VALUE SPECIFIED. USING STEP=1.'))
              kstep=1
             endif
            else ! STEP not specified -- using 1
             kstep=1
            endif
!
            if (inode.ne.missing.and.jnode.ne.missing) then
             do k=inode, jnode, kstep
               j=int_vector_uadd(list,k)
             enddo
            else
              call wrndie(0,whoami,trim(' INVALID RANGE SPECIFIED. SKIPPING ENTRY.'))
            endif
         endif ! i
! read the next number
         if (strl.lt.1) exit
!
         inode=nexti(str, strl)
        enddo
!
       end subroutine ilist_parse
#endif /* automatically protect all code */
      end module parselist
