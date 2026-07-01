! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!



!
      module ivector_list
#if (KEY_STRINGM==1) /*  automatically protect all code */
      use ivector
      use chm_kinds
      implicit none
!
      type int_vlist
       integer, dimension(:), pointer :: i ! integer label
       type (int_vector), dimension(:), pointer :: v ! list of vectors
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
      end type int_vlist
!
      private int_vlist_expand
      integer, parameter, private :: expand_incr=20
!
      contains
       subroutine int_vlist_init( vl, length )
       integer, optional :: length
       integer :: i, llength
       type (int_vlist) :: vl
       if (vl%initialized) return ! do not reinitialize
       if (present(length)) then ; llength=length;
       else ; llength=expand_incr ; endif
       allocate(vl%v(llength))
       allocate(vl%i(llength))
       vl%length=llength
       vl%last=0
! allocate new vectors and point to them
       do i=1,vl%length ; call int_vector_init(vl%v(i)) ; enddo
       vl%i=0
       vl%initialized=.true.
       end subroutine int_vlist_init
!
       subroutine int_vlist_reinit( vl, length )
       integer, optional :: length
       type (int_vlist) :: vl
       if (vl%initialized) call int_vlist_done(vl)
       if (present(length)) then ; call int_vlist_init(vl, length) ; else ; call int_vlist_init(vl) ; endif
       end subroutine int_vlist_reinit
!
       subroutine int_vlist_done( vl )
       integer :: i
       type (int_vlist) :: vl
       if (.not.vl%initialized) return ! do nothing if list not initialized
       if (associated(vl%v)) then
        do i=1,vl%length ; call int_vector_done(vl%v(i)) ; enddo
        deallocate(vl%v)
        vl%length=0
        vl%last=0
       endif
       if (associated(vl%i)) deallocate(vl%i)
       vl%initialized=.false.
       end subroutine int_vlist_done
!
       subroutine int_vlist_expand( vl )
       type (int_vlist) :: vl
       integer :: newlength, i
!
       type (int_vlist) :: wl ! temporary list
!
       if (.not.vl%initialized) then
        call int_vlist_init(vl)
       else
        newlength=vl%length+expand_incr
!
        call int_vlist_init(wl,newlength) ! initialize a new list with larger size
!
        wl%i(1:vl%length)=vl%i ! copy old labels
        wl%v(1:vl%length)=vl%v ! copy old pointers
        deallocate(vl%i) ! delete old labels
        deallocate(vl%v) ! delete old pointers
!
        do i=vl%length+1,newlength;call int_vector_init(wl%v(i));enddo ! allocate space for additional vectors
        wl%last=vl%length
! vl=wl ! copy static data & pointers
        vl%i=>wl%i
        vl%v=>wl%v
        vl%last=wl%last
        vl%length=wl%length
        vl%initialized=wl%initialized
       endif
       end subroutine int_vlist_expand
!
       function int_vlist_add( vl, i, j ) ! add a new list labeled 'i' ; then add element 'j' to it & return index of label i
       type (int_vlist) :: vl
       integer :: i, k, l, int_vlist_add
       integer, optional :: j
!
       if (.not.vl%initialized) call int_vlist_init(vl)
       if (vl%last.eq.vl%length) call int_vlist_expand(vl)
       k=vl%last+1
       vl%i(k)=i ! new list label
       if (present(j)) l=int_vector_add(vl%v(k),j) ! add element to the list
       vl%last=k
       int_vlist_add=k
       end function int_vlist_add
!
       function int_vlist_uadd( vl, i, j ) ! add a new element to the list and return the index of the list
       type (int_vlist) :: vl
       integer :: i, k, l, int_vlist_uadd
       integer, optional :: j
!
       if (.not.vl%initialized) call int_vlist_init(vl)
       do k=1,vl%last
        if (vl%i(k).eq.i) then
         int_vlist_uadd=k
         if (present(j)) l=int_vector_add(vl%v(k),j)
         return
        endif
       enddo
! if we are here, that means a matching entry was not found; we can safely call regular add (not unique)
       if (present(j)) then
        int_vlist_uadd=int_vlist_add( vl, i, j )
       else
        int_vlist_uadd=int_vlist_add( vl, i )
       endif
!
       end function int_vlist_uadd
!
       function int_vlist_uaddu( vl, i, j ) ! add a UNIQUE new element to the list and return the index of the list
       type (int_vlist) :: vl
       integer :: i, k, l, int_vlist_uaddu
       integer, optional :: j
!
       if (.not.vl%initialized) call int_vlist_init(vl)
       do k=1,vl%last
        if (vl%i(k).eq.i) then
         int_vlist_uaddu=k
         if (present(j)) l=int_vector_uadd(vl%v(k),j)
         return
        endif
       enddo
! if we are here, that means a matching entry was not found; we can safely call regular add
       if (present(j)) then
        int_vlist_uaddu=int_vlist_add( vl, i, j )
       else
        int_vlist_uaddu=int_vlist_add( vl, i )
       endif
!
       end function int_vlist_uaddu
!
       function int_vlist_get( vl, i ) result(list) ! if i is found, returns the corresponding list
       type (int_vlist) :: vl
       integer :: i, k
       integer, pointer :: list(:)
       nullify(list)
       do k=1,vl%last
         if (vl%i(k).eq.i) then
          allocate(list(vl%v(k)%last))
          list=vl%v(k)%i(1:vl%v(k)%last) ! note: returning copy of data because otherwise calling sub does not know the list length !
          return
         endif
       enddo
       end function int_vlist_get
!
       function int_vlist_getind( vl,i ) result(j) ! returns j for the first "vl%i(j)=i" match
       type (int_vlist) :: vl
       integer :: i, j
       j=-1
       if (vl%initialized) then
        do j=1,vl%last
         if (vl%i(j).eq.i) exit
        enddo
        if (j.eq.vl%last) then ; if (vl%i(j).ne.i) j=-1 ; endif ! not found entry, just ran to end of loop !
       endif
       end function int_vlist_getind
!
       function int_vlist_delete( vl,i ) ! delete list that corresponds to the tag 'i'
       type (int_vlist) :: vl
       logical :: int_vlist_delete
       integer :: i, k, l
       int_vlist_delete=.false.
       l=vl%last
       do k=1,l
         if (vl%i(k).eq.i) then ! found match at index k
          vl%i(k)=vl%i(l) ! copy label from l into position k
          call int_vector_done(vl%v(k)) ! deallocate vector v(k)
          vl%v(k)=vl%v(l) ! copy pointer to vector v(l) into v(k)
          vl%last=vl%last-1 ! decrement list length
! reinitialize list at position l (the length of the list stays the same)
          nullify(vl%v(l)%i)
          call int_vector_reinit(vl%v(l)) ! need reinit because v(l)%initialized = .true. (despite nullify)
          int_vlist_delete=.true.
          exit
         endif
       enddo
!
       end function int_vlist_delete
!
#endif /* automatically protect all code */
      end module ivector_list
