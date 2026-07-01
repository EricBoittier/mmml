! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!



!
      module rvector_list
#if (KEY_STRINGM==1) /*  automatically protect all code */
      use rvector
      use chm_kinds
      implicit none
!
      type real_vlist
       integer, dimension(:), pointer :: i ! integer label
       type (real_vector), dimension(:), pointer :: v ! list of vectors
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
      end type real_vlist
!
      private real_vlist_expand
      integer, parameter, private :: expand_incr=20
!
      contains
       subroutine real_vlist_init( vl, length )
       integer, optional :: length
       integer :: i, llength
       type (real_vlist) :: vl
       if (vl%initialized) return ! do not reinitialize
       if (present(length)) then ; llength=length;
       else ; llength=expand_incr ; endif
       allocate(vl%v(llength))
       allocate(vl%i(llength))
       vl%length=llength
       vl%last=0
! allocate new vectors and point to them
       do i=1,vl%length ; call real_vector_init(vl%v(i)) ; enddo
       vl%i=0
       vl%initialized=.true.
       end subroutine real_vlist_init
!
       subroutine real_vlist_reinit( vl, length )
       integer, optional :: length
       type (real_vlist) :: vl
       if (vl%initialized) call real_vlist_done(vl)
       if (present(length)) then ; call real_vlist_init(vl, length) ; else ; call real_vlist_init(vl) ; endif
       end subroutine real_vlist_reinit
!
       subroutine real_vlist_done( vl )
       integer :: i
       type (real_vlist) :: vl
       if (.not.vl%initialized) return ! do nothing if list not initialized
       if (associated(vl%v)) then
        do i=1,vl%length ; call real_vector_done(vl%v(i)) ; enddo
        deallocate(vl%v)
        vl%length=0
        vl%last=0
       endif
       if (associated(vl%i)) deallocate(vl%i)
       vl%initialized=.false.
       end subroutine real_vlist_done
!
       subroutine real_vlist_expand( vl )
       type (real_vlist) :: vl
       integer :: newlength, i
!
       type (real_vlist) :: wl ! temporary list
!
       if (.not.vl%initialized) then
        call real_vlist_init(vl)
       else
        newlength=vl%length+expand_incr
!
        call real_vlist_init(wl,newlength) ! initialize a new list with larger size
!
        wl%i(1:vl%length)=vl%i ! copy old labels
        wl%v(1:vl%length)=vl%v ! copy old pointers
        deallocate(vl%i) ! delete old labels
        deallocate(vl%v) ! delete old pointers
!
        do i=vl%length+1,newlength;call real_vector_init(wl%v(i));enddo ! allocate space for additional vectors
        wl%last=vl%length
! vl=wl ! copy static data & pointers
        vl%i=>wl%i
        vl%v=>wl%v
        vl%last=wl%last
        vl%length=wl%length
        vl%initialized=wl%initialized
       endif
       end subroutine real_vlist_expand
!
       function real_vlist_add( vl, i, j ) ! add a new list labeled 'i' ; then add element 'j' to it & return index of label i
       type (real_vlist) :: vl
       integer :: i, k, l, real_vlist_add
       real(chm_real), optional :: j
!
       if (.not.vl%initialized) call real_vlist_init(vl)
       if (vl%last.eq.vl%length) call real_vlist_expand(vl)
       k=vl%last+1
       vl%i(k)=i ! new list label
       if (present(j)) l=real_vector_add(vl%v(k),j) ! add element to the list
       vl%last=k
       real_vlist_add=k
       end function real_vlist_add
!
       function real_vlist_uadd( vl, i, j ) ! add a new element to the list and return the index of the list
       type (real_vlist) :: vl
       integer :: i, k, l, real_vlist_uadd
       real(chm_real), optional :: j
!
       if (.not.vl%initialized) call real_vlist_init(vl)
       do k=1,vl%last
        if (vl%i(k).eq.i) then
         real_vlist_uadd=k
         if (present(j)) l=real_vector_add(vl%v(k),j)
         return
        endif
       enddo
! if we are here, that means a matching entry was not found; we can safely call regular add (not unique)
       if (present(j)) then
        real_vlist_uadd=real_vlist_add( vl, i, j )
       else
        real_vlist_uadd=real_vlist_add( vl, i )
       endif
!
       end function real_vlist_uadd
!
       function real_vlist_uaddu( vl, i, j ) ! add a UNIQUE new element to the list and return the index of the list
       type (real_vlist) :: vl
       integer :: i, k, l, real_vlist_uaddu
       real(chm_real), optional :: j
!
       if (.not.vl%initialized) call real_vlist_init(vl)
       do k=1,vl%last
        if (vl%i(k).eq.i) then
         real_vlist_uaddu=k
         if (present(j)) l=real_vector_uadd(vl%v(k),j)
         return
        endif
       enddo
! if we are here, that means a matching entry was not found; we can safely call regular add
       if (present(j)) then
        real_vlist_uaddu=real_vlist_add( vl, i, j )
       else
        real_vlist_uaddu=real_vlist_add( vl, i )
       endif
!
       end function real_vlist_uaddu
!
       function real_vlist_get( vl, i ) result(list) ! if i is found, returns the corresponding list
       type (real_vlist) :: vl
       integer :: i, k
       real(chm_real), pointer :: list(:)
       nullify(list)
       do k=1,vl%last
         if (vl%i(k).eq.i) then
          allocate(list(vl%v(k)%last))
          list=vl%v(k)%r(1:vl%v(k)%last) ! note: returning copy of data because otherwise calling sub does not know the list length !
          return
         endif
       enddo
       end function real_vlist_get
!
       function real_vlist_getind( vl,i ) result(j) ! returns j for the first "vl%i(j)=i" match
       type (real_vlist) :: vl
       integer :: i, j
       j=-1
       if (vl%initialized) then
        do j=1,vl%last
         if (vl%i(j).eq.i) exit
        enddo
        if (j.eq.vl%last) then ; if (vl%i(j).ne.i) j=-1 ; endif ! not found entry, just ran to end of loop !
       endif
       end function real_vlist_getind
!
       function real_vlist_delete( vl,i ) ! delete list that corresponds to the tag 'i'
       type (real_vlist) :: vl
       logical :: real_vlist_delete
       integer :: i, k, l
       real_vlist_delete=.false.
       l=vl%last
       do k=1,l
         if (vl%i(k).eq.i) then ! found match at index k
          vl%i(k)=vl%i(l) ! copy label from l into position k
          call real_vector_done(vl%v(k)) ! deallocate vector v(k)
          vl%v(k)=vl%v(l) ! copy pointer to vector v(l) into v(k)
          vl%last=vl%last-1 ! decrement list length
! reinitialize list at position l (the length of the list stays the same)
          nullify(vl%v(l)%r)
          call real_vector_reinit(vl%v(l)) ! need reinit because v(l)%initialized = .true. (despite nullify)
          real_vlist_delete=.true.
          exit
         endif
       enddo
!
       end function real_vlist_delete
!
#endif /* automatically protect all code */
      end module rvector_list
