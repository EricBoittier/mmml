! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!

!
      module ivector
#if (KEY_STRINGM==1) /*  automatically protect all code */
      use chm_kinds
      implicit none
!
      type int_vector
       integer, dimension(:), pointer :: i
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
!
      end type int_vector
!
      private int_vector_expand
      integer, parameter, private :: expand_incr=50
!
      contains
       subroutine int_vector_init( v, expand_incr_ )
       type (int_vector) :: v
       integer, optional :: expand_incr_
       integer :: length
       if (v%initialized) return
       if (present(expand_incr_)) then
        length=max(expand_incr_, expand_incr)
       else
        length=expand_incr
       endif
       allocate(v%i(length))
       v%i=0
       v%length=length
       v%last=0
       v%initialized=.true.
       end subroutine int_vector_init
!ccccc
       subroutine int_vector_reinit( v, expand_incr_ )
       type (int_vector) :: v
       integer, optional :: expand_incr_
       if (v%initialized) call int_vector_done(v)
       if (present(expand_incr_)) then ; call int_vector_init(v, expand_incr_) ; else ; call int_vector_init(v) ; endif
       end subroutine int_vector_reinit
!ccccc
       subroutine int_vector_done( v )
       type (int_vector) :: v
       if (associated(v%i)) deallocate(v%i)
       v%length=0
       v%last=0
       v%initialized=.false.
       end subroutine int_vector_done
!ccccc
       subroutine int_vector_expand( v, expand_incr_ )
       type (int_vector) :: v
       integer, optional :: expand_incr_
       integer :: newlength
       integer, dimension(:), allocatable :: p
!
       if (.not.v%initialized) then
        call int_vector_init(v)
       else
! assume length is valid
        if (present(expand_incr_)) then
         newlength=v%length+max(expand_incr_, expand_incr)
        else
         newlength=v%length+expand_incr
        endif
        allocate(p(newlength)) ! new memory
        p(1:v%length)=v%i ! copy old data
        deallocate(v%i) ! delete old data
        allocate(v%i(newlength))
        v%i = p ! copy data
        deallocate(p)
        v%length=newlength
       endif
       end subroutine int_vector_expand
!ccccc
       subroutine int_vector_set( v, w ) ! set vector v to another vector w
       type (int_vector) :: v, w
       if (w%initialized) then
        call int_vector_reinit(v, w%last) ! ensure that v has enough space
        v%i(1:w%last)=w%i(1:w%last)
        v%last=w%last
       endif
!
       end subroutine int_vector_set
!ccccc
       function int_vector_add( v,i ) ! add a new element to the list (not necessarily unique)
! and return its index
       type (int_vector) :: v
       integer :: i
       integer :: j, int_vector_add
!
       if (.not.v%initialized) call int_vector_init(v)
! add element to the list
       if (v%last.eq.v%length) call int_vector_expand(v)
       j=v%last+1
       v%i(j)=i
       v%last=j
       int_vector_add=j
       end function int_vector_add
!ccccc
       function int_vector_uadd( v,i ) ! add a UNIQUE new element to the list and return its index
! if the element already exists, return its index
       type (int_vector) :: v
       integer :: i
       integer :: j, int_vector_uadd
!
       if (.not.v%initialized) call int_vector_init(v)
       do j=1,v%last
        if (v%i(j).eq.i) then
         int_vector_uadd=j
         return
        endif
       enddo
! add element to the list
       if (v%last.eq.v%length) call int_vector_expand(v)
       j=v%last+1
       v%i(j)=i
       v%last=j
       int_vector_uadd=j
       end function int_vector_uadd
!ccccc
       function int_vector_get( v,j ) ! returns v%i(j) if j is valid
       type (int_vector) :: v
       integer :: i, int_vector_get
       integer :: j
       if (.not.v%initialized) then
        i=-1
       elseif (j.gt.v%last.or.j.le.0) then
        i=-1
       else
        i=v%i(j)
       endif
       int_vector_get=i
       end function int_vector_get
!ccccc
       function int_vector_getlast(v) ! returns the last element, if list nonempty
       type (int_vector) :: v
       integer :: i, int_vector_getlast
       if (.not.v%initialized) then
        i=-1
       elseif (v%last.le.0) then
        i=-1
       else
        i=v%i(v%last)
       endif
       int_vector_getlast=i
       end function int_vector_getlast
!ccccc
       function int_vector_getind( v,i ) result(j) ! returns j for the first "v%__DATANAME(j)=i" match
       type (int_vector) :: v
       integer :: i, j
       logical :: found
       found=.false.
       if (v%initialized) then
        do j=1,v%last
         if (v%i(j).eq.i) then ; found=.true. ; exit ; endif
        enddo
       endif
       if (.not.found) j=-1
       end function int_vector_getind
!ccccc
       function int_vector_delete( v,i )
       type (int_vector) :: v
       logical :: int_vector_delete
       integer :: i
       if (i.gt.0.and.i.le.v%last) then ! delete
        if (i.lt.v%last) v%i(i)=v%i(v%last)
        v%last=v%last-1
        int_vector_delete=.true.
       else ! out of bounds
        int_vector_delete=.false.
       endif
       end function int_vector_delete
!ccccc
       function int_vector_eq_ordered(v,w) result(equal)
       type (int_vector) :: v, w
       logical :: equal
       equal = v%initialized.and.w%initialized
       if (equal) equal = v%last .eq. w%last
       if (equal.and.v%last.gt.0) equal=all(v%i(1:v%last).eq.w%i(1:v%last))
       end function int_vector_eq_ordered
!ccccc
       function int_vector_eq_unordered(v,w) result(equal)
       type (int_vector) :: v, w
       logical :: equal
       integer, dimension(:), pointer :: a, b
       integer :: error
       equal = v%last .eq. w%last
       if (equal.and.v%last.gt.0) then
        allocate(a(v%last), b(v%last))
        a=v%i(1:v%last) ! make copies
        b=w%i(1:v%last)
        call isort('i', v%last, a, error) ! sort in increasing order
        call isort('i', v%last, b, error)
        equal=all(a.eq.b)
        deallocate(a,b)
       endif
       end function int_vector_eq_unordered
!ccccc
       subroutine int_vector_sort(v,dir)
       type (int_vector) :: v
       character, optional :: dir
       character :: d
       integer :: error
       if (present(dir)) then ; d=dir ; else ; d = 'i' ; endif ! sort in increasing order by default
       if (v%initialized) then
        if (v%last.gt.1) call isort(d, v%last, v%i(1:v%last), error)
       endif
       end subroutine int_vector_sort
!cccc
#endif /* automatically protect all code */
      end module ivector
