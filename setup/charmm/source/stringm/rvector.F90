! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!

!
      module rvector
#if (KEY_STRINGM==1) /*  automatically protect all code */
      use chm_kinds
      implicit none
!
      type real_vector
       real(chm_real), dimension(:), pointer :: r
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
!
      end type real_vector
!
      private real_vector_expand
      integer, parameter, private :: expand_incr=50
!
      contains
       subroutine real_vector_init( v, expand_incr_ )
       type (real_vector) :: v
       integer, optional :: expand_incr_
       integer :: length
       if (v%initialized) return
       if (present(expand_incr_)) then
        length=max(expand_incr_, expand_incr)
       else
        length=expand_incr
       endif
       allocate(v%r(length))
       v%r=0
       v%length=length
       v%last=0
       v%initialized=.true.
       end subroutine real_vector_init
!ccccc
       subroutine real_vector_reinit( v, expand_incr_ )
       type (real_vector) :: v
       integer, optional :: expand_incr_
       if (v%initialized) call real_vector_done(v)
       if (present(expand_incr_)) then ; call real_vector_init(v, expand_incr_) ; else ; call real_vector_init(v) ; endif
       end subroutine real_vector_reinit
!ccccc
       subroutine real_vector_done( v )
       type (real_vector) :: v
       if (associated(v%r)) deallocate(v%r)
       v%length=0
       v%last=0
       v%initialized=.false.
       end subroutine real_vector_done
!ccccc
       subroutine real_vector_expand( v, expand_incr_ )
       type (real_vector) :: v
       integer, optional :: expand_incr_
       integer :: newlength
       real(chm_real), dimension(:), allocatable :: p
!
       if (.not.v%initialized) then
        call real_vector_init(v)
       else
! assume length is valid
        if (present(expand_incr_)) then
         newlength=v%length+max(expand_incr_, expand_incr)
        else
         newlength=v%length+expand_incr
        endif
        allocate(p(newlength)) ! new memory
        p(1:v%length)=v%r ! copy old data
        deallocate(v%r) ! delete old data
        allocate(v%r(newlength))
        v%r = p ! copy data
        deallocate(p)
        v%length=newlength
       endif
       end subroutine real_vector_expand
!ccccc
       subroutine real_vector_set( v, w ) ! set vector v to another vector w
       type (real_vector) :: v, w
       if (w%initialized) then
        call real_vector_reinit(v, w%last) ! ensure that v has enough space
        v%r(1:w%last)=w%r(1:w%last)
        v%last=w%last
       endif
!
       end subroutine real_vector_set
!ccccc
       function real_vector_add( v,i ) ! add a new element to the list (not necessarily unique)
! and return its index
       type (real_vector) :: v
       real(chm_real) :: i
       integer :: j, real_vector_add
!
       if (.not.v%initialized) call real_vector_init(v)
! add element to the list
       if (v%last.eq.v%length) call real_vector_expand(v)
       j=v%last+1
       v%r(j)=i
       v%last=j
       real_vector_add=j
       end function real_vector_add
!ccccc
       function real_vector_uadd( v,i ) ! add a UNIQUE new element to the list and return its index
! if the element already exists, return its index
       type (real_vector) :: v
       real(chm_real) :: i
       integer :: j, real_vector_uadd
!
       if (.not.v%initialized) call real_vector_init(v)
       do j=1,v%last
        if (v%r(j).eq.i) then
         real_vector_uadd=j
         return
        endif
       enddo
! add element to the list
       if (v%last.eq.v%length) call real_vector_expand(v)
       j=v%last+1
       v%r(j)=i
       v%last=j
       real_vector_uadd=j
       end function real_vector_uadd
!ccccc
       function real_vector_get( v,j ) ! returns v%r(j) if j is valid
       type (real_vector) :: v
       real(chm_real) :: i, real_vector_get
       integer :: j
       if (.not.v%initialized) then
        i=-1
       elseif (j.gt.v%last.or.j.le.0) then
        i=-1
       else
        i=v%r(j)
       endif
       real_vector_get=i
       end function real_vector_get
!ccccc
       function real_vector_getlast(v) ! returns the last element, if list nonempty
       type (real_vector) :: v
       real(chm_real) :: i, real_vector_getlast
       if (.not.v%initialized) then
        i=-1
       elseif (v%last.le.0) then
        i=-1
       else
        i=v%r(v%last)
       endif
       real_vector_getlast=i
       end function real_vector_getlast
!ccccc
       function real_vector_getind( v,i ) result(j) ! returns j for the first "v%__DATANAME(j)=i" match
       type (real_vector) :: v
       integer :: i, j
       logical :: found
       found=.false.
       if (v%initialized) then
        do j=1,v%last
         if (v%r(j).eq.i) then ; found=.true. ; exit ; endif
        enddo
       endif
       if (.not.found) j=-1
       end function real_vector_getind
!ccccc
       function real_vector_delete( v,i )
       type (real_vector) :: v
       logical :: real_vector_delete
       integer :: i
       if (i.gt.0.and.i.le.v%last) then ! delete
        if (i.lt.v%last) v%r(i)=v%r(v%last)
        v%last=v%last-1
        real_vector_delete=.true.
       else ! out of bounds
        real_vector_delete=.false.
       endif
       end function real_vector_delete
!ccccc
       function real_vector_eq_ordered(v,w) result(equal)
       type (real_vector) :: v, w
       logical :: equal
       equal = v%initialized.and.w%initialized
       if (equal) equal = v%last .eq. w%last
       if (equal.and.v%last.gt.0) equal=all(v%r(1:v%last).eq.w%r(1:v%last))
       end function real_vector_eq_ordered
!ccccc
       function real_vector_eq_unordered(v,w) result(equal)
       type (real_vector) :: v, w
       logical :: equal
       real(chm_real), dimension(:), pointer :: a, b
       integer :: error
       equal = v%last .eq. w%last
       if (equal.and.v%last.gt.0) then
        allocate(a(v%last), b(v%last))
        a=v%r(1:v%last) ! make copies
        b=w%r(1:v%last)
        call rsort('i', v%last, a, error) ! sort in increasing order
        call rsort('i', v%last, b, error)
        equal=all(a.eq.b)
        deallocate(a,b)
       endif
       end function real_vector_eq_unordered
!ccccc
       subroutine real_vector_sort(v,dir)
       type (real_vector) :: v
       character, optional :: dir
       character :: d
       integer :: error
       if (present(dir)) then ; d=dir ; else ; d = 'i' ; endif ! sort in increasing order by default
       if (v%initialized) then
        if (v%last.gt.1) call rsort(d, v%last, v%r(1:v%last), error)
       endif
       end subroutine real_vector_sort
!cccc
#endif /* automatically protect all code */
      end module rvector
