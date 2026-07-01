! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
      module multidiag
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
!
      implicit none
      contains
!
      subroutine mdiag(diag,r,n,ndiag)
! V. Ovchinnikov (ovchinnv_at_georgetown_dot_edu), 2010. No Warranty whatsoever.
! multidiagonal matrix solver
! current inplementation uses more space than necessary, but at the benefit of faster memory access
! in the worse case (that of a full matrix), the storage requirement is 2N^2 (vs. N^2)
! No error checking for divide by zero/singularity
! Note: this routine will reduce the matrix into upper diagonal form (i.e. all coefficeints are modified)
! Note also that all diagonal entries corresponding to the same row in matrix have the same row index;
! this means that several entries of the diagonals (except the main diagonal) will not be used;
! e.g.:
! a b|c d e 0 0 0 0
! . a|b c d e 0 0 0
! . .|a b c d e 0 0
! . .|0 a b c d e 0
! in the above, the entries to the left of the boundary (a b & a) diag(-2,1)=diag(-1,1)=diag(-1,2)=0 (and are NOT used; that`s where the storage space
! is wasted)
! Note, finally, that the main diagonal cannot have zero entries (which just reiterates that this is a special purpose solver)
!
      integer :: n, ndiag
! real(chm_real) :: diag(:,:),r(:)
      real(chm_real) :: diag(-ndiag:ndiag,n),r(:)
      integer :: i, j, k, ii
      real(chm_real) :: dummy
!
! (1) transformation to upper diagonal matrix (elimination)
!******************************************************
      do i=1,n-ndiag ! ndiag: maximum number of non-zero diagonals either above or below the main;
                     ! for a tridiagonal matrix, ndiag=1, for a pentadiagonal, ndiag=2
       do j=1,ndiag ! number of rows from which the leading column is eliminated
        ii=i+j
        dummy=diag(-j,ii)/diag(0,i) ! dummy=b(ii)/c(i);
        diag(-j,ii)=0. ! not necessary : for debugging
        do k=-j+1,ndiag-j
          diag(k,ii)=diag(k,ii)-dummy*diag(k+j,i)
        enddo
        r(ii)=r(ii)-r(i)*dummy
       enddo
      enddo
! treat the last ndiag rows differently
      do i=n-ndiag+1,n-1
       do j=1,n-i
        ii=i+j
        dummy=diag(-j,ii)/diag(0,i)
        diag(-j,ii)=0. ! not necessary : for debugging
        do k=-j+1, n-j-i
         diag(k,ii)=diag(k,ii)-dummy*diag(k+j,i)
        enddo
        r(ii)=r(ii)-r(i)*dummy
       enddo
! return
      enddo
! (2) back-substitution
!***************************************************
! special treatment of last ndiag rows
      do i=0,ndiag-1
       ii=n-i
       do j=1,i
        r(ii)=r(ii)-diag(j,ii)*r(ii+j)
       enddo
       r(ii)=r(ii)/diag(0,ii)
      enddo
! other rows
      do ii=n-ndiag,1,-1
       do j=1,ndiag
        r(ii)=r(ii)-diag(j,ii)*r(ii+j)
       enddo
       r(ii)=r(ii)/diag(0,ii)
      enddo
! return
      end subroutine mdiag
!************************************************
      subroutine mdiag2(diag,r,n,ndiag)
! V. Ovchinnikov (ovchinnv_at_georgetown_dot_edu), 2010. No Warranty whatsoever.
! multidiagonal matrix solver; r has many columns, otherwise identical to mdiag (see above)
      integer :: n, ndiag
      real(chm_real) :: diag(-ndiag:ndiag,n),r(:,:)
      integer :: i, j, k, ii
      real(chm_real) :: dummy
!
! (1) transformation to upper diagonal matrix (elimination)
!******************************************************
      do i=1,n-ndiag ! ndiag: maximum number of non-zero diagonals either above or below the main;
                     ! for a tridiagonal matrix, ndiag=1, for a pentadiagonal, ndiag=2
       do j=1,ndiag ! number of rows from which the leading column is eliminated
        ii=i+j
        dummy=diag(-j,ii)/diag(0,i) ! dummy=b(ii)/c(i);
        diag(-j,ii)=0. ! not necessary : for debugging
        do k=-j+1,ndiag-j
          diag(k,ii)=diag(k,ii)-dummy*diag(k+j,i)
        enddo
        r(ii,:)=r(ii,:)-r(i,:)*dummy
       enddo
      enddo
! treat the last ndiag rows differently
      do i=n-ndiag+1,n-1
       do j=1,n-i
        ii=i+j
        dummy=diag(-j,ii)/diag(0,i)
        diag(-j,ii)=0. ! not necessary : for debugging
        do k=-j+1, n-j-i
         diag(k,ii)=diag(k,ii)-dummy*diag(k+j,i)
        enddo
        r(ii,:)=r(ii,:)-r(i,:)*dummy
       enddo
! return
      enddo
! (2) back-substitution
!***************************************************
! special treatment of last ndiag rows
      do i=0,ndiag-1
       ii=n-i
       do j=1,i
        r(ii,:)=r(ii,:)-diag(j,ii)*r(ii+j,:)
       enddo
       r(ii,:)=r(ii,:)/diag(0,ii)
      enddo
! other rows
      do ii=n-ndiag,1,-1
       do j=1,ndiag
        r(ii,:)=r(ii,:)-diag(j,ii)*r(ii+j,:)
       enddo
       r(ii,:)=r(ii,:)/diag(0,ii)
      enddo
! return
      end subroutine mdiag2
!************************************************
      subroutine numdiag(m,n,ndiag)
! given a n x n matrix m, find out how many nonzero diagonals there are, and return in ndiag (notation as in multdiag)
      real(chm_real), intent(in) :: m(:,:)
      integer, intent(out) :: ndiag
      integer, intent(in) :: n
      integer i, j, k
      real(chm_real), parameter :: tol = 1d-15
!
      ndiag=0
      do i=1, n
       do j=1, n
        if (abs(m(i,j)).gt.tol) then
         k=abs(i-j)
         if (k.gt.ndiag) ndiag=k
        endif
       enddo
      enddo
!
      end subroutine numdiag
!*********************************************
      subroutine getdiag(m,diag,n,ndiag)
! given a n x n matrix, extracts the diagonals specified by ndiag, and stores in matrix diag (see notation in multidiag)
      real(chm_real), intent(in) :: m(:,:)
      integer, intent(in) :: ndiag
      integer, intent(in) :: n
! real(chm_real), intent(out) :: diag(:,:)
      real(chm_real), intent(out) :: diag(-ndiag:ndiag,n)
!
      integer i, j, k
!
      do k=-ndiag, ndiag
       do i=1,n ! go over all rows
        j=k+i
        if (j.ge.1.and.j.le.n) diag(k,i)=m(i,j)
       enddo
      enddo
!
      end subroutine getdiag
!********************************************
      subroutine inv_mdiag(m_in,m_out,n,bug)
      use stream
! canned inversion subroutine (for completeness)
      real(chm_real), intent (in) :: m_in(:,:)
      real(chm_real), intent (out) :: m_out(:,:)
      integer :: n, bug
!
      integer :: ndiag, i
      real(chm_real), allocatable :: diag(:,:)
!
      call numdiag(m_in, n, ndiag)
      allocate(diag(-ndiag:ndiag,n))
      call getdiag(m_in, diag, n, ndiag)
      if (any(diag(0,:).eq.0d0)) then ! the main diagonal must not have zero entries
       call wrndie(0,'INV_MDIAG>',trim(' ZERO ON THE MAIN DIAGONAL. ABORT.'))
       bug=-1
       return
      endif
! will invert Ax=b for all b in the Cartesian basis;
! this gives columns in the inverse, which is put back into the rhs
      m_out=0d0; do i=1,n; m_out(i,i)=1d0; enddo
      call mdiag2(diag,m_out,n,ndiag)
      deallocate(diag)
      end subroutine inv_mdiag
!********************************************
!
#endif /* automatically protect all code */
      end module multidiag
