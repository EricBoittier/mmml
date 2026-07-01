MODULE reallocation
  !
  !  This module handles explicit memory reallocation and the
  !  resizing of arrays. It is basically a wrapper around chmalloc
  !  and chmdealloc. The structure of the calls is *heavily* based
  !  around that used in MODULE allocation by Robert (Bob) Petrella.
  !
  !  Example (for a 1D array of integers being resized to 20):
  !   call chmrealloc('myfile.src','RESIZE_FOO','IFOO',20,intg=IFOO)
  !
  !  NOTES:
  !  1) If the lower bound of the array to be resized is not 1, the optional
  !     argument oldlb must be used to tell chmrealloc what it is.
  !  2) The lbou optional argument can be used to specify the lower bound of
  !     the reallocated array.
  !  3) Currently only 1D reallocations are implemented, code for 2D & 3D
  !     reallocations is forthcoming.
  !
  !  Thanks to R. Petrella, L. Nilsson, P. Sherwood, R. Venable, B.R. Brooks,
  !  C. Brooks, M. Crowley, Y. Won, and B. Saybasili
  !                     
  !                 - Tim Miller, Dec. 2008
  !
  use chm_kinds
  use allocation
  use deallocation
  implicit none
  
!#ifdef KEY_RESIZE
  character(len=1), parameter, private :: BLANK = ' ' ! from string.F90
!#endif    

  INTERFACE chmrealloc
     module procedure realloc_1d
     module procedure realloc_2d
     module procedure realloc_3d
  END INTERFACE
  
CONTAINS
  
  SUBROUTINE realloc_1d(filename,procname,arrayname,newsz,crl,crlp,mcrlp,cr4,cr8, &
       rlg,intg,intgp,iby,ci2,ci4,ci8,ch1,ch2,ch4,ch6,ch8,ch16,ch20,ch36,ch80, &
       log,qdie,ierr,&
       oldlb,lbou)
    !
    ! This subroutine handles reallocation of 1D arrays. -BTM
    !
    ! Non-array arguments. Essentially the same as chmalloc except newsz
    ! replaces size and we add the oldlb
    use stream
    ! include stream.f90 for PRNLEV
    !--- ##INCLUDE '~/charmm_fcm/stream.f90'
    ! input variables
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: newsz
    integer,optional,intent(in) :: oldlb,lbou
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all of which are optional
    real(kind=chm_real),allocatable,optional,dimension(:) :: crl
    real(kind=chm_real),pointer,optional,dimension(:) :: crlp
    real(kind=chm_real),pointer,optional,dimension(:) :: mcrlp
    real(kind=chm_real4),allocatable,optional,dimension(:) :: cr4
    real(kind=chm_real8),allocatable,optional,dimension(:) :: cr8
    real,allocatable,optional,dimension(:) :: rlg
    integer,allocatable,optional,dimension(:) :: intg
    integer,pointer,optional,dimension(:) :: intgp
    integer(kind=int_byte),allocatable,optional,dimension(:) :: iby
    integer(kind=chm_int2),allocatable,optional,dimension(:) :: ci2
    integer(kind=chm_int4),allocatable,optional,dimension(:) :: ci4
    integer(kind=chm_int8),allocatable,optional,dimension(:) :: ci8
    character(len=1),allocatable,optional,dimension(:) :: ch1
    character(len=2),allocatable,optional,dimension(:) :: ch2
    character(len=4),allocatable,optional,dimension(:) :: ch4
    character(len=6),allocatable,optional,dimension(:) :: ch6
    character(len=8),allocatable,optional,dimension(:) :: ch8
    character(len=16),allocatable,optional,dimension(:) :: ch16
    character(len=20),allocatable,optional,dimension(:) :: ch20
    character(len=36),allocatable,optional,dimension(:) :: ch36
    character(len=80),allocatable,optional,dimension(:) :: ch80
    logical,allocatable,optional,dimension(:) :: log
    ! local variables
    integer     :: oldsz,locoldlb,locoldub,locerr,newub,newlb,i
    logical     :: qlocdie,qsamelb
    real(kind=chm_real),allocatable,dimension(:) :: tcrl
    real(kind=chm_real),pointer,dimension(:) :: tcrlp
    real(kind=chm_real4),allocatable,dimension(:) :: tcr4 
    real(kind=chm_real8),allocatable,dimension(:) :: tcr8 
    real,allocatable,dimension(:) :: trlg
    integer,allocatable,dimension(:) :: tintg
    integer,pointer,dimension(:) :: tintgp
    integer(kind=int_byte),allocatable,dimension(:) :: tiby     
    integer(kind=chm_int2),allocatable,dimension(:) :: tci2 
    integer(kind=chm_int4),allocatable,dimension(:) :: tci4
    integer(kind=chm_int8),allocatable,dimension(:) :: tci8     
    character(len=1),allocatable,dimension(:) :: tch1 
    character(len=2),allocatable,dimension(:) :: tch2 
    character(len=4),allocatable,dimension(:) :: tch4 
    character(len=6),allocatable,dimension(:) :: tch6 
    character(len=8),allocatable,dimension(:) :: tch8 
    character(len=16),allocatable,dimension(:) :: tch16
    character(len=20),allocatable,dimension(:) :: tch20
    character(len=36),allocatable,dimension(:) :: tch36
    character(len=80),allocatable,dimension(:) :: tch80
    logical,allocatable,dimension(:) :: tlog

    if(present(lbou)) then
       newlb = lbou
    else
       newlb = 1
    endif
    newub = newlb + newsz - 1
    if(present(oldlb)) then
       locoldlb = oldlb
    else
       locoldlb = 1
    endif
    if(present(qdie)) then
       qlocdie = qdie
    else
       qlocdie = .true.
    endif
    locerr = 0
    qsamelb = locoldlb.eq.newlb 

    ! like chmalloc, we have to go through the various optional args, in order
    ! of their frequency within CHARMM.
    ! chm_real
    if(present(crl)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(crl)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tcrl',newsz,crl=tcrl, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tcrl=0.0
!#endif
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tcrl(i+newlb-1) = crl(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tcrl(i+newlb-1) = crl(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,crl=crl,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcrl',newsz,crl=tcrl,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,crl=crl,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcrl',newsz,crl=tcrl, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          crl(i) = tcrl(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tcrl',newsz,crl=tcrl, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

    else if(present(crlp)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(crlp)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tcrlp',newsz,crlp=tcrlp, &
            lbou=newlb,ierr=locerr,&
            qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tcrlp(i+newlb-1) = crlp(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tcrlp(i+newlb-1) = crlp(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array
       call chmdealloc(filename,procname,arrayname,oldsz,crlp=crlp,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcrlp',newsz,crlp=tcrlp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: pointer assignment
       crlp => tcrlp

       ! general integer
    else if(present(intg)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(intg)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tintg',newsz,intg=tintg, &
            lbou=newlb,ierr=locerr,&
            qdie=qlocdie)
!#ifdef KEY_RESIZE
       tintg=0
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tintg(i+newlb-1) = intg(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tintg(i+newlb-1) = intg(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,intg=intg,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tintg',newsz,intg=tintg,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,intg=intg,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tintg',newsz,intg=tintg, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          intg(i) = tintg(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tintg',newsz,intg=tintg, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

    else if(present(intgp)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(intgp)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tintgp',newsz,intgp=tintgp, &
            lbou=newlb,ierr=locerr,&
            qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tintgp(i+newlb-1) = intgp(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tintgp(i+newlb-1) = intgp(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,intgp=intgp,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tintgp',newsz,intgp=tintgp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: pointer assignment
       intgp => tintgp

       ! general real
    else if(present(rlg)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(rlg)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','trlg',newsz,rlg=trlg,lbou=newlb, &
            ierr=locerr,&
            qdie=qlocdie)
       trlg=0 ! hwm2206
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             trlg(i+newlb-1) = rlg(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             trlg(i+newlb-1) = rlg(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,rlg=rlg,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','trlg',newsz,rlg=trlg,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,rlg=rlg,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','trlg',newsz,rlg=trlg, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          rlg(i) = trlg(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','trlg',newsz,rlg=trlg, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=6)
    else if(present(ch6)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch6)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch6',newsz,ch6=tch6, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch6=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch6(i+newlb-1) = ch6(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch6(i+newlb-1) = ch6(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch6=ch6,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch6',newsz,ch6=tch6,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch6=ch6,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch6',newsz,ch6=tch6, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch6(i) = tch6(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch6',newsz,ch6=tch6, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       !character(len=8)
    else if(present(ch8)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch8)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch8',newsz,ch8=tch8, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch8=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch8(i+newlb-1) = ch8(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch8(i+newlb-1) = ch8(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch8=ch8,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch8',newsz,ch8=tch8,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch8=ch8,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch8',newsz,ch8=tch8, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch8(i) = tch8(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch8',newsz,ch8=tch8, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! chm_real4
    else if(present(cr4)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(cr4)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tcr4',newsz,cr4=tcr4,lbou=newlb, &
            ierr=locerr, qdie=qlocdie)
!#ifdef KEY_RESIZE
       tcr4=0.
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tcr4(i+newlb-1) = cr4(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tcr4(i+newlb-1) = cr4(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,cr4=cr4,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcr4',newsz,cr4=tcr4,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,cr4=cr4,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcr4',newsz,cr4=tcr4, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          cr4(i) = tcr4(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tcr4',newsz,cr4=tcr4, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! chm_int8
    else if(present(ci8)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ci8)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tci8',newsz,ci8=tci8,lbou=newlb, &
            ierr=locerr,&
            qdie=qlocdie)
       tci8=0 ! hwm2206
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tci8(i+newlb-1) = ci8(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tci8(i+newlb-1) = ci8(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ci8=ci8,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci8',newsz,ci8=tci8,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ci8=ci8,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci8',newsz,ci8=tci8, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ci8(i) = tci8(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tci8',newsz,ci8=tci8, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! logical
    else if(present(log)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(log)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tlog',newsz,log=tlog,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tlog=.false. ! false by default
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tlog(i+newlb-1) = log(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tlog(i+newlb-1) = log(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,log=log,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tlog',newsz,log=tlog,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,log=log,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tlog',newsz,log=tlog, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          log(i) = tlog(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tlog',newsz,log=tlog, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! less common types
       ! chm_real8
    else if(present(cr8)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(cr8)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tcr8',newsz,cr8=tcr8,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tcr8=0.0
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tcr8(i+newlb-1) = cr8(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tcr8(i+newlb-1) = cr8(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,cr8=cr8, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcr8',newsz,cr8=tcr8,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,cr8=cr8,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcr8',newsz,cr8=tcr8, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          cr8(i) = tcr8(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tcr8',newsz,cr8=tcr8, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! chm_int4
    else if(present(ci4)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ci4)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tci4',newsz,ci4=tci4,lbou=newlb, &
            ierr=locerr,&
            qdie=qlocdie)
       tci4=0 ! hwm2206
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tci4(i+newlb-1) = ci4(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tci4(i+newlb-1) = ci4(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ci4=ci4,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci4',newsz,ci4=tci4,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ci4=ci4,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci4',newsz,ci4=tci4, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ci4(i) = tci4(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tci4',newsz,ci4=tci4, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! int_byte
    else if(present(iby)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(iby)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tiby',newsz,iby=tiby,lbou=newlb, &
            ierr=locerr, qdie=qlocdie)
       tiby=0 ! hwm2206
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tiby(i+newlb-1) = iby(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tiby(i+newlb-1) = iby(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,iby=iby,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tiby',newsz,iby=tiby,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,iby=iby,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tiby',newsz,iby=tiby, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          iby(i) = tiby(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tiby',newsz,iby=tiby, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! chm_int2
    else if(present(ci2)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ci2)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tci2',newsz,ci2=tci2,lbou=newlb, &
            ierr=locerr,  qdie=qlocdie)
       tci2=0 ! hwm2206
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tci2(i+newlb-1) = ci2(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tci2(i+newlb-1) = ci2(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ci2=ci2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci2',newsz,ci2=tci2,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ci2=ci2,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tci2',newsz,ci2=tci2, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ci2(i) = tci2(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tci2',newsz,ci2=tci2, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=1)
    else if(present(ch1)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch1)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch1',newsz,ch1=tch1,lbou=newlb, &
            ierr=locerr, qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch1=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch1(i+newlb-1) = ch1(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch1(i+newlb-1) = ch1(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch1=ch1,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch1',newsz,ch1=tch1,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch1=ch1,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch1',newsz,ch1=tch1, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch1(i) = tch1(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch1',newsz,ch1=tch1, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=2)
    else if(present(ch2)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch2)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch2',newsz,ch2=tch2,lbou=newlb, &
            ierr=locerr,&
            qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch2=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch2(i+newlb-1) = ch2(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch2(i+newlb-1) = ch2(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch2=ch2, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch2',newsz,ch2=tch2,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch2=ch2,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch2',newsz,ch2=tch2, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch2(i) = tch2(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch2',newsz,ch2=tch2, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif


       ! character(len=4)
    else if(present(ch4)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch4)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch4',newsz,ch4=tch4,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch4=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch4(i+newlb-1) = ch4(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch4(i+newlb-1) = ch4(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch4=ch4, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch4',newsz,ch4=tch4,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch4=ch4,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch4',newsz,ch4=tch4, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch4(i) = tch4(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch4',newsz,ch4=tch4, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif



       ! character(len=16)
    else if(present(ch16)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch16)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch16',newsz,ch16=tch16, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch16=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch16(i+newlb-1) = ch16(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch16(i+newlb-1) = ch16(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch16=ch16, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch16',newsz,ch16=tch16,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch16=ch16,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch16',newsz,ch16=tch16, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch16(i) = tch16(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch16',newsz,ch16=tch16, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=20)
    else if(present(ch20)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch20)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch20',newsz,ch20=tch20, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch20=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch20(i+newlb-1) = ch20(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch20(i+newlb-1) = ch20(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch20=ch20, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch20',newsz,ch20=tch20,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch20=ch20,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch20',newsz,ch20=tch20, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch20(i) = tch20(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch20',newsz,ch20=tch20, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=36)
    else if(present(ch36)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch36)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch36',newsz,ch36=tch36, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch36=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch36(i+newlb-1) = ch36(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch36(i+newlb-1) = ch36(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch36=ch36, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch36',newsz,ch36=tch36,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch36=ch36,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch36',newsz,ch36=tch36, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch36(i) = tch36(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch36',newsz,ch36=tch36, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

       ! character(len=80)
    else if(present(ch80)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(ch80)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tch80',newsz,ch80=tch80, &
            lbou=newlb,ierr=locerr,qdie=qlocdie)
!#ifdef KEY_RESIZE
       tch80=BLANK
!#endif       
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tch80(i+newlb-1) = ch80(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tch80(i+newlb-1) = ch80(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,ch80=ch80, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch80',newsz,ch80=tch80,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,ch80=ch80,lbou=newlb, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tch80',newsz,ch80=tch80, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array
       do i = newlb,newub
          ch80(i) = tch80(i)
       enddo
       call chmdealloc('realloc_mod.src','realloc_1d','tch80',newsz,ch80=tch80, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

    else if(present(mcrlp)) then

       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(mcrlp)
       locoldub = locoldlb + oldsz - 1
       if(qsamelb.and.(newsz.eq.oldsz)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_1d','tcrlp',newsz,mcrlp=tcrlp, &
            lbou=newlb,ierr=locerr,&
            qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif
       if(newsz.ge.oldsz) then
          do i = 1,oldsz
             tcrlp(i+newlb-1) = mcrlp(i+locoldlb-1)
          enddo
       else
          do i = 1,newsz
             tcrlp(i+newlb-1) = mcrlp(i+locoldlb-1)
          enddo
       endif

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,mcrlp=mcrlp,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_1d','tcrlp',newsz,mcrlp=tcrlp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: pointer assignment
       mcrlp => tcrlp

    else
       write(6,*) 'No array matched list of available types.'
    endif

  END SUBROUTINE realloc_1d

  SUBROUTINE realloc_2d(filename,procname,arrayname,newsz,newsz2,crl,crlp,mcrlp,intg,intgp,&
       qdie,ierr,oldlb,oldlb2,lbou,lbou2)
    !
    ! This subroutine handles reallocation of 2D arrays. -APH
    !
    ! Non-array arguments. Essentially the same as chmalloc except newsz
    ! replaces size and we add the oldlb
    use stream
    ! include stream.f90 for PRNLEV
    !--- ##INCLUDE '~/charmm_fcm/stream.f90'
    ! input variables
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: newsz, newsz2
    integer,optional,intent(in) :: oldlb,oldlb2,lbou,lbou2
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all of which are optional
    real(kind=chm_real),allocatable,optional,dimension(:,:) :: crl
    real(kind=chm_real),pointer,optional,dimension(:,:) :: crlp
    real(kind=chm_real),pointer,optional,dimension(:,:) :: mcrlp
    integer,allocatable,optional,dimension(:,:) :: intg
    integer,pointer,optional,dimension(:,:) :: intgp
    ! local variables
    integer     :: oldsz,oldsz2,locoldlb,locoldlb2,locoldub,locoldub2,locerr
    integer     :: newub,newub2,newlb,newlb2,i,sz,sz2
    logical     :: qlocdie,qsamelb,qsamelb2
    real(kind=chm_real),allocatable,dimension(:,:) :: tcrl
    real(kind=chm_real),pointer,dimension(:,:) :: tcrlp
    integer,allocatable,dimension(:,:) :: tintg
    integer,pointer,dimension(:,:) :: tintgp

    if(present(lbou)) then
       newlb = lbou
    else
       newlb = 1
    endif
    newub = newlb + newsz - 1
    if(present(lbou2)) then
       newlb2 = lbou2
    else
       newlb2 = 1
    endif
    newub2 = newlb2 + newsz2 - 1

    if(present(oldlb)) then
       locoldlb = oldlb
    else
       locoldlb = 1
    endif
    if(present(oldlb2)) then
       locoldlb2 = oldlb2
    else
       locoldlb2 = 1
    endif

    if(present(qdie)) then
       qlocdie = qdie
    else
       qlocdie = .true.
    endif
    locerr = 0
    qsamelb = locoldlb.eq.newlb
    qsamelb2 = locoldlb2.eq.newlb2

    ! like chmalloc, we have to go through the various optional args, in order
    ! of their frequency within CHARMM.
    ! chm_real
    if(present(crl)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(crl,1)
       oldsz2 = size(crl,2)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       if(qsamelb.and.qsamelb2.and.(newsz.eq.oldsz).and.(newsz2.eq.oldsz2)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crl=tcrl, &
            lbou=newlb,lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       tcrl(1+newlb-1:sz+newlb-1,1+newlb2-1:sz2+newlb2-1) = &
            crl(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1)

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,crl=crl,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crl=tcrl,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,newsz2,crl=crl,lbou=newlb,lbou2=newlb2, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crl=tcrl, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array

       crl(newlb:newub,newlb2:newub2) = tcrl(newlb:newub,newlb2:newub2)
       call chmdealloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crl=tcrl, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif
    elseif(present(intg)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(intg,1)
       oldsz2 = size(intg,2)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       if(qsamelb.and.qsamelb2.and.(newsz.eq.oldsz).and.(newsz2.eq.oldsz2)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intg=tintg, &
            lbou=newlb,lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       tintg(1+newlb-1:sz+newlb-1,1+newlb2-1:sz2+newlb2-1) = &
            intg(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1)

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,intg=intg,ierr=locerr,&
            qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intg=tintg,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,newsz2,intg=intg,lbou=newlb,&
            lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intg=tintg, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array

       intg(newlb:newub,newlb2:newub2) = tintg(newlb:newub,newlb2:newub2)
       call chmdealloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intg=tintg, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif
    elseif(present(intgp)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(intgp,1)
       oldsz2 = size(intgp,2)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       if(qsamelb.and.qsamelb2.and.(newsz.eq.oldsz).and.(newsz2.eq.oldsz2)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_2d','tintgp',newsz,newsz2,intgp=tintgp, &
            lbou=newlb,lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       tintgp(1+newlb-1:sz+newlb-1,1+newlb2-1:sz2+newlb2-1) = &
            intgp(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1)

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,intgp=intgp,ierr=locerr,&
            qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tintgp',newsz,newsz2,intgp=tintgp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,newsz2,intgp=intgp,lbou=newlb,&
            lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intgp=tintgp, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array

       intg(newlb:newub,newlb2:newub2) = tintg(newlb:newub,newlb2:newub2)
       call chmdealloc('realloc_mod.src','realloc_2d','tintg',newsz,newsz2,intgp=tintgp, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif
    else if(present(crlp)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(crlp,1)
       oldsz2 = size(crlp,2)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       if(qsamelb.and.qsamelb2.and.(newsz.eq.oldsz).and.(newsz2.eq.oldsz2)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crlp=tcrlp, &
            lbou=newlb,lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       tcrlp(newlb:sz+newlb-1,newlb2:sz2+newlb2-1) = &
            crlp(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1)

       ! step 3: (a) deallocate the original array
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,crlp=crlp,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,crlp=tcrlp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       
       ! step 4: pointer assignment
       crlp => tcrlp

    else if(present(mcrlp)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(mcrlp,1)
       oldsz2 = size(mcrlp,2)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       if(qsamelb.and.qsamelb2.and.(newsz.eq.oldsz).and.(newsz2.eq.oldsz2)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,mcrlp=tcrlp, &
            lbou=newlb,lbou2=newlb2,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       tcrlp(newlb:sz+newlb-1,newlb2:sz2+newlb2-1) = &
            mcrlp(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1)

       ! step 3: (a) deallocate the original array
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,mcrlp=mcrlp,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_2d','tcrl',newsz,newsz2,mcrlp=tcrlp,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       
       ! step 4: pointer assignment
       mcrlp => tcrlp

    else
       write(6,*) 'No array matched list of available types.'
    endif

  END SUBROUTINE realloc_2d

  SUBROUTINE realloc_3d(filename,procname,arrayname,newsz,newsz2,newsz3,&
       crl,intg,qdie,ierr,oldlb,oldlb2,oldlb3,lbou,lbou2,lbou3)
    !
    ! This subroutine handles reallocation of 3D arrays. -APH
    !
    ! Non-array arguments. Essentially the same as chmalloc except newsz
    ! replaces size and we add the oldlb
    use stream
    ! include stream.f90 for PRNLEV
    !--- ##INCLUDE '~/charmm_fcm/stream.f90'
    ! input variables
    character(len=*),intent(in) :: filename,procname,arrayname
    integer,intent(in) :: newsz, newsz2, newsz3
    integer,optional,intent(in) :: oldlb,oldlb2,oldlb3,lbou,lbou2,lbou3
    logical,optional,intent(in) :: qdie
    integer,optional,intent(out) :: ierr
    ! passed arrays, all of which are optional
    real(kind=chm_real),allocatable,optional,dimension(:,:,:) :: crl
    integer,allocatable,optional,dimension(:,:,:) :: intg
    ! local variables
    integer     :: oldsz,oldsz2,oldsz3,locoldlb,locoldlb2,locoldlb3
    integer     :: locoldub,locoldub2,locoldub3,locerr
    integer     :: newub,newub2,newub3,newlb,newlb2,newlb3,i,sz,sz2,sz3
    logical     :: qlocdie,qsamelb,qsamelb2,qsamelb3
    real(kind=chm_real),allocatable,dimension(:,:,:) :: tcrl
    integer,allocatable,dimension(:,:,:) :: tintg

    if(present(lbou)) then
       newlb = lbou
    else
       newlb = 1
    endif
    newub = newlb + newsz - 1
    if(present(lbou2)) then
       newlb2 = lbou2
    else
       newlb2 = 1
    endif
    newub2 = newlb2 + newsz2 - 1
    if(present(lbou3)) then
       newlb3 = lbou3
    else
       newlb3 = 1
    endif
    newub3 = newlb3 + newsz3 - 1

    if(present(oldlb)) then
       locoldlb = oldlb
    else
       locoldlb = 1
    endif
    if(present(oldlb2)) then
       locoldlb2 = oldlb2
    else
       locoldlb2 = 1
    endif
    if(present(oldlb3)) then
       locoldlb3 = oldlb3
    else
       locoldlb3 = 1
    endif

    if(present(qdie)) then
       qlocdie = qdie
    else
       qlocdie = .true.
    endif
    locerr = 0
    qsamelb = locoldlb.eq.newlb
    qsamelb2 = locoldlb2.eq.newlb2
    qsamelb3 = locoldlb3.eq.newlb3

    ! like chmalloc, we have to go through the various optional args, in order
    ! of their frequency within CHARMM.
    ! chm_real
    if(present(crl)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(crl,1)
       oldsz2 = size(crl,2)
       oldsz3 = size(crl,3)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       locoldub3 = locoldlb3 + oldsz3 - 1
       if(qsamelb.and.qsamelb2.and.qsamelb3.and.(newsz.eq.oldsz).and.&
            (newsz2.eq.oldsz2).and.(newsz3.eq.oldsz3)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_3d','tcrl',newsz,newsz2,newsz3,crl=tcrl, &
            lbou=newlb,lbou2=newlb2,lbou3=newlb3,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       sz3 = min(newsz3,oldsz3)
       tcrl(1+newlb-1:sz+newlb-1,1+newlb2-1:sz2+newlb2-1,1+newlb3-1:sz3+newlb3-1) = &
            crl(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1,1+locoldlb3-1:sz3+locoldlb3-1)

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,oldsz3,crl=crl,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_3d','tcrl',newsz,newsz2,newsz3,crl=tcrl,&
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,newsz2,newsz3,crl=crl,&
            lbou=newlb,lbou2=newlb2,lbou3=newlb3,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_3d','tcrl',newsz,newsz2,newsz3,crl=tcrl, &
               ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array

       crl(newlb:newub,newlb2:newub2,newlb3:newub3) = tcrl(newlb:newub,newlb2:newub2,newlb3:newub3)
       call chmdealloc('realloc_mod.src','realloc_3d','tcrl',newsz,newsz2,newsz3,crl=tcrl, &
            ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif
    elseif(present(intg)) then
       ! step 1: get sizes and make sure we aren't performing a no-op
       oldsz = size(intg,1)
       oldsz2 = size(intg,2)
       oldsz3 = size(intg,3)
       locoldub = locoldlb + oldsz - 1
       locoldub2 = locoldlb2 + oldsz2 - 1
       locoldub3 = locoldlb3 + oldsz3 - 1
       if(qsamelb.and.qsamelb2.and.qsamelb3.and.(newsz.eq.oldsz)&
            .and.(newsz2.eq.oldsz2).and.(newsz3.eq.oldsz3)) return

       ! step 2: (a) allocate a new, temporary array to hold the data while the original array is
       ! destroyed; (b) copy the data from the original to the temp. array.
       call chmalloc('realloc_mod.src','realloc_3d','tintg',newsz,newsz2,newsz3,&
            intg=tintg,lbou=newlb,lbou2=newlb2,lbou3=newlb3,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(prnlev.ge.2) write(outu,'(A)')  &
               'CHMALLOC failed in CHMREALLOC! Aborting reallocation!'
          if(present(ierr)) ierr = locerr
          return
       endif

       sz = min(newsz,oldsz)
       sz2 = min(newsz2,oldsz2)
       sz3 = min(newsz3,oldsz3)
       tintg(1+newlb-1:sz+newlb-1,1+newlb2-1:sz2+newlb2-1,1+newlb3-1:sz3+newlb3-1) = &
            intg(1+locoldlb-1:sz+locoldlb-1,1+locoldlb2-1:sz2+locoldlb2-1,1+locoldlb3-1:sz3+locoldlb3-1)

       ! step 3: (a) deallocate the original array, (b) allocate it at the new size/bounds
       call chmdealloc(filename,procname,arrayname,oldsz,oldsz2,oldsz3,&
            intg=intg,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMDEALLOC failed in CHMREALLOC! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_3d','tintg',newsz,newsz2,newsz3,&
               intg=tintg,ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif
       call chmalloc(filename,procname,arrayname,newsz,newsz2,newsz3,intg=intg,lbou=newlb,&
            lbou2=newlb2,lbou3=newlb3,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)') &
               'CHMALLOC failed allocating new size! Aborting reallocation!'
          call chmdealloc('realloc_mod.src','realloc_3d','tintg',newsz,newsz2,newsz3,&
               intg=tintg,ierr=locerr,qdie=qlocdie)
          if(locerr.ne.0) then
             if(prnlev.ge.2) write(outu,'(A)') &
                  'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
             if(present(ierr)) ierr = locerr
          endif
          return
       endif

       ! step 4: (a) copy the data back, (d) deallocate the temporary array

       intg(newlb:newub,newlb2:newub2,newlb3:newub3) = tintg(newlb:newub,newlb2:newub2,newlb3:newub3)
       call chmdealloc('realloc_mod.src','realloc_3d','tintg',newsz,newsz2,newsz3,&
            intg=tintg,ierr=locerr,qdie=qlocdie)
       if(locerr.ne.0) then
          if(present(ierr)) ierr = locerr
          if(prnlev.ge.2) write(outu,'(A)')  &
               'Severe problem: CHMREALLOC cannot deallocate temporary storage!!!'
       endif

    else
       write(6,*) 'No array matched list of available types.'
    endif

  END SUBROUTINE realloc_3d

END MODULE reallocation



