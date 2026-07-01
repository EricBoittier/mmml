module datstr
  use chm_kinds
  use chm_types
  implicit none

  interface NBGROW
    module procedure nbgrow_ints_size, nbgrow_ints_muladd, &
         nbgrow_reals_size, nbgrow_reals_muladd
  end interface NBGROW

  interface IMGROW
    module procedure imgrow_size, imgrow_muladd
  end interface IMGROW

contains
  LOGICAL FUNCTION USEDDT_nbond(BASE)
    !-----------------------------------------------------------------------
    !     USEDDT DETERMINES WHETHER THE DATA STRUCTURE IS IN USE.

    type(nonbondDataStructure) BASE

    USEDDT_nbond=associated(base%IBLO14)

    RETURN
  END FUNCTION USEDDT_nbond

  LOGICAL FUNCTION USEDDT_image(BASE)
    !-----------------------------------------------------------------------
    !     USEDDT DETERMINES WHETHER THE DATA STRUCTURE IS IN USE.

    type(imageDataStructure) BASE

    USEDDT_image=allocated(base%IMATPT)

    RETURN
  END FUNCTION USEDDT_image

  SUBROUTINE FREEDT_nbond(BASE)
    !-----------------------------------------------------------------------
    !     THIS ENTRY POINT FREES THE SPACES ASSOCIATED WITH A DATA
    !     STRUCTURE.
    !
    type(nonbondDataStructure) BASE

    if (.not. BASE%IS_ALIAS) then
       if (associated(base%INBLO)) deallocate(base%INBLO)
       if (associated(base%INBLOG)) deallocate(base%INBLOG)
       if (associated(base%INB14)) deallocate(base%INB14)
       if (associated(base%IBLO14)) deallocate(base%IBLO14)
       if (associated(base%ING14)) deallocate(base%ING14)
       if (associated(base%IGLO14)) deallocate(base%IGLO14)
       if (associated(base%LASTX)) deallocate(base%LASTX)
       if (associated(base%LASTY)) deallocate(base%LASTY)
       if (associated(base%LASTZ)) deallocate(base%LASTZ)
       if (associated(base%JNBG)) deallocate(base%JNBG)
       if (associated(base%JNB)) deallocate(base%JNB)
    endif
! NNBOPT, NNBDIS, NNBINT arrays of "freed" nonbondDataStructure deep copies "hang"
! (wasted memory)

    call UNALIAS_nbond(BASE)

    RETURN
  END SUBROUTINE FREEDT_nbond

  SUBROUTINE UNALIAS_nbond(BASE)
    !-----------------------------------------------------------------------
    ! Nullifies all pointers in the given nonbond data structure.
    !
    type(nonbondDataStructure) BASE

    nullify(base%INBLO)
    nullify(base%INBLOG)
    nullify(base%INB14)
    nullify(base%IBLO14)
    nullify(base%ING14)
    nullify(base%IGLO14)
    nullify(base%LASTX)
    nullify(base%LASTY)
    nullify(base%LASTZ)
    nullify(base%JNBG)
    nullify(base%JNB)

    RETURN
  END SUBROUTINE UNALIAS_nbond

  SUBROUTINE FREEDT_image(BASE)
    !-----------------------------------------------------------------------
    !     THIS ENTRY POINT FREES THE SPACES ASSOCIATED WITH A DATA
    !     STRUCTURE.
    !
    !need stat variable?
    type(imageDataStructure) BASE

    if(allocated(base%IMATPT)) deallocate(base%IMATPT)
    if(allocated(base%IMATTR)) deallocate(base%IMATTR)
    if(allocated(base%IMINB)) deallocate(base%IMINB)
    if(allocated(base%IMIBLO)) deallocate(base%IMIBLO)
    if(allocated(base%IMING)) deallocate(base%IMING)
    if(allocated(base%IMIGLO)) deallocate(base%IMIGLO)
    if(allocated(base%IMJNB)) deallocate(base%IMJNB)
    if(allocated(base%IMBLO)) deallocate(base%IMBLO)
    if(allocated(base%IMJNBS)) deallocate(base%IMJNBS)
    if(allocated(base%IMBLOS)) deallocate(base%IMBLOS)
    if(allocated(base%IMJNBG)) deallocate(base%IMJNBG)
    if(allocated(base%IMBLOG)) deallocate(base%IMBLOG)
    if(allocated(base%IMJNBX)) deallocate(base%IMJNBX)
    if(allocated(base%IMBLOX)) deallocate(base%IMBLOX)
    if(allocated(base%IMCENF)) deallocate(base%IMCENF)

    RETURN
  END SUBROUTINE FREEDT_image

  SUBROUTINE DUPLDT_nbond(NEWBAS,BASE)
    !
    ! Makes a deep copy of a nonbond data structure.
    !
    type(nonbondDataStructure),intent(inout) :: NEWBAS
    type(nonbondDataStructure),intent(in) :: BASE

    newbas%IS_ALIAS = .false.

    ! fixed-size arrays
    newbas%LNBOPT=base%LNBOPT
    newbas%NBDIST=base%NBDIST
    newbas%NBINTS=base%NBINTS

    call copy_ints_pp(newbas%INBLO, base%INBLO)
    call copy_ints_pp(newbas%INBLOG, base%INBLOG)
    call copy_ints_pp(newbas%INB14, base%INB14)
    call copy_ints_pp(newbas%IBLO14, base%IBLO14)
    call copy_ints_pp(newbas%ING14, base%ING14)
    call copy_ints_pp(newbas%IGLO14, base%IGLO14)
    call copy_reals_pp(newbas%LASTX, base%LASTX)
    call copy_reals_pp(newbas%LASTY, base%LASTY)
    call copy_reals_pp(newbas%LASTZ, base%LASTZ)
    call copy_ints_pp(newbas%JNBG, base%JNBG)
    call copy_ints_pp(newbas%JNB, base%JNB)

    RETURN
  END SUBROUTINE DUPLDT_nbond

  SUBROUTINE ALIASDT_nbond(NEWBAS,BASE)
    !-----------------------------------------------------------------------
    !
    ! Makes a shallow copy of a nonbond data structure.
    ! (except for deep copies of fixed-size arrays)
    type(nonbondDataStructure),intent(inout) :: NEWBAS
    type(nonbondDataStructure),intent(in) :: BASE

    newbas%IS_ALIAS = .true.

    ! fixed-size arrays
    newbas%LNBOPT=base%LNBOPT
    newbas%NBDIST=base%NBDIST
    newbas%NBINTS=base%NBINTS
    call alias_ints_pp(newbas%INBLO, base%INBLO)
    call alias_ints_pp(newbas%INBLOG, base%INBLOG)
    call alias_ints_pp(newbas%INB14, base%INB14)
    call alias_ints_pp(newbas%IBLO14, base%IBLO14)
    call alias_ints_pp(newbas%ING14, base%ING14)
    call alias_ints_pp(newbas%IGLO14, base%IGLO14)
    call alias_reals_pp(newbas%LASTX, base%LASTX)
    call alias_reals_pp(newbas%LASTY, base%LASTY)
    call alias_reals_pp(newbas%LASTZ, base%LASTZ)
    call alias_ints_pp(newbas%JNBG, base%JNBG)
    call alias_ints_pp(newbas%JNB, base%JNB)

    RETURN
  END SUBROUTINE ALIASDT_nbond

  SUBROUTINE COPY_nbond_excl(NEWBAS,BASE)
    !
    ! Makes a copy of a nonbond data structure exclusion info.
    !
    type(nonbondDataStructure),intent(inout) :: NEWBAS
    type(nonbondDataStructure),intent(in) :: BASE

    ! exclusion counts
    newbas%NBINTS(4)=base%NBINTS(4) ! NNB14
    newbas%NBINTS(5)=base%NBINTS(5) ! NNG14

    call copy_ints_pp(newbas%INB14, base%INB14)
    call copy_ints_pp(newbas%IBLO14, base%IBLO14)
    call copy_ints_pp(newbas%ING14, base%ING14)
    call copy_ints_pp(newbas%IGLO14, base%IGLO14)

    RETURN
  END SUBROUTINE COPY_nbond_excl

  SUBROUTINE DUPLDT_image(NEWBAS,BASE)
    !-----------------------------------------------------------------------
    !
    ! Makes a deep copy of an image data structure.
    !
    type(imageDataStructure),intent(inout) :: NEWBAS
    type(imageDataStructure),intent(in) :: BASE

    ! image structures are never aliases

    call copy_ints_aa(newbas%IMATPT, base%IMATPT)
    call copy_ints_aa(newbas%IMATTR, base%IMATTR)
    call copy_ints_aa(newbas%IMINB, base%IMINB)
    call copy_ints_aa(newbas%IMIBLO, base%IMIBLO)
    call copy_ints_aa(newbas%IMING, base%IMING)
    call copy_ints_aa(newbas%IMIGLO, base%IMIGLO)
    call copy_ints_aa(newbas%IMJNB, base%IMJNB)
    call copy_ints_aa(newbas%IMBLO, base%IMBLO)
    call copy_ints_aa(newbas%IMJNBS, base%IMJNBS)
    call copy_ints_aa(newbas%IMBLOS, base%IMBLOS)
    call copy_ints_aa(newbas%IMJNBG, base%IMJNBG)
    call copy_ints_aa(newbas%IMBLOG, base%IMBLOG)
    call copy_ints_aa(newbas%IMJNBX, base%IMJNBX)
    call copy_ints_aa(newbas%IMBLOX, base%IMBLOX)
    call copy_ints_aa(newbas%IMCENF, base%IMCENF)

    ! copy scalar Integers
    newbas%NIMINB=base%NIMINB
    newbas%NIMING=base%NIMING
    newbas%NIMNB=base%NIMNB
    newbas%NIMNBS=base%NIMNBS
    newbas%NIMNBG=base%NIMNBG
    newbas%NIMNBX=base%NIMNBX

    RETURN
  END SUBROUTINE DUPLDT_image

  subroutine copy_ints_aa(dest, src)
    !-----------------------------------------------------------------------
     integer,allocatable,dimension(:),intent(inout) :: dest
     integer,allocatable,dimension(:),intent(in) :: src
     if (allocated(dest)) deallocate(dest)
     if (allocated(src)) then
        allocate(dest(size(src)))
        dest = src
     endif
  end subroutine copy_ints_aa

  subroutine copy_ints_pp(dest, src)
    !-----------------------------------------------------------------------
     integer,pointer,dimension(:) :: dest
     integer,pointer,dimension(:) :: src
     if (associated(dest)) deallocate(dest)
     if (associated(src)) then
        allocate(dest(size(src)))
        dest = src
     else
        nullify(dest)
     endif
  end subroutine copy_ints_pp

  subroutine copy_reals_pp(dest, src)
    !-----------------------------------------------------------------------
     real(chm_real),pointer,dimension(:) :: dest
     real(chm_real),pointer,dimension(:) :: src
     if (associated(dest)) deallocate(dest)
     if (associated(src)) then
        allocate(dest(size(src)))
        dest = src
     else
        nullify(dest)
     endif
  end subroutine copy_reals_pp

  subroutine alias_ints_pa(dest, src)
    !-----------------------------------------------------------------------
     integer,pointer,dimension(:) :: dest
     integer,allocatable,dimension(:),target :: src
     if (allocated(src)) then
        dest => src
     else
        nullify(dest)
     endif
  end subroutine alias_ints_pa

  subroutine alias_ints_pp(dest, src)
    !-----------------------------------------------------------------------
     integer,pointer,dimension(:) :: dest
     integer,pointer,dimension(:) :: src
     if (associated(src)) then
        dest => src
     else
        nullify(dest)
     endif
  end subroutine alias_ints_pp

  subroutine alias_reals_pp(dest, src)
    !-----------------------------------------------------------------------
     real(chm_real),pointer,dimension(:) :: dest
     real(chm_real),pointer,dimension(:) :: src
     if (associated(src)) then
        dest => src
     else
        nullify(dest)
     endif
  end subroutine alias_reals_pp

  SUBROUTINE NBSET_FROM_IMG_SX(nb, img)
    !-----------------------------------------------------------------------
     type(nonbondDataStructure),intent(inout) :: nb
     type(imageDataStructure),intent(in) :: img
     call require_alias(nb)
     call alias_ints_pa(nb%INBLO, img%IMBLOS)
     call alias_ints_pa(nb%JNB, img%IMJNBS)
     call alias_ints_pa(nb%INBLOG, img%IMBLOX)
     call alias_ints_pa(nb%JNBG, img%IMJNBX)
  END SUBROUTINE NBSET_FROM_IMG_SX

  SUBROUTINE NBSET_FROM_IMG_G(nb, img)
    !-----------------------------------------------------------------------
     type(nonbondDataStructure),intent(inout) :: nb
     type(imageDataStructure),intent(in) :: img
     call require_alias(nb)
     call alias_ints_pa(nb%INBLO, img%IMBLO)
     call alias_ints_pa(nb%JNB, img%IMJNB)
     call alias_ints_pa(nb%INBLOG, img%IMBLOG)
     call alias_ints_pa(nb%JNBG, img%IMJNBG)
  END SUBROUTINE NBSET_FROM_IMG_G

  SUBROUTINE NBSET_B14_FROM_IMG(nb, img)
    !-----------------------------------------------------------------------
     type(nonbondDataStructure),intent(inout) :: nb
     type(imageDataStructure),intent(in) :: img
     call require_alias(nb)
     call alias_ints_pa(nb%IBLO14, img%IMIBLO)
     call alias_ints_pa(nb%INB14, img%IMINB)
  END SUBROUTINE NBSET_B14_FROM_IMG

  SUBROUTINE NBSET_G14_FROM_IMG(nb, img)
    !-----------------------------------------------------------------------
     type(nonbondDataStructure),intent(inout) :: nb
     type(imageDataStructure),intent(in) :: img
     call require_alias(nb)
     call alias_ints_pa(nb%IGLO14, img%IMIGLO)
     call alias_ints_pa(nb%ING14, img%IMING)
  END SUBROUTINE NBSET_G14_FROM_IMG

  subroutine nbgrow_ints_size(member, newsize)
    implicit none
     integer,pointer,dimension(:) :: member
     integer,intent(in) :: newsize
     if (associated(member)) then
        if (newsize > size(member)) then
           deallocate(member)
           allocate(member(newsize))
        endif
     else
        allocate(member(newsize))
     endif
     member(:) = 0
  end subroutine nbgrow_ints_size

  subroutine nbgrow_ints_muladd(member, factor, addl)
    implicit none
     integer,pointer,dimension(:) :: member
     real,intent(in) :: factor
     integer,intent(in) :: addl
     integer :: newsize
     if (associated(member)) then
        newsize = factor * size(member) + addl
        deallocate(member)
        allocate(member(newsize))
     else
        allocate(member(addl))
     endif
     member(:) = 0
  end subroutine nbgrow_ints_muladd

  subroutine nbgrow_reals_size(member, newsize)
     real(chm_real),pointer,dimension(:) :: member
     integer,intent(in) :: newsize
     if (associated(member)) then
        if (newsize > size(member)) then
           deallocate(member)
           allocate(member(newsize))
        endif
     else
        allocate(member(newsize))
     endif
     member(:) = 0.0
   end subroutine nbgrow_reals_size

  subroutine nbgrow_reals_muladd(member, factor, addl)
     implicit none
     real(chm_real),pointer,dimension(:) :: member
     real,intent(in) :: factor
     integer,intent(in) :: addl
     integer :: newsize
     if (associated(member)) then
        newsize = factor * size(member) + addl
        deallocate(member)
        allocate(member(newsize))
     else
        allocate(member(addl))
     endif
     member(:) = 0.0
  end subroutine nbgrow_reals_muladd

  subroutine imgrow_size(member, newsize)
    ! use memory, only: chmalloc, chmrealloc
    
    implicit none
    
     integer,allocatable,dimension(:) :: member
     integer,intent(in) :: newsize

     if (allocated(member)) then
        if (newsize > size(member)) then
           ! call chmrealloc(__FILE__, 'imgrow_size', 'member', &
           !      newsize, intg = member)
           deallocate(member)
           allocate(member(newsize))
        end if
     else
        ! call chmalloc(__FILE__, 'imgrow_size', 'member', &
        !      newsize, intg = member)
        allocate(member(newsize))
     end if
     member(:) = 0
   end subroutine imgrow_size

  subroutine imgrow_muladd(member, factor, addl)
    ! use memory, only: chmalloc, chmrealloc

    implicit none

    integer,allocatable,dimension(:) :: member
     real,intent(in) :: factor
     integer,intent(in) :: addl
     integer :: newsize
     if (allocated(member)) then
        newsize = factor * size(member) + addl
        ! call chmrealloc(__FILE__, 'imgrow_muladd', 'member', &
        !      newsize, intg = member)
        deallocate(member)
        allocate(member(newsize))
     else
        ! call chmalloc(__FILE__, 'imgrow_muladd', 'member', &
        !      newsize, intg = member)
        allocate(member(addl))
     endif
     member(:) = 0
   end subroutine imgrow_muladd

  subroutine require_alias(nb)
    !-----------------------------------------------------------------------
     type(nonbondDataStructure),intent(in) :: nb
     if (.not. nb%IS_ALIAS) then
        call wrndie(-4, '<NBSET_FROM_IMG>', &
              'Nonbond structure must be an alias in order to reassociate')
     endif
  end subroutine require_alias

  SUBROUTINE PRINTDT_nbond(iname,bname,BASE)
    !-----------------------------------------------------------------------
    !     Prints a summary of the nonbond data structure

  use stream

    character(len=*) iname,bname
    type(nonbondDataStructure) BASE

    !!QQQQ
    !!  write(outu,'(40x,l3)') allocated(base%inblo)

    write(outu,165) ' '
    write(outu,165) '>>>'
    call print_one_dt_intp(iname,bname,'INBLO',base%INBLO)
    call print_one_dt_intp(iname,bname,'INBLOG',base%INBLOG)
    write(outu,135) iname,bname,'LNBOPT',base%LNBOPT
135 format(' STRUCTURE: ',A,':',A,':',A,':',40L2)
    write(outu,145) iname,bname,'NBDIST',base%NBDIST
145 format(' STRUCTURE: ',A,':',A,':',A,':',4(/10(1PG12.4)))
    write(outu,155) iname,bname,'NBINTS',base%NBINTS
155 format(' STRUCTURE: ',A,':',A,':',A,':',10I5)
    call print_one_dt_intp(iname,bname,'INB14',base%INB14)
    call print_one_dt_intp(iname,bname,'IBLO14',base%IBLO14)
    call print_one_dt_intp(iname,bname,'ING14',base%ING14)
    call print_one_dt_intp(iname,bname,'IGLO14',base%IGLO14)
    call print_one_dt_r8p(iname,bname,'LASTX',base%LASTX)
    call print_one_dt_r8p(iname,bname,'LASTY',base%LASTY)
    call print_one_dt_r8p(iname,bname,'LASTZ',base%LASTZ)
    call print_one_dt_intp(iname,bname,'JNBG',base%JNBG)
    call print_one_dt_intp(iname,bname,'JNB',base%JNB)
    write(outu,165) '<<<'
    write(outu,165) ' '
165 format(A)


    RETURN
  END SUBROUTINE PRINTDT_nbond

  SUBROUTINE PRINTDT_image(iname,bname,BASE)
    !-----------------------------------------------------------------------
    !     Prints a summary of the image data structure
    !
  use stream

    character(len=*) iname,bname
    type(imageDataStructure) BASE

    write(outu,165) ' '
    write(outu,165) '>>>'
    call print_one_dt_intar(iname,bname,'IMATPT',base%IMATPT)
    call print_one_dt_intar(iname,bname,'IMATTR',base%IMATTR)
    call print_one_dt_int(iname,bname,'NIMINB',base%NIMINB)
    call print_one_dt_intar(iname,bname,'IMINB',base%IMINB)
    call print_one_dt_intar(iname,bname,'IMIBLO',base%IMIBLO)
    call print_one_dt_int(iname,bname,'NIMING',base%NIMING)
    call print_one_dt_intar(iname,bname,'IMING',base%IMING)
    call print_one_dt_intar(iname,bname,'IMIGLO',base%IMIGLO)
    call print_one_dt_int(iname,bname,'NIMNB',base%NIMNB)
    call print_one_dt_intar(iname,bname,'IMJNB',base%IMJNB)
    call print_one_dt_intar(iname,bname,'IMBLO',base%IMBLO)
    call print_one_dt_int(iname,bname,'NIMNBS',base%NIMNBS)
    call print_one_dt_intar(iname,bname,'IMJNBS',base%IMJNBS)
    call print_one_dt_intar(iname,bname,'IMBLOS',base%IMBLOS)
    call print_one_dt_int(iname,bname,'NIMNBG',base%NIMNBG)
    call print_one_dt_intar(iname,bname,'IMJNBG',base%IMJNBG)
    call print_one_dt_intar(iname,bname,'IMBLOG',base%IMBLOG)
    call print_one_dt_int(iname,bname,'NIMNBX',base%NIMNBX)
    call print_one_dt_intar(iname,bname,'IMJNBX',base%IMJNBX)
    call print_one_dt_intar(iname,bname,'IMBLOX',base%IMBLOX)
    call print_one_dt_intar(iname,bname,'IMCENF',base%IMCENF)
    write(outu,165) '<<<'
    write(outu,165) ' '
165 format(A)

    RETURN
  END SUBROUTINE PRINTDT_image

  subroutine print_one_dt_int(iname,bname,name,value)
    !-----------------------------------------------------------------------
  use stream
    character(len=*) iname,bname,name
    integer value

    write(outu,55) iname,bname,name,value
55  format(' STRUCTURE: ',A,':',A,':',A,':',5i8)

    RETURN
  END subroutine print_one_dt_int

  subroutine print_one_dt_intar(iname,bname,name,array)
    !-----------------------------------------------------------------------
  use stream
    character(len=*) iname,bname,name
    integer,allocatable,dimension(:) :: array

    integer i

    !!      write(outu,'(40x,l3)') allocated(array)

    if (allocated(array)) then
       i=size(array)
       write(outu,45) iname,bname,name,i
       if(i > 5) i=10
       write(outu,55) iname,bname,name,array(1:i)
    else
       write(outu,56) iname,bname,name
    endif

45  format(' STRUCTURE: ',A,':',A,':',A,'  size=',i8)
55  format('            ',A,':',A,':',A,':',10i5)
56  format('            ',A,':',A,':',A,': **NA**')

    RETURN
  END subroutine print_one_dt_intar

  subroutine print_one_dt_intp(iname,bname,name,ptr)
    !-----------------------------------------------------------------------
  use stream
    character(len=*) iname,bname,name
    integer,pointer,dimension(:) :: ptr

    integer i

    !!      write(outu,'(40x,l3)') allocated(array)

    if (associated(ptr)) then
       i=size(ptr)
       write(outu,45) iname,bname,name,i
       if(i > 5) i=10
       write(outu,55) iname,bname,name,ptr(1:i)
    else
       write(outu,56) iname,bname,name
    endif

45  format(' STRUCTURE: ',A,':',A,':',A,'  size=',i8)
55  format('            ',A,':',A,':',A,':',10i5)
56  format('            ',A,':',A,':',A,': **NA**')

    RETURN
  END subroutine print_one_dt_intp

  subroutine print_one_dt_r8p(iname,bname,name,ptr)
    !-----------------------------------------------------------------------
  use stream
    character(len=*) iname,bname,name
    real(chm_real),pointer,dimension(:) :: ptr

    integer i

    if (associated(ptr)) then
       i=size(ptr)
       write(outu,45) iname,bname,name,i
       if(i > 5) i=5
       write(outu,55) iname,bname,name,ptr(1:i)
    else
       write(outu,56) iname,bname,name
    endif

45  format(' STRUCTURE: ',A,':',A,':',A,'  size=',i8)
55  format('            ',A,':',A,':',A,':',5(1PG12.4))
56  format('            ',A,':',A,':',A,': **NA**')

    RETURN
  END subroutine print_one_dt_r8p

end module datstr


