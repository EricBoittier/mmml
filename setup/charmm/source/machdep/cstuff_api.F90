module cstuff
  use, intrinsic :: iso_c_binding, only: c_int

  implicit none

  interface
     ! this section references POSIX C functions
     integer(c_int) function setenv(envname, envval, overwrite) bind(C)
       use, intrinsic :: iso_c_binding, only: c_char, c_int
       implicit none
       character(kind=c_char) :: envname(*)
       character(kind=c_char) :: envval(*)
       integer(c_int), value :: overwrite
     end function setenv

     integer(c_int) function getpid() bind(C)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
     end function getpid

     ! this section references cstuff.c
     integer(c_int) function unbuffer_stdout() bind(C)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
     end function unbuffer_stdout

     subroutine readnamd(x, y, z, ptr_natom, fname, ptr_flen, ptr_ier) bind(C)
       use, intrinsic :: iso_c_binding, only: c_double, c_int, c_char
       implicit none
       real(c_double), dimension(*) :: x, y, z
       integer(c_int) :: ptr_natom, ptr_flen, ptr_ier
       character(kind=c_char) :: fname(*)
     end subroutine readnamd
     
     subroutine writenamd(x, y, z, ptr_natom, fname, ptr_flen, ptr_ier) bind(C)
       use, intrinsic :: iso_c_binding, only: c_double, c_int, c_char
       implicit none
       real(c_double), dimension(*) :: x, y, z
       integer(c_int) :: ptr_natom, ptr_flen, ptr_ier
       character(kind=c_char) :: fname(*)
     end subroutine writenamd     

     integer(c_int) function get_username(name, name_length) bind(C)
       use, intrinsic :: iso_c_binding, only: c_char, c_int
       implicit none
       character(kind=c_char) :: name(*)
       integer(c_int) :: name_length
     end function get_username

     integer(c_int) function expand_tilde(in_exp, in_length, &
       out_exp, out_length) bind(C)
       use, intrinsic :: iso_c_binding, only: c_char, c_int
       implicit none
       character(kind=c_char) :: in_exp(*), out_exp(*)
       integer(c_int), value :: in_length
       integer(c_int) :: out_length
     end function expand_tilde

     integer(c_int) function fsystem(com, com_length) bind(C)
       use, intrinsic :: iso_c_binding, only: c_char, c_int
       implicit none
       character(kind=c_char) :: com(*)
       integer(c_int), value :: com_length
     end function fsystem

     subroutine fputenv(env, len) bind(C)
       use, intrinsic :: iso_c_binding, only: c_char, c_int
       implicit none
       character(kind=c_char) :: env(*)
       integer(c_int), value :: len
     end subroutine fputenv

     subroutine stack_trace() bind(C)
       implicit none
     end subroutine stack_trace
  end interface
end module cstuff
