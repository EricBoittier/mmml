!> C-callable queries into the residue topology (RTF) module.
!
! Exposes what the currently-loaded RTF contains so that a C caller --
! pyCHARMM in particular -- can ask which residues and atom types are
! available, analogous to listing CHARMM's builtin/substitution
! variables.  The data live in module `rtf` (source/io/rtfio.F90):
!   aa(1:nrtrs)    residue names
!   atct(1:natct)  atom-type names
! Both are character(len=6) arrays.
!
! The name queries follow the same convention as api_eval's builtins
! list: call the *_num function to size the caller's buffers, then call
! the *_names subroutine with an array of c_ptr, one per name buffer.
module api_rtf
  implicit none
contains

  !> Maximum number of characters in a residue or atom-type name.
  !  Use to size the per-name buffers passed to the *_names routines.
  function rtf_name_max() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use rtf, only: aa
    implicit none
    integer(c_int) :: n
    n = len(aa)
  end function rtf_name_max

  !> Number of residues defined in the currently-loaded RTF.
  function rtf_num_residues() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use rtf, only: nrtrs
    implicit none
    integer(c_int) :: n
    n = nrtrs
  end function rtf_num_residues

  !> Fill out_names with the residue names.  out_names must be an array
  !  of rtf_num_residues() c_ptr, each pointing at a buffer of at least
  !  rtf_name_max() characters.
  subroutine rtf_residue_names(out_names) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use api_util, only: f2c_string
    use rtf, only: aa, nrtrs
    implicit none
    type(c_ptr), target, dimension(*) :: out_names
    integer :: i

    if (.not. allocated(aa)) return
    do i = 1, nrtrs
       call f2c_string(aa(i), out_names(i), len(aa))
    end do
  end subroutine rtf_residue_names

  !> Number of atom types defined in the currently-loaded RTF.
  function rtf_num_atom_types() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use rtf, only: natct
    implicit none
    integer(c_int) :: n
    n = natct
  end function rtf_num_atom_types

  !> Fill out_names with the atom-type names.  out_names must be an array
  !  of rtf_num_atom_types() c_ptr, each pointing at a buffer of at least
  !  rtf_name_max() characters.
  subroutine rtf_atom_type_names(out_names) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use api_util, only: f2c_string
    use rtf, only: atct, natct
    implicit none
    type(c_ptr), target, dimension(*) :: out_names
    integer :: i

    if (.not. allocated(atct)) return
    do i = 1, natct
       call f2c_string(atct(i), out_names(i), len(atct))
    end do
  end subroutine rtf_atom_type_names

end module api_rtf
