!> @file api_keywords.F90
!>
!> @brief C-binding shims that expose the build's pref.dat keyword list
!>        to non-Fortran callers (in particular, pyCHARMM).
!>
!> CHARMM is configured at compile time by enabling/disabling pref
!> keywords (`OPENMM`, `BLADE`, `DOMDEC`, `FFTDOCK`, ...). The set of
!> keywords baked into the running binary is held in
!> `keywords::pref_keys` (built from `keywords.inc.in` by CMake).
!> Internally CHARMM exposes that list via the `pref` script command,
!> which prints it to OUTU but provides no programmatic readback.
!>
!> The shims below give pyCHARMM (and any other C-API consumer) a
!> direct way to ask "is feature X compiled into this build?"
!> without having to parse `pref` output or guess from missing
!> symbols. The expected use is a one-time bulk fetch on the Python
!> side that builds a frozenset; subsequent `has(feature)` queries
!> are pure Python.
module api_keywords

  implicit none

contains

  !> @brief Number of pref keywords compiled into the running CHARMM
  !>        binary.
  !>
  !> Companion to ::api_keywords_get for sizing buffers on the C side.
  !>
  !> @return integer(c_int) the number of entries in
  !>         `keywords::pref_keys` for this build (typically ~100).
  function api_keywords_count() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_int
    use keywords, only: num_pref_keys

    implicit none

    integer(c_int) :: n

    n = num_pref_keys
  end function api_keywords_count

  !> @brief Maximum length, in characters, of any single pref keyword.
  !>
  !> Use this to size a per-keyword string buffer on the C side
  !> before calling ::api_keywords_get_all.
  !>
  !> @return integer(c_int) the LEN parameter of the
  !>         `keywords::pref_keys` array.
  function api_keywords_max_len() bind(c) result(max_len)
    use, intrinsic :: iso_c_binding, only: c_int
    use keywords, only: pref_keys

    implicit none

    integer(c_int) :: max_len

    max_len = len(pref_keys(1))
  end function api_keywords_max_len

  !> @brief Fill caller-allocated buffers with every pref keyword in
  !>        the running build.
  !>
  !> The caller must allocate ::api_keywords_count() string buffers
  !> of length ::api_keywords_max_len() + 1 and pass an array of
  !> `c_ptr` pointers to them as `out_keys`. Each buffer is
  !> NUL-terminated on return so the C side can treat it as a
  !> standard C string.
  !>
  !> Mirrors the bulk-fetch pattern used by `eval_get_all_params`
  !> (cf. source/api/api_eval.F90) and `builtins_reals_get`.
  !>
  !> @param[in,out] out_keys array of C pointers, one per keyword,
  !>                each pointing at a caller-allocated buffer of
  !>                at least ::api_keywords_max_len() + 1 bytes.
  subroutine api_keywords_get_all(out_keys) bind(c)
    use, intrinsic :: iso_c_binding, only: c_null_char, c_ptr
    use api_util, only: f2c_string
    use keywords, only: pref_keys, num_pref_keys

    implicit none

    type(c_ptr), target, dimension(*) :: out_keys

    integer :: i

    do i = 1, num_pref_keys
       call f2c_string(trim(pref_keys(i)) // c_null_char, out_keys(i), &
            len(pref_keys(i)) + 1)
    end do
  end subroutine api_keywords_get_all

end module api_keywords
