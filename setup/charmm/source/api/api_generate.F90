!> @brief functions to construct and manipulate the PSF
!
! The central data structure in CHARMM, the PSF holds lists giving every
! bond, bond angle, torsion angle, and improper torsion angle
! as well as information needed to generate the hydrogen bonds and
! the non-bonded list.
module api_generate
  use, intrinsic :: iso_c_binding

  implicit none

contains

  function generate_get_autod() bind(c) result(out_autod)
    use, intrinsic :: iso_c_binding, only: c_int
    use rtf, only: rtfautod
    use api_util, only: f2c_logical
    implicit none
    integer(c_int) :: out_autod

    out_autod = f2c_logical(rtfautod)
  end function generate_get_autod


  function generate_get_autot() bind(c) result(out_autot)
    use, intrinsic :: iso_c_binding, only: c_int
    use rtf, only: rtfautot
    use api_util, only: f2c_logical
    implicit none
    integer(c_int) :: out_autot

    out_autot = f2c_logical(rtfautot)
  end function generate_get_autot


  integer(c_int) function generate_segment(c_opts) bind(c)
    use api_types, only: &
         c_generate_settings, generate_settings, &
         c2f_generate_settings
    use genpsf_m, only: genpsf
    use psf, only: nseg, nictot

    implicit none

    type(c_generate_settings), intent(in) :: c_opts
    type(generate_settings) :: f_opts
    integer :: istart = 0

    generate_segment = 0

    f_opts = c2f_generate_settings(c_opts)

    if (trim(f_opts%patf) .eq. '') f_opts%patf = 'DEFA'

    if (trim(f_opts%patl) .eq. '') f_opts%patl = 'DEFA'

    If (nseg .le. 0) then
       istart = 1
    else
       istart = nictot(nseg) + 1
    end if

    call genpsf(f_opts, istart)

    generate_segment = 1 ! signal success
  end function generate_segment

  !> @brief generate a segment from the last read sequence
  !
  !> @param[in] new_segid an array of len=1 characters naming the new segment
  !> @param[in] len_segid the length of new_segid array
  integer(c_int) function generate_setup(new_segid, len_segid) bind(c)
#if KEY_CHEQ==1
    use cheq, only: qcgrp, molbl, molsrt, allocate_cheq
#endif
    use memory, only: chmalloc, chmdealloc
    use psf, only: &
         nseg, segid, &
         nres, maxres, &
         nictot, nbdrude, &
         ib, jb, igpbs, &
         ngrp, natom, nbond 
    use stream, only: outu, prnlev
    use string, only: encodi, trima
#if KEY_REPLICA==1
    use replica_mod, only: qrep
#endif
    use genpsf_m, only: genic
    use api_util, only: c2f_string
#ifdef KEY_RESIZE
    use resize, only: resize_psf
#endif    

    implicit none

    ! args
    character(len = 1, kind = c_char) :: new_segid(*)
    integer(kind = c_int) :: len_segid

    ! local vars
    character(len = 8) patf, patl
    character(len = 8) wrdtmp
    integer wtlen, i, idummy, istart, istop, nsegm
    integer, allocatable, dimension(:) :: i_sv

    generate_setup = -1

    nsegm  = nseg
    nseg   = nseg + 1
#ifdef KEY_RESIZE
    call resize_psf('api_generate.F90','generate_setup','NSEG',nseg, .true.)
#endif    

    patf = 'DEFA'
    patl = 'DEFA'

    segid(nseg) = c2f_string(new_segid, len_segid)
    if (segid(nseg) .eq. ' ') call encodi(nseg, segid(nseg), 8, idummy)

    do i = 1, nsegm
       if(segid(i) .eq. segid(nseg)) then
          nseg = nsegm
          call wrndie(-2, '<CHARMM>', 'DUPLICATE SEGMENT NAMES')
          return
       end if
    end do

    istart = nictot(nseg) + 1
    istop  = nres
#ifdef KEY_RESIZE
    call genic(istart, istop, .false., .true., patf, patl,.false.)
#else
    call chmalloc('genpsf.src', 'GENPSF', 'I_sv', maxres, intg = i_sv)
    call genic(istart, istop, .false., .true., patf, patl, i_sv, .false.)
    call chmdealloc('genpsf.src', 'GENPSF', 'I_sv', maxres, intg = i_sv)
#endif
#if KEY_CHEQ==1
#if KEY_REPLICA==1
    if (.not. qrep) then
#endif
       !  find molecule label
       if (allocated(molbl)) then
          ! psf changed, check whether arrays are still right size
          call allocate_cheq(natom, ngrp)
          call molsrt(natom, ib, jb, nbond, qcgrp, ngrp, igpbs)
          if (qcgrp .and. prnlev .ge. 2) then
             write(outu, '(a,/,a)') &
                  'GENPSF> Treat GRP as Molecules in CG dynamics.', &
                  '  This is true for the whole PSF.'
          end if
       end if
#if KEY_REPLICA==1
    end if
#endif
#endif /* KEY_CHEQ==1 */

    wrdtmp = segid(nseg)
    wtlen = len(wrdtmp)
    call trima(wrdtmp, wtlen)

    if (prnlev .ge. 2) then
       write(outu, '(a, i3, a, a, a)') &
            ' GENPSF> Segment ', nseg, &
            ' has been generated. Its identifier is ', wrdtmp(1:wtlen), '.'
    end if

    call psfsum(outu) ! print structure file counters

    nbdrude=nbond

    generate_setup = 1
  end function generate_setup

end module api_generate
