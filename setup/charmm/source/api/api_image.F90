!> manipulate settings for crystals and finite point groups
module api_image
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief defines a cubic lattice and constants for a new crystal
  !
  !> param[in] center_x
  !>           x component of the new center
  !> param[in] center_y
  !>           y component of the new center
  !> param[in] center_z
  !>           z component of the new center
  !> param[in] selection
  !>           int array: 1 == atom selected
  !> param[in] mode
  !>           int: 0 FIXE, 1 BYSE, 2 BYRE, 3 BYGR, 4 BYAT
  !> @return success
  !>         1 if success
  function image_setup_centering( &
       center_x, center_y, center_z, &
       selection, mode) &
       result(success)
    use bases_fcm, only: bimag
    use image, only: limcen, imxcen, imycen, imzcen
    use psf, only: natom
    use stream, only: outu, prnlev
    
    implicit none

    ! result
    integer :: success

    ! args
    real(8), intent(in) :: center_x, center_y, center_z
    integer, intent(in) :: selection(natom)
    integer, intent(in) :: mode

    ! locals
    integer :: i

    success = 0

    imxcen = center_x
    imycen = center_y
    imzcen = center_z
   
    if (mode < 0 .or. mode > 4) then
       call wrndie(-1, '<IMSPEC>', 'UNRECOGNIZED COMMAND')
    end if

    do i = 1, natom
       if(selection(i) .gt. 0) then
          bimag%imcenf(i) = mode
          limcen = .true.
       end if
    end do

    if (prnlev .ge. 2) then
       if (limcen) then
          write(outu, '(a)') ' IMAGE CENTERING ON FOR SOME ATOMS'
       else
          write(outu, '(a)') ' IMAGE CENTERING TURNED OFF'
       end if
    end if

    success = 1
  end function image_setup_centering

  !> @brief setup centering for a segment for the next image update
  !
  !> param[in] center_x
  !>           x component of the new center
  !> param[in] center_y
  !>           y component of the new center
  !> param[in] center_z
  !>           z component of the new center
  !> param[in] segid
  !>           string name of segment to center
  !> @return success
  !>         1 if success
  function image_setup_segment( &
       center_x, center_y, center_z, &
       c_segid) &
       result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int
    use api_select, only: select_segid_range
    use psf, only: natom
    use image, only: imxcen, imycen, imzcen
    implicit none

    ! result
    integer(c_int) :: success

    ! args
    real(c_double), intent(in) :: center_x, center_y, center_z
    character(len=1, kind=c_char), dimension(*) :: c_segid

    ! locals
    integer :: selection(natom)
    success = select_segid_range(c_segid, c_segid, selection)
    success = image_setup_centering(center_x, center_y, center_z, selection, 1)
  end function image_setup_segment

  !> @brief setup centering for a residue for the next image update
  !
  !> param[in] center_x
  !>           x component of the new center
  !> param[in] center_y
  !>           y component of the new center
  !> param[in] center_z
  !>           z component of the new center
  !> param[in] resname
  !>           string name of residue to center
  !> @return success
  !>         1 if success
  function image_setup_residue( &
       center_x, center_y, center_z, &
       c_resname) &
       result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int
    use api_select, only: select_resname_range
    use psf, only: natom
    implicit none

    ! result
    integer(c_int) :: success

    ! args
    real(c_double), intent(in) :: center_x, center_y, center_z
    character(len=1, kind=c_char), dimension(*) :: c_resname

    ! locals
    integer :: selection(natom)

    success = select_resname_range(c_resname, c_resname, selection)
    success = image_setup_centering(center_x, center_y, center_z, selection, 2)
  end function image_setup_residue

  !> @brief setup centering for a residue for the next image update
  !
  !> param[in] center_x
  !>           x component of the new center
  !> param[in] center_y
  !>           y component of the new center
  !> param[in] center_z
  !>           z component of the new center
  !> param[in] selection
  !>           selection(i) == 1 <=> atom i selected
  !> @return success
  !>         1 if success
  function image_setup_selection( &
       center_x, center_y, center_z, &
       selection) &
       result(success) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int

    implicit none

    ! result
    integer(c_int) :: success

    ! args
    real(c_double), intent(in) :: center_x, center_y, center_z
    integer(c_int), dimension(*), intent(in) :: selection

    success = image_setup_centering(center_x, center_y, center_z, selection, 4)
  end function image_setup_selection
  
  ! Addition Kai Toepfer May 2022
  !> @brief export a copy of xucell (unit cell parameter)
  !
  !> @param[out] out_ucell has 3+3 basis vector length and angle values
  !> @return success
  !>         1 <=> success
  function image_get_ucell(out_ucell) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use api_util, only: f2c_logical
    use image, only: xucell
    
    implicit none
    
    ! args
    real(c_double) :: out_ucell(*)
    
    ! locals
    logical :: qsuccess
    
    ! result
    integer(c_int) :: success
    
    qsuccess = .false.

    out_ucell(1:6) = xucell(1:6)

    qsuccess = .true.
    success = f2c_logical(qsuccess)
    
  end function image_get_ucell
  
  !> @brief export a copy of number of image cells
  !
  !> @param[out] out_ntrans integer number of image transformations
  !> @return success
  !>         1 <=> success
  function image_get_ntrans(out_ntrans) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use image, only: ntrans
    
    implicit none
    
    ! args
    integer(c_int) :: out_ntrans
    
    ! locals
    logical :: qsuccess
    
    ! result
    integer(c_int) :: success
    
    qsuccess = .false.

    out_ntrans = ntrans

    qsuccess = .true.
    success = f2c_logical(qsuccess)
    
  end function image_get_ntrans
  
  
  !> @brief Export image nonbond exclusion list sizes (post-UPIMNB/MKIMNB).
  !
  !> Lets Python observe MKIMNB buffer use without parsing ``RESIZING`` logs.
  !> ``niminb`` is the filled image-primary exclusion count; ``iminb_capacity``
  !> is ``size(bimag%IMINB)`` after any resize. When capacity equals niminb the
  !> next rebuild may resize again.
  function image_get_iminb_stats( &
       out_natom, out_natim, out_ntrans, out_nnb, &
       out_niminb, out_iminb_capacity, &
       out_nimnb, out_imjnb_capacity, &
       out_niming, out_mlpot_active) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_util, only: f2c_logical
    use bases_fcm, only: bimag
    use image, only: ntrans, natim
    use psf, only: natom, nnb
    use api_func, only: mlpot_is_set
    
    implicit none
    
    integer(c_int), intent(out) :: out_natom
    integer(c_int), intent(out) :: out_natim
    integer(c_int), intent(out) :: out_ntrans
    integer(c_int), intent(out) :: out_nnb
    integer(c_int), intent(out) :: out_niminb
    integer(c_int), intent(out) :: out_iminb_capacity
    integer(c_int), intent(out) :: out_nimnb
    integer(c_int), intent(out) :: out_imjnb_capacity
    integer(c_int), intent(out) :: out_niming
    integer(c_int), intent(out) :: out_mlpot_active
    integer(c_int) :: success
    
    out_natom = natom
    out_natim = natim
    out_ntrans = ntrans
    out_nnb = nnb
    out_niminb = bimag%NIMINB
    out_nimnb = bimag%NIMNB
    out_niming = bimag%NIMING
    out_mlpot_active = f2c_logical(mlpot_is_set())
    
    if (allocated(bimag%IMINB)) then
       out_iminb_capacity = size(bimag%IMINB)
    else
       out_iminb_capacity = 0
    endif
    
    if (allocated(bimag%IMJNB)) then
       out_imjnb_capacity = size(bimag%IMJNB)
    else
       out_imjnb_capacity = 0
    endif
    
    success = 1
    
  end function image_get_iminb_stats


  !> @brief Count MIC-mapped primary pairs in JNB + image IMJNB lists.
  function image_get_mic_pair_count() bind(c) result(n_pairs)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bimag, bnbnd
    use image, only: natim, ntrans
    use psf, only: natom

    implicit none

    integer(c_int) :: n_pairs
    integer :: i, w, istart, v, nb, pi, pj, ai, aj

    n_pairs = 0
    if (natom <= 0) return

    if (associated(bnbnd%jnb) .and. associated(bnbnd%inblo)) then
       do i = 1, natom
          if (i > 1) then
             istart = bnbnd%inblo(i-1) + 1
          else
             istart = 1
          endif
          do w = istart, bnbnd%inblo(i)
             ai = i - 1
             aj = bnbnd%jnb(w) - 1
             if (ai < aj) n_pairs = n_pairs + 1
          enddo
       enddo
    endif

    if (ntrans == 0 .or. natim <= natom) return
    if (.not. associated(bimag%imattr)) return
    if (.not. associated(bimag%imjnb) .or. .not. associated(bimag%imblo)) return

    do v = natom + 1, natim
       pi = bimag%imattr(v)
       if (v > 1) then
          istart = bimag%imblo(v-1) + 1
       else
          istart = 1
       endif
       do w = istart, bimag%imblo(v)
          nb = bimag%imjnb(w)
          if (nb <= natom) then
             pj = nb
          else
             pj = bimag%imattr(nb)
          endif
          ai = pi - 1
          aj = pj - 1
          if (ai < aj) n_pairs = n_pairs + 1
       enddo
    enddo
  end function image_get_mic_pair_count


  !> @brief Export MIC-mapped primary pairs from JNB + image IMJNB (0-based, i < j).
  function image_export_mic_pairs( &
       out_i, out_j, max_pairs, out_count) bind(c) result(success)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bimag, bnbnd
    use image, only: natim, ntrans
    use psf, only: natom

    implicit none

    integer(c_int), dimension(*), intent(out) :: out_i, out_j
    integer(c_int), value :: max_pairs
    integer(c_int), intent(out) :: out_count
    integer(c_int) :: success
    integer :: i, w, istart, v, nb, pi, pj, ai, aj, c

    success = 0
    out_count = 0
    if (max_pairs <= 0) return
    if (natom <= 0) return

    c = 0
    if (associated(bnbnd%jnb) .and. associated(bnbnd%inblo)) then
       do i = 1, natom
          if (i > 1) then
             istart = bnbnd%inblo(i-1) + 1
          else
             istart = 1
          endif
          do w = istart, bnbnd%inblo(i)
             if (c >= max_pairs) exit
             ai = i - 1
             aj = bnbnd%jnb(w) - 1
             if (ai < aj) then
                c = c + 1
                out_i(c) = ai
                out_j(c) = aj
             endif
          enddo
          if (c >= max_pairs) exit
       enddo
    endif

    if (c < max_pairs .and. ntrans /= 0 .and. natim > natom) then
       if (associated(bimag%imattr) .and. associated(bimag%imjnb) &
            .and. associated(bimag%imblo)) then
          do v = natom + 1, natim
             if (c >= max_pairs) exit
             pi = bimag%imattr(v)
             if (v > 1) then
                istart = bimag%imblo(v-1) + 1
             else
                istart = 1
             endif
             do w = istart, bimag%imblo(v)
                if (c >= max_pairs) exit
                nb = bimag%imjnb(w)
                if (nb <= natom) then
                   pj = nb
                else
                   pj = bimag%imattr(nb)
                endif
                ai = pi - 1
                aj = pj - 1
                if (ai < aj) then
                   c = c + 1
                   out_i(c) = ai
                   out_j(c) = aj
                endif
             enddo
          enddo
       endif
    endif

    out_count = c
    success = 1
  end function image_export_mic_pairs
  
  
  !> @brief Update image - primary atoms non bonded exclusion list
  !
  subroutine image_update_bimag() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use bases_fcm, only: bimag
    use upimag_util, only: upimnb
    
    implicit none
    
    call upimnb(bimag)
    
  end subroutine image_update_bimag
  
#endif /* KEY_LIBRARY */
end module api_image
