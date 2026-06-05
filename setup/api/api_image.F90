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
