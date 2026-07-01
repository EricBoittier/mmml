!> @brief routines to construct cartesian coordinates from internal coordinate values
!
! There are flexible editing commands for manipulating the data structure.
! This module, together with the coordinate
! manipulation commands (see the `coor` module) and the
! I/O commands (see the `charmm` module), provides model
! building functionality.
! The internal coordinate
! data structure can also be used for analysis purposes.
module api_ic
  use, intrinsic :: iso_c_binding, only: c_int

  implicit none

contains

  !> @brief fill internal coords with values from parameter file
  !
  !> @param[in] all_flag if nonzero, angle&bond vals filled over existing vals
  !> @return integer one indicates success, any other value indicates failure
  integer(c_int) function ic_fill_from_param_file(all_flag) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int

    use intcor_module, only: icr_struct
    use intcor2, only: purgic
    use param_store, only: set_param
    use psf, only: iac, imove, natom

    implicit none

    ! args
    integer(c_int) :: all_flag

    ! local vars
    logical all

    ic_fill_from_param_file = -1

    all = .true.
    if (all_flag == 0) all = .false.

    call purgic(icr_struct%lenic, &
         icr_struct%b1ic, icr_struct%b2ic, &
         icr_struct%t1ic, icr_struct%t2ic, &
         icr_struct%pic, icr_struct%iar, &
         icr_struct%jar, icr_struct%kar, &
         icr_struct%lar, icr_struct%tar, &
         .true., all, .false., .false., .false., &
         natom, imove, iac)

    call set_param('NIC',icr_struct%lenic)
    ic_fill_from_param_file = 1
  end function ic_fill_from_param_file

  !> @brief print the internal coordinates of a molecule
  !
  !> @return integer one indicates success, any other value indicates failure
  integer(c_int) function ic_print() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use, intrinsic :: iso_fortran_env, only: output_unit

    use intcor_module, only: icr_struct
    use intcor2, only: writic

    implicit none

    ic_print = -1

    call writic(1, icr_struct%lenic, 1, 1, output_unit, &
         icr_struct%b1ic, icr_struct%b2ic, &
         icr_struct%t1ic, icr_struct%t2ic, &
         icr_struct%pic, icr_struct%iar, &
         icr_struct%jar, icr_struct%kar, &
         icr_struct%lar, icr_struct%tar)

    ic_print = 1
  end function ic_print

  !> @brief returns index of atom with particular name
  !
  ! A return value of zero means the atom was not found.
  !
  !> @param[in] res_num the index of the residue that contains the atom
  !> @param[in] atom_name a string containing the name of the atom
  !> @return zero or the index of the atom
  integer function find_atom_index(res_num, atom_name)
    use, intrinsic :: iso_fortran_env, only: output_unit

    use chutil, only: matom
    use psf, only: atype, ibase, nictot, nseg
    use stream, only: wrnlev

    implicit none

    ! args
    integer :: res_num
    character(len = *) :: atom_name

    ! local vars
    integer :: atom_index, len_name, nres

    find_atom_index = 0

    if (res_num <= 0) then ! can't find this atom (bad residue number)
       if (wrnlev >= 2) write(output_unit, '(x, a, x, i4)') &
            'ERROR IN NXTATM: Unrecognizable residue number', &
            res_num

       return
    end if

    nres=nictot(nseg+1)
    atom_index = matom(res_num, atom_name, atype, ibase, 1, nres, .true.)

    if (atom_index <= 0) then ! can't find this atom (bad atom name)
       len_name = len_trim(atom_name)
       if(wrnlev >= 2) write(output_unit, '(x, a, a, a, x, i4)') &
            'ERROR IN NXTATM: Unrecognizable ATOM "', &
            atom_name(1:len_name), &
            '" for residue number:', res_num

    end if

    find_atom_index = atom_index
  end function find_atom_index

  !> @brief replace or create a dihedral angle
  !
  !> @param[in] res_num1 residue number for atom 1
  !> @param[in] atom_name1 atom name
  !> @param[in] res_num2 residue number for atom 2
  !> @param[in] atom_name2 atom name
  !> @param[in] res_num3 residue number for atom 3
  !> @param[in] atom_name3 atom name
  !> @param[in] res_num4 residue number for atom 4
  !> @param[in] atom_name4 atom name
  !> @param[in] new_psi the new angle in degrees between the two planes
  !> @return integer 1 == success, any other value indicates failure
  integer(c_int) function ic_edit_dihedral( &
       res_num1, atom_name1, &
       res_num2, atom_name2, &
       res_num3, atom_name3, &
       res_num4, atom_name4, &
       new_psi) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double

    use api_util, only: c2f_string
    use intcor_module, only: icr_struct
    use stream, only: outu, prnlev, wrnlev

    implicit none

    ! args
    integer(c_int) :: res_num1, res_num2, res_num3, res_num4
    character(len = 1, kind = c_char), dimension(*) :: &
         atom_name1, atom_name2, atom_name3, atom_name4
    real(c_double) :: new_psi

    ! local vars
    character(len = 8) :: atom_str1, atom_str2, atom_str3, atom_str4
    logical :: t, lpos, lneg, found
    integer :: iic, i, j, k, l, nnew, nold

    ic_edit_dihedral = -1

    nold = icr_struct%lenic
    nnew = 0

    atom_str1 = c2f_string(atom_name1, 8)
    i = find_atom_index(res_num1, atom_str1)

    atom_str2 = c2f_string(atom_name2, 8)
    j = find_atom_index(res_num2, atom_str2)

    atom_str3 = c2f_string(atom_name3, 8)
    k = find_atom_index(res_num3, atom_str3)
    t = k < 0
    if (t) k = -k

    atom_str4 = c2f_string(atom_name4, 8)
    l = find_atom_index(res_num4, atom_str4)

    IF(I <= 0.OR.J <= 0.OR.K <= 0.OR.L <= 0) THEN
       ! ATOM-DOESNT-EXIST
       if (wrnlev >= 2) write(outu, '(15x, a, x, a, x, a)') &
            '**** ERROR.', &
            'ATOM OF INTERNAL COORDINATE DOES NOT EXIST.', &
            'IGNORING ****'

       CALL DIEWRN(0)
    ELSE
       !
       FOUND=.FALSE.
       DO IIC=1,ICR_STRUCT%LENIC
          IF(I == ICR_STRUCT%IAR(IIC).AND.L == ICR_STRUCT%LAR(IIC)) THEN
             LPOS=(J == ICR_STRUCT%JAR(IIC).AND.K == ICR_STRUCT%KAR(IIC))
             LNEG=(J == ICR_STRUCT%KAR(IIC).AND.K == ICR_STRUCT%JAR(IIC))
          ELSE
             IF(I == ICR_STRUCT%LAR(IIC).AND.L == ICR_STRUCT%IAR(IIC)) THEN
                LNEG=(J == ICR_STRUCT%JAR(IIC).AND.K == ICR_STRUCT%KAR(IIC))
                LPOS=(J == ICR_STRUCT%KAR(IIC).AND.K == ICR_STRUCT%JAR(IIC))
             ELSE
                LNEG=.FALSE.
                LPOS=.FALSE.
             ENDIF
          ENDIF
          !
          IF(LNEG) THEN
             IF(PRNLEV >= 2) WRITE(OUTU,315) IIC
315          FORMAT(15X,'FOUND DIHEDRAL IN IC',I5,' AS OPPOSITE')
             IF(T.NEQV.ICR_STRUCT%TAR(IIC).AND.WRNLEV >= 2) WRITE(OUTU,316)
316          FORMAT(20X,'BUT TYPE VALUES DONT MATCH')
             IF(T.EQV.ICR_STRUCT%TAR(IIC)) FOUND=.TRUE.
             ICR_STRUCT%PIC(IIC)=-NEW_PSI
             IF(ICR_STRUCT%PIC(IIC) >  180.0) ICR_STRUCT%PIC(IIC)=ICR_STRUCT%PIC(IIC)-360.0
             IF(ICR_STRUCT%PIC(IIC) < -180.0) ICR_STRUCT%PIC(IIC)=ICR_STRUCT%PIC(IIC)+360.0
          ENDIF
          IF(LPOS) THEN
             IF(PRNLEV >= 2) WRITE(OUTU,325) IIC
             IF(T.NEQV.ICR_STRUCT%TAR(IIC) .AND. WRNLEV >= 2) &
                  WRITE(OUTU,316)
             ICR_STRUCT%PIC(IIC)=NEW_PSI
             IF(ICR_STRUCT%PIC(IIC) >  180.0) ICR_STRUCT%PIC(IIC)=ICR_STRUCT%PIC(IIC)-360.0
             IF(ICR_STRUCT%PIC(IIC) < -180.0) ICR_STRUCT%PIC(IIC)=ICR_STRUCT%PIC(IIC)+360.0
             IF(T.EQV.ICR_STRUCT%TAR(IIC)) FOUND=.TRUE.
325          FORMAT(15X,'FOUND IN IC',I5,' AS POSITIVE')
          ENDIF
       ENDDO

       IF (.NOT. FOUND) THEN
          ! add-new-ic-element
          ! TODO CHECK if LENIC is equal to INTLENX
          ICR_STRUCT%LENIC=ICR_STRUCT%LENIC+1
          NNEW=NNEW+1

          if (prnlev >= 2) write(outu, '(15x, a, i5)') &
               'DIHEDRAL NOT FOUND. ADDING NEW IC', &
               icr_struct%lenic

          ICR_STRUCT%IAR(ICR_STRUCT%LENIC)=I
          ICR_STRUCT%JAR(ICR_STRUCT%LENIC)=J
          ICR_STRUCT%KAR(ICR_STRUCT%LENIC)=K
          ICR_STRUCT%LAR(ICR_STRUCT%LENIC)=L
          ICR_STRUCT%TAR(ICR_STRUCT%LENIC)=T
          icr_struct%B1IC(ICR_STRUCT%LENIC)=0.0
          icr_struct%B2IC(ICR_STRUCT%LENIC)=0.0
          icr_struct%T1IC(ICR_STRUCT%LENIC)=0.0
          icr_struct%T2IC(ICR_STRUCT%LENIC)=0.0
          ICR_STRUCT%PIC(ICR_STRUCT%LENIC)=NEW_PSI
       END IF ! atom dne
    END IF ! .not. found

    if (icr_struct%lenic >= icr_struct%intlen) then
       call wrndie(-4, '<EDITIC>', &
            'Too many IC edits (out of memory), split the IC EDIT command')
    end if

    if (nnew .ne. 1) then
       ic_edit_dihedral = 1
       return
    end if

    ic_edit_dihedral = ic_update(nold)
  end function ic_edit_dihedral
  
  !> @brief replace or create an angle
  !
  !> @param[in] res_num1 residue number for atom 1
  !> @param[in] atom_name1 atom name
  !> @param[in] res_num2 residue number for atom 2
  !> @param[in] atom_name2 atom name
  !> @param[in] res_num3 residue number for atom 3
  !> @param[in] atom_name3 atom name
  !> @param[in] new_angle the new angle in degrees
  !> @return integer 1 == success, any other value indicates failure
  integer(c_int) function ic_edit_angle( &
       res_num1, atom_name1, &
       res_num2, atom_name2, &
       res_num3, atom_name3, &
       new_angle) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double

    use api_util, only: c2f_string
    use intcor_module, only: icr_struct
    use stream, only: outu, prnlev, wrnlev

    implicit none

    ! args
    integer(c_int) :: res_num1, res_num2, res_num3
    character(len = 1, kind = c_char), dimension(*) :: &
         atom_name1, atom_name2, atom_name3
    real(c_double) :: new_angle

    ! local vars
    character(len = 8) :: atom_str1, atom_str2, atom_str3, atom_str4
    logical :: found
    integer :: iic, i, j, k, ivl, nnew, nold

    ic_edit_angle = -1

    nold = icr_struct%lenic
    nnew = 0

    atom_str1 = c2f_string(atom_name1, 8)
    i = find_atom_index(res_num1, atom_str1)

    atom_str2 = c2f_string(atom_name2, 8)
    j = find_atom_index(res_num2, atom_str2)

    atom_str3 = c2f_string(atom_name3, 8)
    k = find_atom_index(res_num3, atom_str3)

    IF(I <= 0.OR.J <= 0.OR.K <= 0) THEN ! atom dne
       if (wrnlev >= 2) write(outu, '(15x, a, x, a, x, a)') &
            '**** ERROR.', &
            'ATOM OF INTERNAL COORDINATE DOES NOT EXIST.', &
            'IGNORING ****'

       CALL DIEWRN(0)
    ELSE
       FOUND=.FALSE.
       DO IIC=1,ICR_STRUCT%LENIC
          IVL=0
          IF(I == ICR_STRUCT%IAR(IIC).OR.K == ICR_STRUCT%IAR(IIC)) IVL=IVL+1
          IF(I == ICR_STRUCT%JAR(IIC).OR.K == ICR_STRUCT%JAR(IIC)) IVL=IVL+2
          IF(I == ICR_STRUCT%KAR(IIC).OR.K == ICR_STRUCT%KAR(IIC)) IVL=IVL+4
          IF(I == ICR_STRUCT%LAR(IIC).OR.K == ICR_STRUCT%LAR(IIC)) IVL=IVL+8
          IF(.NOT.((IVL /= 3.OR.J /= ICR_STRUCT%KAR(IIC).OR..NOT.ICR_STRUCT%TAR(IIC)) &
               .AND.(IVL /= 5.OR.J /= ICR_STRUCT%JAR(IIC).OR.ICR_STRUCT%TAR(IIC)))) THEN
             ICR_STRUCT%T2IC(IIC)=NEW_ANGLE
             FOUND=.TRUE.
             IF(PRNLEV >= 2) WRITE(OUTU,215) IIC

215          FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
          ENDIF
          IF(IVL == 10.AND.J == ICR_STRUCT%KAR(IIC)) THEN
             icr_struct%T1IC(IIC)=NEW_ANGLE
             FOUND=.TRUE.
             IF(PRNLEV >= 2) WRITE(OUTU,225) IIC

225          FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
          ENDIF
       ENDDO

       IF(.NOT.FOUND) THEN ! add new ic elt
          IF(WRNLEV >= 2) WRITE(OUTU,275)

275       FORMAT(/10X,'ERROR IN EDITIC. INTERNAL ANGLE ', &
               'COORDINATE CANT BE FOUND. ADDING ****'/)

          ! add new ic element
          icr_struct%lenic = icr_struct%lenic + 1
          if (prnlev >= 2) write(outu, '(15x, a, i5)') &
               'DIHEDRAL NOT FOUND. ADDING NEW IC', &
               icr_struct%lenic

          nnew = nnew + 1

          ICR_STRUCT%IAR(ICR_STRUCT%LENIC)=I
          ICR_STRUCT%JAR(ICR_STRUCT%LENIC)=K
          ICR_STRUCT%KAR(ICR_STRUCT%LENIC)=J
          ICR_STRUCT%LAR(ICR_STRUCT%LENIC)=-99
          ICR_STRUCT%TAR(ICR_STRUCT%LENIC)=.TRUE.
          icr_struct%B1IC(ICR_STRUCT%LENIC)=0.0
          icr_struct%B2IC(ICR_STRUCT%LENIC)=0.0
          icr_struct%T1IC(ICR_STRUCT%LENIC)=0.0
          ICR_STRUCT%T2IC(ICR_STRUCT%LENIC)=NEW_ANGLE
          icr_struct%PIC(ICR_STRUCT%LENIC)=0.0
       END IF ! .not. found
    END IF ! atom dne

    if (icr_struct%lenic >= icr_struct%intlen) then
       call wrndie(-4, '<EDITIC>', &
            'Too many IC edits (out of memory), split the IC EDIT command')
    end if

    if (nnew .ne. 1) then
       ic_edit_angle = 1
       return
    end if

    ic_edit_angle = ic_update(nold)
  end function ic_edit_angle
  
  !> @brief change the distance between two atoms
  !
  !> @param[in] res_num1 residue number for atom 1
  !> @param[in] atom_name1 atom name
  !> @param[in] res_num2 residue number for atom 2
  !> @param[in] atom_name2 atom name
  !> @param[in] new_dist the new distance
  !> @return integer 1 == success, any other value indicates failure
  integer(c_int) function ic_edit_dist( &
       res_num1, atom_name1, &
       res_num2, atom_name2, &
       new_dist) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double

    use api_util, only: c2f_string
    use intcor_module, only: icr_struct
    use stream, only: outu, prnlev, wrnlev

    implicit none

    ! args
    integer(c_int) :: res_num1, res_num2
    character(len = 1, kind = c_char), dimension(*) :: &
         atom_name1, atom_name2
    real(c_double) :: new_dist

    ! local vars
    character(len = 8) :: atom_str1, atom_str2 ! todo: replace 128
    integer nold, nnew, i, j, iic, ivl
    logical found

    ic_edit_dist = -1

    atom_str1 = c2f_string(atom_name1, 8)
    atom_str2 = c2f_string(atom_name2, 8)

    nold = icr_struct%lenic
    nnew = 0

    i = find_atom_index(res_num1, atom_str1)
    j = find_atom_index(res_num2, atom_str2)

    if (i <= 0 .or. j <= 0) then ! atom doesnt exist
       if (wrnlev >= 2) write(outu, '(15x, a, x, a, x, a)') &
            '**** ERROR.', &
            'ATOM OF INTERNAL COORDINATE DOES NOT EXIST.', &
            'IGNORING ****'

       call diewrn(0)
    else
       found = .false.
       do iic = 1, icr_struct%lenic
          ivl = 0
          if (i == icr_struct%iar(iic) .or. j == icr_struct%iar(iic)) ivl = ivl + 1
          if (i == icr_struct%jar(iic) .or. j == icr_struct%jar(iic)) ivl = ivl + 2
          if (i == icr_struct%kar(iic) .or. j == icr_struct%kar(iic)) ivl = ivl + 4
          if (i == icr_struct%lar(iic) .or. j == icr_struct%lar(iic)) ivl = ivl + 8

          if((ivl == 5 .and. icr_struct%tar(iic)) .or. &
               (ivl == 3 .and. .not. icr_struct%tar(iic))) then
             icr_struct%b2ic(iic) = new_dist
             found = .true.
             if (prnlev >= 2) write(outu, 115) iic

115          FORMAT(15X, 'FOUND IN IC', I5, ' ON LEFT  SIDE')
          end if

          if (ivl == 12) then
             icr_struct%b1ic(iic) = new_dist
             found = .true.
             if (prnlev >= 2) write(outu, 125) iic

125          FORMAT(15X, 'FOUND IN IC', I5, ' ON RIGHT SIDE')
          end if
       end do

       if (.not. found) then
          if (wrnlev >= 2) write(outu, 175)

175       format(/10x, 'ERROR IN EDITIC. INTERNAL DISTANCE ', &
               'COORDINATE CANT BE FOUND. ADDING ****'/)

          ! add new ic element
          icr_struct%lenic = icr_struct%lenic + 1
          if (prnlev >= 2) write(outu, '(15x, a, i5)') &
               'DIHEDRAL NOT FOUND. ADDING NEW IC', &
               icr_struct%lenic

          nnew = nnew + 1
          icr_struct%iar(icr_struct%lenic) = i
          icr_struct%jar(icr_struct%lenic) = j
          icr_struct%kar(icr_struct%lenic) = -99
          icr_struct%lar(icr_struct%lenic) = -99
          icr_struct%tar(icr_struct%lenic) = .false.
          icr_struct%b1ic(icr_struct%lenic) = 0.0
          icr_struct%b2ic(icr_struct%lenic) = new_dist
          icr_struct%t1ic(icr_struct%lenic) = 0.0
          icr_struct%t2ic(icr_struct%lenic) = 0.0
          icr_struct%pic(icr_struct%lenic) = 0.0
       end if
    end if

    if (icr_struct%lenic >= icr_struct%intlen) then
       call wrndie(-4, '<EDITIC>', &
            'Too many IC edits (out of memory), split the IC EDIT command')
    end if

    if (nnew .ne. 1) then
       ic_edit_dist = 1
       return
    end if

    ic_edit_dist = ic_update(nold)
  end function ic_edit_dist

  !> @brief update the internal coords based on newly edited angles/distances
  !
  ! to be called at the end off all ic_edit functions
  !
  !> @param[in] old_end the last index in the old internal coord table
  !> @return integer error code, success == 1
  integer function ic_update(old_end) bind(c)
    use intcor_module, only: icr_struct
    use stream, only: outu, prnlev, wrnlev
    use intcor2, only: writic
    use param_store, only: set_param

    implicit none

    ! args
    integer, intent(in) :: old_end

    ! local vars
    character(len = 128) :: atom_str1, atom_str2 ! the 128 needs to be updated to the real max

    integer inc, i, j, k, l, iic
    integer istart

    logical kill

    ic_update = 0
    kill = .false.

    ! this should be made into a function
    inc = old_end + 1
    i = icr_struct%iar(inc)
    j = icr_struct%jar(inc)
    if (icr_struct%tar(inc)) j = icr_struct%kar(inc)

    if (icr_struct%b2ic(inc) < 0.001) then
       loop420: do iic = 1, icr_struct%lenic
          if (iic == inc) cycle loop420
          if (icr_struct%iar(iic) == i .and. icr_struct%jar(iic) == j .and. .not. icr_struct%tar(iic)) &
               goto 415
          if (icr_struct%iar(iic) == j .and. icr_struct%jar(iic) == i .and. .not. icr_struct%tar(iic)) &
               goto 415
          if (icr_struct%iar(iic) == i .and. icr_struct%kar(iic) == j .and. icr_struct%tar(iic)) goto 415
          if (icr_struct%iar(iic) == j .and. icr_struct%kar(iic) == i .and. icr_struct%tar(iic)) goto 415
          if (icr_struct%lar(iic) == i .and. icr_struct%kar(iic) == j) goto 417
          if (icr_struct%lar(iic) == j .and. icr_struct%kar(iic) == i) goto 417
          cycle loop420
415       if(icr_struct%b2ic(iic) <= 0.001) cycle loop420
          icr_struct%b2ic(inc) = icr_struct%b2ic(iic)
          cycle loop420
417       if(icr_struct%b1ic(iic) <= 0.001) cycle loop420
          icr_struct%b2ic(inc) = icr_struct%b1ic(iic)
       end do loop420
       if (icr_struct%b2ic(inc) <= 0.001) kill = .true.
    end if

    k = icr_struct%kar(inc)
    l = icr_struct%lar(inc)
    if (icr_struct%b1ic(inc) < 0.001) then ! goto 460
       loop450: do iic = 1, icr_struct%lenic
          if (iic == inc) cycle loop450
          if(icr_struct%iar(iic) == k.and.icr_struct%jar(iic) == l.and..not.icr_struct%tar(iic)) &
               goto 445
          if(icr_struct%iar(iic) == l.and.icr_struct%jar(iic) == k.and..not.icr_struct%tar(iic)) &
               goto 445
          if(icr_struct%iar(iic) == k.and.icr_struct%kar(iic) == l.and.icr_struct%tar(iic)) goto 445
          if(icr_struct%iar(iic) == l.and.icr_struct%kar(iic) == k.and.icr_struct%tar(iic)) goto 445
          if(icr_struct%lar(iic) == k.and.icr_struct%kar(iic) == l) goto 447
          if(icr_struct%lar(iic) == l.and.icr_struct%kar(iic) == k) goto 447
          cycle loop450
445       if(icr_struct%b2ic(iic) <= 0.001) cycle loop450
          icr_struct%b1ic(inc)=icr_struct%b2ic(iic)
          cycle loop450
447       if(icr_struct%b1ic(iic) <= 0.001) cycle loop450
          icr_struct%b1ic(inc)=icr_struct%b1ic(iic)
       end do loop450
       if(icr_struct%b1ic(inc) <= 0.001) kill=.true.
    end if   !460  continue

    if (icr_struct%tar(inc)) k = icr_struct%jar(inc)
    if (icr_struct%t2ic(inc) < 0.001) then !goto 500
       loop490: do iic = 1, icr_struct%lenic
          if(iic == inc) cycle loop490
          if(icr_struct%kar(iic) /= j) goto 470
          if(icr_struct%lar(iic) == i.and.icr_struct%jar(iic) == k) goto 487
          if(icr_struct%lar(iic) == k.and.icr_struct%jar(iic) == i) goto 487
          if(icr_struct%iar(iic) == i.and.icr_struct%jar(iic) == k.and.icr_struct%tar(iic)) goto 485
          if(icr_struct%iar(iic) == k.and.icr_struct%jar(iic) == i.and.icr_struct%tar(iic)) goto 485
          cycle loop490
470       if(icr_struct%jar(iic) /= j.or..not.icr_struct%tar(iic)) cycle loop490
          if(icr_struct%iar(iic) == i.and.icr_struct%kar(iic) == k) goto 485
          if(icr_struct%iar(iic) == k.and.icr_struct%kar(iic) == i) goto 485
          cycle loop490
485       if(icr_struct%t2ic(iic) <= 0.001) cycle loop490
          icr_struct%t2ic(inc)=icr_struct%t2ic(iic)
          cycle loop490
487       if(icr_struct%t1ic(iic) <= 0.001) cycle loop490
          icr_struct%t2ic(inc)=icr_struct%t1ic(iic)
       end do loop490
       if (icr_struct%t2ic(inc) <= 0.001) kill=.true.
    end if  !500  continue

    i=icr_struct%lar(inc)
    j=icr_struct%kar(inc)
    k=icr_struct%jar(inc)
    if (icr_struct%t1ic(inc) < 0.001) then !goto 540
       loop530: do iic=1,icr_struct%lenic
          if(iic == inc) cycle loop530
          !
          if(icr_struct%kar(iic) /= j) goto 510
          if(icr_struct%lar(iic) == i.and.icr_struct%jar(iic) == k) goto 527
          if(icr_struct%lar(iic) == k.and.icr_struct%jar(iic) == i) goto 527
          if(icr_struct%iar(iic) == i.and.icr_struct%jar(iic) == k.and.icr_struct%tar(iic)) goto 525
          if(icr_struct%iar(iic) == k.and.icr_struct%jar(iic) == i.and.icr_struct%tar(iic)) goto 525
          cycle loop530
510       if(icr_struct%jar(iic) /= j.or..not.icr_struct%tar(iic)) cycle loop530
          if(icr_struct%iar(iic) == i.and.icr_struct%kar(iic) == k) goto 525
          if(icr_struct%iar(iic) == k.and.icr_struct%kar(iic) == i) goto 525
          cycle loop530
525       if(icr_struct%t2ic(iic) <= 0.001) cycle loop530
          icr_struct%t1ic(inc)=icr_struct%t2ic(iic)
          cycle loop530
527       if(icr_struct%t1ic(iic) <= 0.001) cycle loop530
          icr_struct%t1ic(inc)=icr_struct%t1ic(iic)
       enddo loop530
       if (icr_struct%t1ic(inc) <= 0.001) kill = .true.
    end if ! 540 continue

    if(kill .and. wrnlev >= 2) write(outu,905)

905 FORMAT(/,15X,'ERROR IN EDIT TERMINATION. SOME NEW INTERNAL', &
         ' COORDINATES UNDEFINED. CONTINUING'/)

    istart = old_end + 1
    if (istart > icr_struct%lenic) return

    if (prnlev >= 2) write(outu, 605)

605 FORMAT(/15X, 'EDIT COMPLETE. NEW COORDINATES ARE GIVEN.'/)

    call writic(istart, icr_struct%lenic, &
         -1, 0, outu, &
         icr_struct%b1ic, icr_struct%b2ic, &
         icr_struct%t1ic, icr_struct%t2ic, &
         icr_struct%pic, &
         icr_struct%iar, icr_struct%jar, icr_struct%kar, icr_struct%lar, &
         icr_struct%tar)

    call set_param('NIC', icr_struct%lenic)

    ic_update = 1
  end function ic_update

  !> @brief place first three atoms for building reference
  !
  ! When the cartesian coordinates are not specified for any atoms,
  ! the BUILd command cannot be used to generate positions since all positions
  ! are determined relative to known positions. The SEED command specifies the
  ! positions of the three atoms. It puts the first at the origin, the second
  ! on the x-axis, and the third in the xy-plane. The three atoms must have
  ! entries in the IC file corresponding to: dist 1-2, angle 1-2-3, dist 2-3.
  !
  !> @param[in] res_num1 residue number for atom 1
  !> @param[in] atom_name1 atom name
  !> @param[in] res_num2 residue number for atom 2
  !> @param[in] atom_name2 atom name
  !> @param[in] res_num3 residue number for atom 3
  !> @param[in] atom_name3 atom name
  !> @param[in] new_angle the new angle in degrees
  !> @return integer 1 == success, any other value indicates failure
  integer(c_int) function ic_seed( &
       res_num1, atom_name1, &
       res_num2, atom_name2, &
       res_num3, atom_name3) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char

    use api_util, only: c2f_string
    use coord, only: x, y, z
    use intcor_module, only: icr_struct
    use intcor2, only: seed

    implicit none

    ! args
    integer(c_int) :: res_num1, res_num2, res_num3
    character(len = 1, kind = c_char), dimension(*) :: &
         atom_name1, atom_name2, atom_name3

    ! local vars
    character(len = 8) :: atom_str1, atom_str2, atom_str3, atom_str4
    integer :: i, j, k

    ic_seed = -1

    atom_str1 = c2f_string(atom_name1, 8)
    i = find_atom_index(res_num1, atom_str1)

    atom_str2 = c2f_string(atom_name2, 8)
    j = find_atom_index(res_num2, atom_str2)

    atom_str3 = c2f_string(atom_name3, 8)
    k = find_atom_index(res_num3, atom_str3)

    if (i <= 0 .or. j <= 0 .or. k <= 0) then
       call wrndie(0,'<INTCOR>','ATOM OF SEED DOES NOT EXIST.')
       return
    end if

    call seed(i, j, k, &
         x, y, z, &
         icr_struct%lenic, &
         icr_struct%b1ic, icr_struct%b2ic, &
         icr_struct%t1ic, icr_struct%t2ic, &
         icr_struct%pic, icr_struct%iar, &
         icr_struct%jar, icr_struct%kar, &
         icr_struct%lar, icr_struct%tar)

    ic_seed = 1
  end function ic_seed

  !> @brief determine coordinates for all unspecified atoms
  !
  !> @return integer 1 == success, any other value indicates failure
  integer(c_int) function ic_build() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char

    use coord, only: x, y, z
    use intcor_module, only: icr_struct
    use intcor2, only: bildc
    use psf, only: natom

    implicit none

    call bildc(1, icr_struct%lenic, x, y, z, &
         icr_struct%b1ic, icr_struct%b2ic, &
         icr_struct%t1ic, icr_struct%t2ic, &
         icr_struct%pic, icr_struct%iar, &
         icr_struct%jar, icr_struct%kar, &
         icr_struct%lar, icr_struct%tar, &
         natom)

    ic_build = 1
  end function ic_build

end module api_ic
