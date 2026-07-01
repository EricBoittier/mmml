module blade_coords_module
  implicit none

#if KEY_BLADE == 1
  interface
     subroutine blade_init_coordinates(system, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: n
     end subroutine blade_init_coordinates

     subroutine blade_dest_coordinates(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_dest_coordinates

     subroutine blade_add_coordinates_position(system, i, x, y, z) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i
       real(c_double), value :: x, y, z
     end subroutine blade_add_coordinates_position

     subroutine blade_add_coordinates_velocity(system, i, vx, vy, vz) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i
       real(c_double), value :: vx, vy, vz
     end subroutine blade_add_coordinates_velocity

     subroutine blade_add_coordinates_box(system, n,&
          a, b, c, alpha, beta, gamma) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: n
       real(c_double), value :: &
            a, b, c, alpha, beta, gamma
     end subroutine blade_add_coordinates_box

     function blade_get_atom_count(system) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int) :: n
     end function blade_get_atom_count

     function charmm_recv_position(system, out_pos) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_pos(*)
       integer(c_int) :: n
     end function charmm_recv_position

     function charmm_send_position(system, out_pos) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_pos(*)
       integer(c_int) :: n
     end function charmm_send_position

     function charmm_recv_velocity(system, out_vel) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_vel(*)
       integer(c_int) :: n
     end function charmm_recv_velocity

     function charmm_send_velocity(system, out_vel) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_vel(*)
       integer(c_int) :: n
     end function charmm_send_velocity

     function blade_get_lambda_count(system) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int) :: n
     end function blade_get_lambda_count

     function charmm_recv_theta(system, out_the) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_the(*)
       integer(c_int) :: n
     end function charmm_recv_theta

     function charmm_send_theta(system, out_the) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_the(*)
       integer(c_int) :: n
     end function charmm_send_theta

     function charmm_recv_thetavelocity(system, out_the) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_the(*)
       integer(c_int) :: n
     end function charmm_recv_thetavelocity

     function charmm_send_thetavelocity(system, out_the) result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_the(*)
       integer(c_int) :: n
     end function charmm_send_thetavelocity

     subroutine charmm_recv_box(system, out_box) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_box(6)
     end subroutine charmm_recv_box

     subroutine charmm_send_box(system, out_box) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_box(6)
     end subroutine charmm_send_box

     function blade_get_energy_count() result(n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int

       implicit none

       integer(c_int) :: n
     end function blade_get_energy_count

     subroutine charmm_recv_energy(system, out_energy) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_energy(*)
     end subroutine charmm_recv_energy

     subroutine charmm_recv_force(system, out_force, out_flambda) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: out_force(*), out_flambda(*)
     end subroutine charmm_recv_force

     subroutine blade_calc_lambda_from_theta(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_calc_lambda_from_theta

     subroutine blade_init_lambda_from_theta(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_init_lambda_from_theta

     subroutine blade_prettify_position(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_prettify_position

      subroutine blade_recv_state(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_state

      subroutine blade_send_state(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_send_state

      subroutine blade_recv_position(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_position

      subroutine blade_recv_lambda(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_lambda

      subroutine blade_recv_theta(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_theta

      subroutine blade_recv_energy(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_energy

      subroutine blade_recv_force(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_recv_force
  end interface

contains

  subroutine copy_coords_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use coord, only: x, y, z
    use psf, only: natom
    use image, only: qorth, xtlrot

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_pos(:)
    integer :: n, nelts, err

    n = blade_get_atom_count(system)
    nelts = 3 * n

    allocate(blade_pos(nelts), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_coords_b2c', &
            'memory allocation error')
    end if

    n = charmm_recv_position(system, blade_pos)

    if (n .ne. natom) then
       call wrndie(-5, 'copy_coords_b2c', &
            'number of atoms differ between blade and charmm')
    end if

    if (qorth) then
       x(1:n) = blade_pos(1:nelts:3)
       y(1:n) = blade_pos(2:nelts:3)
       z(1:n) = blade_pos(3:nelts:3)
    else
       x(1:n) = xtlrot(1,1)*blade_pos(1:nelts:3)+xtlrot(2,1)*blade_pos(2:nelts:3)+xtlrot(3,1)*blade_pos(3:nelts:3)
       y(1:n) = xtlrot(1,2)*blade_pos(1:nelts:3)+xtlrot(2,2)*blade_pos(2:nelts:3)+xtlrot(3,2)*blade_pos(3:nelts:3)
       z(1:n) = xtlrot(1,3)*blade_pos(1:nelts:3)+xtlrot(2,3)*blade_pos(2:nelts:3)+xtlrot(3,3)*blade_pos(3:nelts:3)
    endif

    deallocate(blade_pos, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_coords_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_coords_b2c

  subroutine copy_coords_c2b(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use coord, only: x, y, z
    use psf, only: natom
    use image, only: qorth, xtlrot

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_pos(:)
    integer :: i, n, nelts, err

    n = blade_get_atom_count(system)
    nelts = 3 * n

    allocate(blade_pos(nelts), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_coords_c2b', &
            'memory allocation error')
    end if

    if (n .ne. natom) then
       call wrndie(-5, 'copy_coords_c2b', &
            'number of atoms differ between blade and charmm')
    end if

    do i = 1,n
      if (qorth) then
         blade_pos(3*i-2) = x(i)
         blade_pos(3*i-1) = y(i)
         blade_pos(3*i-0) = z(i)
      else
         blade_pos(3*i-2) = xtlrot(1,1)*x(i)+xtlrot(1,2)*y(i)+xtlrot(1,3)*z(i)
         blade_pos(3*i-1) = xtlrot(2,1)*x(i)+xtlrot(2,2)*y(i)+xtlrot(2,3)*z(i)
         blade_pos(3*i-0) = xtlrot(3,1)*x(i)+xtlrot(3,2)*y(i)+xtlrot(3,3)*z(i)
      endif
    enddo

    n = charmm_send_position(system, blade_pos)

    deallocate(blade_pos, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_coords_c2b', &
            'memory de-allocation error')
    end if
  end subroutine copy_coords_c2b

  subroutine copy_velocity_b2c(system,vx,vy,vz)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use chm_kinds

    use psf, only: natom
    use image, only: qorth, xtlrot

    implicit none

    type(c_ptr) :: system
    real(chm_real), intent(inout) :: vx(:), vy(:), vz(:)

    real(c_double), allocatable :: blade_vel(:)
    integer :: n, nelts, err

    n = blade_get_atom_count(system)
    nelts = 3 * n

    allocate(blade_vel(nelts), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_velocity_b2c', &
            'memory allocation error')
    end if

    n = charmm_recv_velocity(system, blade_vel)

    if (n .ne. natom) then
       call wrndie(-5, 'copy_velocity_b2c', &
            'number of atoms differ between blade and charmm')
    end if

    if (qorth) then
       vx(1:n) = blade_vel(1:nelts:3)
       vy(1:n) = blade_vel(2:nelts:3)
       vz(1:n) = blade_vel(3:nelts:3)
    else
       vx(1:n) = xtlrot(1,1)*blade_vel(1:nelts:3)+xtlrot(2,1)*blade_vel(2:nelts:3)+xtlrot(3,1)*blade_vel(3:nelts:3)
       vy(1:n) = xtlrot(1,2)*blade_vel(1:nelts:3)+xtlrot(2,2)*blade_vel(2:nelts:3)+xtlrot(3,2)*blade_vel(3:nelts:3)
       vz(1:n) = xtlrot(1,3)*blade_vel(1:nelts:3)+xtlrot(2,3)*blade_vel(2:nelts:3)+xtlrot(3,3)*blade_vel(3:nelts:3)
    endif

    deallocate(blade_vel, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_velocity_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_velocity_b2c

  subroutine copy_velocity_c2b(system,vx,vy,vz)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use chm_kinds

    use psf, only: natom
    use image, only: qorth, xtlrot

    implicit none

    type(c_ptr) :: system
    real(chm_real), intent(inout) :: vx(:), vy(:), vz(:)

    real(c_double), allocatable :: blade_vel(:)
    integer :: i, n, nelts, err

    n = blade_get_atom_count(system)
    nelts = 3 * n

    allocate(blade_vel(nelts), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_velocity_c2b', &
            'memory allocation error')
    end if

    if (n .ne. natom) then
       call wrndie(-5, 'copy_velocity_c2b', &
            'number of atoms differ between blade and charmm')
    end if

    do i = 1,n
      if (qorth) then
         blade_vel(3*i-2) = vx(i)
         blade_vel(3*i-1) = vy(i)
         blade_vel(3*i-0) = vz(i)
      else
         blade_vel(3*i-2) = xtlrot(1,1)*vx(i)+xtlrot(1,2)*vy(i)+xtlrot(1,3)*vz(i)
         blade_vel(3*i-1) = xtlrot(2,1)*vx(i)+xtlrot(2,2)*vy(i)+xtlrot(2,3)*vz(i)
         blade_vel(3*i-0) = xtlrot(3,1)*vx(i)+xtlrot(3,2)*vy(i)+xtlrot(3,3)*vz(i)
      endif
    enddo

    n = charmm_send_velocity(system, blade_vel)

    deallocate(blade_vel, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_velocity_c2b', &
            'memory de-allocation error')
    end if
  end subroutine copy_velocity_c2b

#if KEY_BLOCK == 1
  subroutine copy_theta_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use lambdam, only: thetamld, nsitemld, nsubmld, blckmld
    use block_ltm, only: nblock

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_the(:)
    integer :: n, err
    integer :: i, j

    n = blade_get_lambda_count(system)

    allocate(blade_the(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_theta_b2c', &
            'memory allocation error')
    end if

    n = charmm_recv_theta(system, blade_the)

    if (n .ne. nblock) then
       call wrndie(-5, 'copy_theta_b2c', &
            'number of blocks differ between blade and charmm')
    end if

    ! bixlam(1:n) = blade_the(1:n) ! so much easier for lambda
    do i = 2, nsitemld
       do j = 1, nsubmld(i)
          thetamld(i,j)=blade_the(blckmld(i,j))
       enddo
    enddo

    deallocate(blade_the, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_theta_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_theta_b2c

  subroutine copy_theta_c2b(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use lambdam, only: thetamld, nsitemld, nsubmld, blckmld
    use block_ltm, only: nblock

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_the(:)
    integer :: n, err
    integer :: i, j

    n = blade_get_lambda_count(system)

    allocate(blade_the(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_theta_c2b', &
            'memory allocation error')
    end if

    if (n .ne. nblock) then
       call wrndie(-5, 'copy_theta_c2b', &
            'number of blocks differ between blade and charmm')
    end if

    do i = 2, nsitemld
       do j = 1, nsubmld(i)
          blade_the(blckmld(i,j))=thetamld(i,j)
       enddo
    enddo

    n = charmm_send_theta(system, blade_the)

    deallocate(blade_the, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_theta_c2b', &
            'memory de-allocation error')
    end if
  end subroutine copy_theta_c2b

  subroutine copy_thetavelocity_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use lambdam, only: thetavmld, nsitemld, nsubmld, blckmld
    use block_ltm, only: nblock

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_the(:)
    integer :: n, err
    integer :: i, j

    n = blade_get_lambda_count(system)

    allocate(blade_the(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_thetavelocity_b2c', &
            'memory allocation error')
    end if

    n = charmm_recv_thetavelocity(system, blade_the)

    if (n .ne. nblock) then
       call wrndie(-5, 'copy_thetavelocity_b2c', &
            'number of blocks differ between blade and charmm')
    end if

    ! bixlam(1:n) = blade_the(1:n) ! so much easier for lambda
    do i = 2, nsitemld
       do j = 1, nsubmld(i)
          thetavmld(i,j)=blade_the(blckmld(i,j))
       enddo
    enddo

    deallocate(blade_the, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_thetavelocity_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_thetavelocity_b2c

  subroutine copy_thetavelocity_c2b(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double

    use lambdam, only: thetavmld, nsitemld, nsubmld, blckmld
    use block_ltm, only: nblock

    implicit none

    type(c_ptr) :: system

    real(c_double), allocatable :: blade_the(:)
    integer :: n, err
    integer :: i, j

    n = blade_get_lambda_count(system)

    allocate(blade_the(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_thetavelocity_c2b', &
            'memory allocation error')
    end if

    if (n .ne. nblock) then
       call wrndie(-5, 'copy_thetavelocity_c2b', &
            'number of blocks differ between blade and charmm')
    end if

    do i = 2, nsitemld
       do j = 1, nsubmld(i)
          blade_the(blckmld(i,j))=thetavmld(i,j)
       enddo
    enddo

    n = charmm_send_thetavelocity(system, blade_the)

    deallocate(blade_the, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_thetavelocity_c2b', &
            'memory de-allocation error')
    end if
  end subroutine copy_thetavelocity_c2b
#endif /* KEY_BLOCK */

  subroutine copy_box_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use image, only: xucell, xtlabc, qorth, xtlrot

    implicit none

    type(c_ptr) :: system
    real(c_double) :: blade_box(6)

    call charmm_recv_box(system, blade_box)
    xucell(1:6)=blade_box(1:6)
    call xtlaxs(xtlabc,xucell)
    call xtlaxsacc(qorth,xtlrot,xucell,xtlabc)
    call xtlmsr(xucell)
  end subroutine copy_box_b2c

  subroutine copy_box_c2b(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use image, only: xucell

    implicit none

    type(c_ptr) :: system
    real(c_double) :: blade_box(6)

    blade_box(1:6)=xucell(1:6)
    call charmm_send_box(system, blade_box)
  end subroutine copy_box_c2b

  subroutine copy_energy_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use energym

    implicit none
    type(c_ptr) :: system

    real(c_double), allocatable :: blade_energy(:)
    integer :: n, err

    n = blade_get_energy_count()

    allocate(blade_energy(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_energy_b2c', &
            'memory allocation error')
    end if

    call charmm_recv_energy(system, blade_energy)

    eterm = 0
    eterm(bond)=blade_energy(1) ! eebond
    eterm(angle)=blade_energy(2) ! eeangle
    eterm(ureyb)=blade_energy(3) ! eeurey
    eterm(dihe)=blade_energy(4) ! eedihe
    eterm(imdihe)=blade_energy(5) ! eeimpr
    eterm(cmap)=blade_energy(6) ! eecmap
    ! Put both elec and vdw into vdw because it's 15% slower to write kernels that separate them
    ! eterm(elec)=blade_energy(7) ! eeelec
    ! eterm(vdw)=blade_energy(8) ! eevdw
    eterm(vdw)=blade_energy(7)+blade_energy(8) !   eenb14 + eenbdirect
    eterm(ewksum)=blade_energy(9) ! eenbrecip
    eterm(ewself)=blade_energy(10) ! eenbrecipself
    eterm(ewexcl)=blade_energy(11) ! eenbrecipexcl
    ! eelambda - missing - msld biasing potentials
    ! eebias - missing - BLOCK CATS restraints and total charge restraints
    eprop(epot)=blade_energy(14) ! eepotential
    eprop(totke)=blade_energy(15) ! eekinetic
    eprop(tote)=blade_energy(16) ! eetotal

    deallocate(blade_energy, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_energy_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_energy_b2c

  subroutine copy_force_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double
    use deriv, only: dx, dy, dz
    use image, only: qorth, xtlrot

#if KEY_BLOCK == 1
    use lambdam, only: qldm, biflam, qmld, biflam2
#endif

    implicit none
    type(c_ptr) :: system

    real(c_double), allocatable :: blade_force(:)
    real(c_double), allocatable :: blade_flambda(:)
    integer :: n, nelts, nL, err

    n = blade_get_atom_count(system)
    nelts = 3*n
    nL = blade_get_lambda_count(system)

    allocate(blade_force(3*n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_force_b2c', &
            'memory allocation error')
    end if

    allocate(blade_flambda(n), stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_force_b2c', &
            'memory allocation error')
    end if

    call charmm_recv_force(system, blade_force, blade_flambda)

    if (qorth) then
       dx(1:n) = blade_force(1:nelts:3)
       dy(1:n) = blade_force(2:nelts:3)
       dz(1:n) = blade_force(3:nelts:3)
    else
       dx(1:n) = xtlrot(1,1)*blade_force(1:nelts:3)+xtlrot(2,1)*blade_force(2:nelts:3)+xtlrot(3,1)*blade_force(3:nelts:3)
       dy(1:n) = xtlrot(1,2)*blade_force(1:nelts:3)+xtlrot(2,2)*blade_force(2:nelts:3)+xtlrot(3,2)*blade_force(3:nelts:3)
       dz(1:n) = xtlrot(1,3)*blade_force(1:nelts:3)+xtlrot(2,3)*blade_force(2:nelts:3)+xtlrot(3,3)*blade_force(3:nelts:3)
    endif

#if KEY_BLOCK == 1
    if (qldm) biflam(1:nL) = blade_flambda(1:nL)

    if (qmld) biflam2(1:nL) = 0
#endif /* KEY_BLOCK */

    deallocate(blade_force, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_force_b2c', &
            'memory de-allocation error')
    end if

    deallocate(blade_flambda, stat=err)
    if (err .ne. 0) then
       call wrndie(-5, 'copy_force_b2c', &
            'memory de-allocation error')
    end if
  end subroutine copy_force_b2c

  subroutine copy_state_b2c(system,vx,vy,vz)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use chm_kinds
    use lambdam, only: qmld

    implicit none

    type(c_ptr) :: system
    real(chm_real), intent(inout) :: vx(:), vy(:), vz(:)

    call blade_recv_state(system)

    call blade_prettify_position(system)

    call copy_box_b2c(system) ! Need to receive box first to get rotation matrix
    call copy_coords_b2c(system)
    call copy_velocity_b2c(system,vx,vy,vz)
    if (qmld) call copy_theta_b2c(system)
    if (qmld) call copy_thetavelocity_b2c(system)
  end subroutine copy_state_b2c

  subroutine copy_state_c2b(system,vx,vy,vz)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use chm_kinds
    use lambdam, only: qmld

    implicit none

    type(c_ptr) :: system
    real(chm_real), intent(inout) :: vx(:), vy(:), vz(:)

    call copy_box_c2b(system)
    call copy_coords_c2b(system)
    call copy_velocity_c2b(system,vx,vy,vz)
    if (qmld) call copy_theta_c2b(system)
    if (qmld) call copy_thetavelocity_c2b(system)

    call blade_send_state(system)

    call blade_init_lambda_from_theta(system)
  end subroutine copy_state_c2b

  subroutine copy_spatial_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr

    implicit none

    type(c_ptr) :: system

    call blade_recv_position(system)
    call blade_prettify_position(system)
    call copy_box_b2c(system) ! Need to receive box first to get rotation matrix
    call copy_coords_b2c(system)
  end subroutine copy_spatial_b2c

  subroutine copy_alchemical_b2c(system)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use lambdam, only: qmld

    implicit none

    type(c_ptr) :: system

    call blade_recv_theta(system)
    if (qmld) call copy_theta_b2c(system)
  end subroutine copy_alchemical_b2c
#endif /* KEY_BLADE */

end module blade_coords_module
