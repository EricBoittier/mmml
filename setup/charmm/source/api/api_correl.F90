!> @brief Stateless geometry and trajectory I/O functions for analysis
!
! Provides bind(c) functions for:
! - Computing distances, angles, and dihedrals (math follows GETICV)
! - Reading DCD trajectory files directly (bypasses lingo layer)
!
! All state lives on the Python side; these are pure math + I/O.
! The DCD reader uses a single saved file unit for sequential access.
module api_correl
  use, intrinsic :: iso_c_binding, only: c_int, c_double, c_char
  use consta, only: raddeg, cosmax, pi
  use chm_kinds, only: chm_real4

  implicit none

  !> @brief Private state for the DCD reader.
  !
  ! Only one DCD file can be open at a time through this API.
  ! The Fortran unit, atom count, and flags are saved between calls.
  !
  ! NOTE: dcd_unit is assigned by OPEN(NEWUNIT=...), which returns a
  ! *negative* value.  "Is a file open?" must therefore be tracked with
  ! dcd_is_open, never by testing the sign of dcd_unit.
  integer, private, save :: dcd_unit = 0
  integer, private, save :: dcd_natom = 0
  logical, private, save :: dcd_has_crystal = .false.
  ! CHEQ trajectories (ICNTRL(13) == 1) write an extra per-frame charge
  ! record after Z; read_frame() must consume it to stay frame-aligned.
  logical, private, save :: dcd_has_cheq = .false.
  logical, private, save :: dcd_is_open = .false.
  ! Single-precision scratch buffer for one coordinate record, allocated
  ! once per open() (sized to dcd_natom) and reused by every read_frame()
  ! call, then freed by close() — avoids an allocate/deallocate per frame.
  real(chm_real4), private, save, allocatable :: dcd_temp(:)

contains

  !> @brief find atom index by residue number and atom name
  !
  !> @param[in] res_num residue number (1-based)
  !> @param[in] atom_name null-terminated C string
  !> @return atom index (1-based) or 0 if not found
  integer(c_int) function correl_find_atom(res_num, atom_name) bind(c)
    use api_util, only: c2f_string
    use api_ic, only: find_atom_index

    implicit none

    integer(c_int), intent(in) :: res_num
    character(kind=c_char), intent(in) :: atom_name(*)

    character(len=8) :: fname

    fname = c2f_string(atom_name, 8)
    correl_find_atom = find_atom_index(res_num, fname)
  end function correl_find_atom

  !> @brief compute distance between atoms i and j
  !
  !> @param[in] x, y, z coordinate arrays (natom elements)
  !> @param[in] i, j atom indices (1-based)
  !> @return distance in Angstroms
  real(c_double) function correl_distance(x, y, z, i, j) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: i, j

    real(c_double) :: dx, dy, dz

    dx = x(i) - x(j)
    dy = y(i) - y(j)
    dz = z(i) - z(j)
    correl_distance = sqrt(dx*dx + dy*dy + dz*dz)
  end function correl_distance

  !> @brief compute angle i-j-k in degrees
  !
  !> @param[in] x, y, z coordinate arrays (natom elements)
  !> @param[in] i, j, k atom indices (1-based), j is the vertex
  !> @return angle in degrees (0..180)
  real(c_double) function correl_angle(x, y, z, i, j, k) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: i, j, k

    real(c_double) :: fx, fy, fz, gx, gy, gz
    real(c_double) :: fr, gr, cst

    ! vectors j->i and j->k
    fx = x(i) - x(j)
    fy = y(i) - y(j)
    fz = z(i) - z(j)

    gx = x(k) - x(j)
    gy = y(k) - y(j)
    gz = z(k) - z(j)

    fr = sqrt(fx*fx + fy*fy + fz*fz)
    gr = sqrt(gx*gx + gy*gy + gz*gz)

    if (fr < 1.0d-10) fr = 1.0d-10
    if (gr < 1.0d-10) gr = 1.0d-10
    cst = (fx*gx + fy*gy + fz*gz) / (fr * gr)
    if (abs(cst) >= 1.0d0) cst = sign(1.0d0, cst)

    correl_angle = acos(cst) * raddeg
  end function correl_angle

  !> @brief compute dihedral angle i-j-k-l in degrees
  !
  ! Sign convention matches GETICV: positive = right-hand rule around j-k bond.
  !
  !> @param[in] x, y, z coordinate arrays (natom elements)
  !> @param[in] i, j, k, l atom indices (1-based)
  !> @return dihedral angle in degrees (-180..180)
  real(c_double) function correl_dihedral(x, y, z, i, j, k, l) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: i, j, k, l

    real(c_double) :: fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(c_double) :: ax, ay, az, bx, by, bz
    real(c_double) :: ra, rb, cst, phi
    real(c_double) :: cx, cy, cz

    ! bond vectors: f = i->j, g = j->k (central bond), h = l->k
    fx = x(i) - x(j)
    fy = y(i) - y(j)
    fz = z(i) - z(j)

    gx = x(j) - x(k)
    gy = y(j) - y(k)
    gz = z(j) - z(k)

    hx = x(l) - x(k)
    hy = y(l) - y(k)
    hz = z(l) - z(k)

    ! normal vectors via cross products (matches GETICV)
    ax = fy*gz - fz*gy
    ay = fz*gx - fx*gz
    az = fx*gy - fy*gx

    bx = hy*gz - hz*gy
    by = hz*gx - hx*gz
    bz = hx*gy - hy*gx

    ra = sqrt(ax*ax + ay*ay + az*az)
    rb = sqrt(bx*bx + by*by + bz*bz)

    if (ra < 1.0d-10) ra = 1.0d-10
    if (rb < 1.0d-10) rb = 1.0d-10

    ! normalize
    ax = ax / ra;  ay = ay / ra;  az = az / ra
    bx = bx / rb;  by = by / rb;  bz = bz / rb

    cst = ax*bx + ay*by + az*bz
    if (abs(cst) >= 1.0d0) cst = sign(1.0d0, cst)
    phi = acos(cst)

    ! sign from triple product (matches GETICV convention)
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    if (gx*cx + gy*cy + gz*cz > 0.0d0) phi = -phi

    correl_dihedral = phi * raddeg
  end function correl_dihedral

  !> @brief compute distance series for all frames
  !
  !> @param[in] x, y, z coordinate arrays (natom*nframes), frame-major
  !> @param[in] natom number of atoms per frame
  !> @param[in] nframes number of frames
  !> @param[in] i, j atom indices (1-based)
  !> @param[out] out output array (nframes elements)
  !> @return 1 on success
  integer(c_int) function correl_distance_series(x, y, z, natom, nframes, &
       i, j, out) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: natom, nframes, i, j
    real(c_double), intent(out) :: out(*)

    integer :: f, off
    real(c_double) :: dx, dy, dz

    do f = 1, nframes
       off = (f - 1) * natom
       dx = x(off + i) - x(off + j)
       dy = y(off + i) - y(off + j)
       dz = z(off + i) - z(off + j)
       out(f) = sqrt(dx*dx + dy*dy + dz*dz)
    end do

    correl_distance_series = 1
  end function correl_distance_series

  !> @brief compute angle series for all frames
  !
  !> @param[in] x, y, z coordinate arrays (natom*nframes), frame-major
  !> @param[in] natom number of atoms per frame
  !> @param[in] nframes number of frames
  !> @param[in] i, j, k atom indices (1-based), j is vertex
  !> @param[out] out output array (nframes elements), degrees
  !> @return 1 on success
  integer(c_int) function correl_angle_series(x, y, z, natom, nframes, &
       i, j, k, out) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: natom, nframes, i, j, k
    real(c_double), intent(out) :: out(*)

    integer :: f, off
    real(c_double) :: fx, fy, fz, gx, gy, gz, fr, gr, cst

    do f = 1, nframes
       off = (f - 1) * natom
       fx = x(off + i) - x(off + j)
       fy = y(off + i) - y(off + j)
       fz = z(off + i) - z(off + j)

       gx = x(off + k) - x(off + j)
       gy = y(off + k) - y(off + j)
       gz = z(off + k) - z(off + j)

       fr = sqrt(fx*fx + fy*fy + fz*fz)
       gr = sqrt(gx*gx + gy*gy + gz*gz)

       if (fr < 1.0d-10) fr = 1.0d-10
       if (gr < 1.0d-10) gr = 1.0d-10
       cst = (fx*gx + fy*gy + fz*gz) / (fr * gr)
       if (abs(cst) >= 1.0d0) cst = sign(1.0d0, cst)

       out(f) = acos(cst) * raddeg
    end do

    correl_angle_series = 1
  end function correl_angle_series

  !> @brief compute dihedral series for all frames
  !
  !> @param[in] x, y, z coordinate arrays (natom*nframes), frame-major
  !> @param[in] natom number of atoms per frame
  !> @param[in] nframes number of frames
  !> @param[in] i, j, k, l atom indices (1-based)
  !> @param[out] out output array (nframes elements), degrees (-180..180)
  !> @return 1 on success
  integer(c_int) function correl_dihedral_series(x, y, z, natom, nframes, &
       i, j, k, l, out) bind(c)
    implicit none

    real(c_double), intent(in) :: x(*), y(*), z(*)
    integer(c_int), intent(in), value :: natom, nframes, i, j, k, l
    real(c_double), intent(out) :: out(*)

    integer :: f, off
    real(c_double) :: fx, fy, fz, gx, gy, gz, hx, hy, hz
    real(c_double) :: ax, ay, az, bx, by, bz
    real(c_double) :: ra, rb, cst, phi
    real(c_double) :: cx, cy, cz

    do f = 1, nframes
       off = (f - 1) * natom

       fx = x(off + i) - x(off + j)
       fy = y(off + i) - y(off + j)
       fz = z(off + i) - z(off + j)

       gx = x(off + j) - x(off + k)
       gy = y(off + j) - y(off + k)
       gz = z(off + j) - z(off + k)

       hx = x(off + l) - x(off + k)
       hy = y(off + l) - y(off + k)
       hz = z(off + l) - z(off + k)

       ax = fy*gz - fz*gy
       ay = fz*gx - fx*gz
       az = fx*gy - fy*gx

       bx = hy*gz - hz*gy
       by = hz*gx - hx*gz
       bz = hx*gy - hy*gx

       ra = sqrt(ax*ax + ay*ay + az*az)
       rb = sqrt(bx*bx + by*by + bz*bz)

       if (ra < 1.0d-10) ra = 1.0d-10
       if (rb < 1.0d-10) rb = 1.0d-10

       ax = ax / ra;  ay = ay / ra;  az = az / ra
       bx = bx / rb;  by = by / rb;  bz = bz / rb

       cst = ax*bx + ay*by + az*bz
       if (abs(cst) >= 1.0d0) cst = sign(1.0d0, cst)
       phi = acos(cst)

       cx = ay*bz - az*by
       cy = az*bx - ax*bz
       cz = ax*by - ay*bx
       if (gx*cx + gy*cy + gz*cz > 0.0d0) phi = -phi

       out(f) = phi * raddeg
    end do

    correl_dihedral_series = 1
  end function correl_dihedral_series

  ! ==================================================================
  ! DCD trajectory reader — direct binary I/O, no lingo overhead
  ! ==================================================================

  !> @brief Open a DCD file and read its header.
  !
  ! Reads the ICNTRL header, title, and atom count. The file remains
  ! open for subsequent correl_dcd_read_frame() calls.
  !
  !> @param[in]  path       null-terminated file path
  !> @param[out] natom_out  number of atoms per frame
  !> @param[out] nframes_out number of frames in file (from header)
  !> @param[out] delta_out  time step in AKMA units
  !> @param[out] skip_out   saving interval (steps between frames)
  !> @param[out] istep1_out step number of the first stored frame (ICNTRL(2)).
  !!             Frame i (1-based) is at step istep1_out + (i-1)*skip_out.
  !> @return 1 on success; -1 on I/O failure; -2 if the trajectory uses a
  !!         variant this direct reader does not handle (fixed atoms or a
  !!         4D trajectory), so the caller can fall back to the lingo path.
  integer(c_int) function correl_dcd_open(path, natom_out, nframes_out, &
       delta_out, skip_out, istep1_out) bind(c)
    use chm_kinds
    use api_util, only: c2f_string

    implicit none

    character(kind=c_char), intent(in) :: path(*)
    integer(c_int), intent(out) :: natom_out, nframes_out, skip_out, istep1_out
    real(c_double), intent(out) :: delta_out

    character(len=512) :: fpath
    character(len=4) :: hdr
    integer :: icntrl(20), ios, ntitle, i
    character(len=80) :: title_line
    real(chm_real4) :: delta4

    correl_dcd_open = -1

    ! close any previously open file
    if (dcd_is_open) close(dcd_unit, iostat=ios)
    dcd_is_open = .false.
    if (allocated(dcd_temp)) deallocate(dcd_temp)

    fpath = c2f_string(path, 512)

    open(newunit=dcd_unit, file=trim(fpath), status='old', &
         form='unformatted', access='sequential', iostat=ios)
    if (ios /= 0) then
       dcd_unit = 0
       return
    end if

    ! Record 1: header + ICNTRL
    read(dcd_unit, iostat=ios) hdr, icntrl
    if (ios /= 0) then
       close(dcd_unit); dcd_unit = 0; return
    end if

    ! This reader handles the full X/Y/Z-per-frame layout, with or without
    ! the leading crystal record (ICNTRL(11)) and with or without a
    ! trailing CHEQ charge record (ICNTRL(13), skipped in read_frame).
    ! Fixed atoms (ICNTRL(9) /= 0 -> a FREEAT index record plus
    ! free-atom-only frames) and 4D trajectories (ICNTRL(12) == 1 -> an
    ! extra per-frame coordinate record interleaved before any CHEQ record)
    ! need the full READCV remapping, so bail with -2 and let the caller
    ! use lingo.
    if (icntrl(9) /= 0 .or. icntrl(12) == 1) then
       close(dcd_unit); dcd_unit = 0
       correl_dcd_open = -2
       return
    end if

    ! Record 2: titles
    read(dcd_unit, iostat=ios) ntitle, (title_line, i=1, ntitle)
    if (ios /= 0) then
       close(dcd_unit); dcd_unit = 0; return
    end if

    ! Record 3: natom
    read(dcd_unit, iostat=ios) dcd_natom
    if (ios /= 0) then
       close(dcd_unit); dcd_unit = 0; return
    end if

    ! Decode header fields
    nframes_out = icntrl(1)
    istep1_out = icntrl(2)
    skip_out = icntrl(3)
    dcd_has_crystal = (icntrl(11) == 1)
    ! CHEQ (fluctuating-charge) trajectories append one charge record per
    ! frame; read_frame() skips it so the X/Y/Z stream stays aligned.
    dcd_has_cheq = (icntrl(13) == 1)

    ! Delta is stored as transferred real*4 bits
    delta4 = transfer(icntrl(10), delta4)
    delta_out = real(delta4, c_double)

    natom_out = dcd_natom
    ! One-time scratch buffer reused by every read_frame() call.
    allocate(dcd_temp(dcd_natom))
    dcd_is_open = .true.
    correl_dcd_open = 1
  end function correl_dcd_open

  !> @brief Read the next frame from the open DCD file.
  !
  ! Reads x, y, z coordinate arrays (single precision in file,
  ! converted to double precision on output). Handles optional
  ! crystal record.
  !
  !> @param[out] x  x-coordinates (natom elements, double precision)
  !> @param[out] y  y-coordinates (natom elements, double precision)
  !> @param[out] z  z-coordinates (natom elements, double precision)
  !> @return 1 on success, 0 at end of file, -1 on error
  integer(c_int) function correl_dcd_read_frame(x, y, z) bind(c)
    use chm_kinds

    implicit none

    real(c_double), intent(out) :: x(*), y(*), z(*)

    real(chm_real8) :: xtlabc(6)
    integer :: ios, i

    correl_dcd_read_frame = -1

    if (.not. dcd_is_open .or. dcd_natom <= 0) return

    ! Optional crystal unit cell record
    if (dcd_has_crystal) then
       read(dcd_unit, iostat=ios) xtlabc
       if (ios /= 0) then
          correl_dcd_read_frame = 0  ! EOF
          return
       end if
    end if

    ! X coordinates
    read(dcd_unit, iostat=ios) (dcd_temp(i), i=1, dcd_natom)
    if (ios /= 0) then
       correl_dcd_read_frame = 0
       return
    end if
    do i = 1, dcd_natom
       x(i) = real(dcd_temp(i), c_double)
    end do

    ! Y coordinates
    read(dcd_unit, iostat=ios) (dcd_temp(i), i=1, dcd_natom)
    if (ios /= 0) then
       correl_dcd_read_frame = -1
       return
    end if
    do i = 1, dcd_natom
       y(i) = real(dcd_temp(i), c_double)
    end do

    ! Z coordinates
    read(dcd_unit, iostat=ios) (dcd_temp(i), i=1, dcd_natom)
    if (ios /= 0) then
       correl_dcd_read_frame = -1
       return
    end if
    do i = 1, dcd_natom
       z(i) = real(dcd_temp(i), c_double)
    end do

    ! CHEQ trajectories store an extra per-frame charge record after Z.
    ! We do not use the charges, but must consume the record to keep the
    ! stream aligned for the next frame; otherwise X/Y/Z would be read
    ! from the wrong records and eventually fail or return garbage.
    if (dcd_has_cheq) then
       read(dcd_unit, iostat=ios) (dcd_temp(i), i=1, dcd_natom)
       if (ios /= 0) then
          correl_dcd_read_frame = -1
          return
       end if
    end if

    correl_dcd_read_frame = 1
  end function correl_dcd_read_frame

  !> @brief Close the open DCD file.
  !
  !> @return 1 on success
  integer(c_int) function correl_dcd_close() bind(c)
    implicit none

    integer :: ios

    correl_dcd_close = 1
    if (dcd_is_open) then
       close(dcd_unit, iostat=ios)
       dcd_unit = 0
       dcd_natom = 0
       dcd_has_crystal = .false.
       dcd_has_cheq = .false.
       dcd_is_open = .false.
    end if
    if (allocated(dcd_temp)) deallocate(dcd_temp)
  end function correl_dcd_close

end module api_correl
