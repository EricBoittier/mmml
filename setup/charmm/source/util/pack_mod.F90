module pack_mod
  use, intrinsic :: iso_c_binding
  implicit none

  private

  ! Public C subroutines
  public pack_int_double, pack_int_byte
  public pack_double_double, pack_double_byte
  public pack_double_zeros
  public unpack_int_double, unpack_int_byte
  public unpack_double_double, unpack_double_byte
  public pack_xyz_double, pack_xyz_byte, pack_xyz_atom_byte
  public unpack_xyz_group, unpack_xyz_atom
  public pack_coord
  public unpack_coord
  public unpack_force

  interface

     subroutine pack_int_double(in, inlen, out, outpos) bind(C)
       import
       integer(c_int), intent(in) :: in, inlen
       real(c_double), intent(inout) :: out
       integer(c_int), intent(inout) :: outpos
     end subroutine pack_int_double

     subroutine pack_int_byte(in, inlen, out, outpos) bind(C)
       import
       integer(c_int), intent(in) :: in, inlen
       integer(c_int8_t), intent(inout) :: out
       integer(c_int), intent(inout) :: outpos
     end subroutine pack_int_byte

     subroutine pack_double_double(in, inlen, out, outpos) bind(C)
       import
       real(c_double), intent(in) :: in
       integer(c_int), intent(in) :: inlen
       real(c_double), intent(inout) :: out
       integer(c_int), intent(inout) :: outpos
     end subroutine pack_double_double

     subroutine pack_double_byte(in, inlen, out, outpos) bind(C)
       import
       real(c_double), intent(in) :: in
       integer(c_int), intent(in) :: inlen
       integer(c_int8_t), intent(inout) :: out
       integer(c_int), intent(inout) :: outpos
     end subroutine pack_double_byte

     subroutine pack_double_zeros(inlen, out, outpos) bind(C)
       import
       integer(c_int), intent(in) :: inlen
       integer(c_int8_t), intent(inout) :: out
       integer(c_int), intent(inout) :: outpos
     end subroutine pack_double_zeros

     subroutine unpack_int_double(in, inpos, out, outlen) bind(C)
       import
       real(c_double), intent(in) :: in
       integer(c_int), intent(in) :: inpos
       integer(c_int), intent(inout) :: out
       integer(c_int), intent(in) :: outlen
     end subroutine unpack_int_double

     subroutine unpack_int_byte(in, inpos, out, outlen) bind(C)
       import
       integer(c_int8_t), intent(in) :: in
       integer(c_int), intent(in) :: inpos
       integer(c_int), intent(inout) :: out
       integer(c_int), intent(in) :: outlen
     end subroutine unpack_int_byte

     subroutine unpack_double_double(in, inpos, out, outlen) bind(C)
       import
       real(c_double), intent(in) :: in
       integer(c_int), intent(inout) :: inpos
       real(c_double), intent(inout) :: out
       integer(c_int), intent(in) :: outlen
     end subroutine unpack_double_double

     subroutine unpack_double_byte(in, inpos, out, outlen) bind(C)
       import
       integer(c_int8_t), intent(in) :: in
       integer(c_int), intent(inout) :: inpos
       real(c_double), intent(inout) :: out
       integer(c_int), intent(in) :: outlen
     end subroutine unpack_double_byte

     subroutine pack_xyz_double(ngroupl, groupl, group, buffer, k, x, y, z) bind(C)
       import
       integer(c_int), intent(in) :: ngroupl, groupl(*), group(*)
       real(c_double), intent(inout) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(in) :: x(*), y(*), z(*)
     end subroutine pack_xyz_double

     subroutine pack_xyz_byte(ngroupl, groupl, group, buffer, k, x, y, z) bind(C)
       import
       integer(c_int), intent(in) :: ngroupl, groupl(*), group(*)
       integer(c_int8_t), intent(inout) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(in) :: x(*), y(*), z(*)
     end subroutine pack_xyz_byte

     subroutine pack_xyz_atom_byte(natoml, atoml, buffer, k, x, y, z) bind(C)
       import
       integer(c_int), intent(in) :: natoml, atoml(*)
       integer(c_int8_t), intent(inout) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(in) :: x(*), y(*), z(*)
     end subroutine pack_xyz_atom_byte

     subroutine unpack_xyz_group(ngroupl, groupl, group, buffer, k, x, y, z, natom, atomlist) &
          bind(C)
       import
       integer(c_int), intent(in) :: ngroupl, groupl(*), group(*)
       integer(c_int8_t), intent(in) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(inout) :: x(*), y(*), z(*)
       integer(c_int), intent(inout) :: natom, atomlist(*)
     end subroutine unpack_xyz_group

     subroutine unpack_xyz_atom(natoml, atoml, buffer, k, x, y, z, atomlist) &
          bind(C)
       import
       integer(c_int), intent(in) :: natoml, atoml(*)
       integer(c_int8_t), intent(in) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(inout) :: x(*), y(*), z(*)
       integer(c_int), intent(inout) :: atomlist(*)
     end subroutine unpack_xyz_atom

     subroutine pack_coord(ngroupl, groupl, grouppos, group, buffer, k, x, y, z) bind(C)
       import
       integer(c_int), intent(in) :: ngroupl, groupl(*), grouppos(*), group(*)
       real(c_double), intent(inout) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(in) :: x(*), y(*), z(*)
     end subroutine pack_coord

     subroutine unpack_coord(ngroupl, groupl, group, buffer, k, x, y, z) bind(C)
       import
       integer(c_int), intent(in) :: ngroupl, groupl(*), group(*)
       real(c_double), intent(in) :: buffer
       integer(c_int), intent(inout) :: k
       real(c_double), intent(inout) :: x(*), y(*), z(*)
     end subroutine unpack_coord

     subroutine unpack_force(natoml, atoml, k, buffer, forcex, forcey, forcez) bind(C)
       import
       integer(c_int), intent(in) :: natoml, atoml(*)
       integer(c_int), intent(inout) :: k
       integer(c_int8_t), intent(in) :: buffer
       real(c_double), intent(inout) :: forcex(*), forcey(*), forcez(*)
     end subroutine unpack_force

  end interface

end module pack_mod
