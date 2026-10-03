module omm_torch
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
  implicit none

#ifdef KEY_OMMTORCH
  interface
     function torch_create(filename) bind(c) result(new_force)
       use, intrinsic :: iso_c_binding, only: c_char, c_ptr
       implicit none
       character(len=1, kind=c_char) :: filename(*)
       type(c_ptr) :: new_force
     end function torch_create

     subroutine torch_set_uses_pbc(force) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: force
     end subroutine torch_set_uses_pbc

     subroutine torch_set_outputs_forces(force) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: force
     end subroutine torch_set_outputs_forces

     function torch_add_global_param(force, name, val) &
          bind(c) result(param_index)
       use, intrinsic :: iso_c_binding, only: &
            c_char, c_double, c_int, c_ptr
       implicit none
       type(c_ptr), value :: force
       character(len=1, kind=c_char) :: name(*)
       real(c_double), value :: val
       integer(c_int) :: param_index
     end function torch_add_global_param

     subroutine torch_set_global_param(force, param_index, val) bind(c)
       use, intrinsic :: iso_c_binding, only: &
            c_char, c_double, c_int, c_ptr
       implicit none
       type(c_ptr), value :: force
       integer(c_int), value :: param_index
       real(c_double), value :: val
     end subroutine torch_set_global_param
  end interface
#endif  /* KEY_OMMTORCH */
end module
