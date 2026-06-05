module api_custom
  implicit none

  private

  interface
     function callback(current_step, natoms, &
          vx, vy, vz, &
          xnew, ynew, znew, &
          xold, yold, zold) result(ret_val) bind(c)
       use, intrinsic :: iso_c_binding, only: c_double, c_int
       implicit none
       integer(c_int), value :: current_step, natoms
       real(c_double) :: ret_val
       real(c_double), dimension(*) :: &
            vx, vy, vz, &
            xnew, ynew, znew, &
            xold, yold, zold
     end function callback
  end interface

  procedure(callback), pointer :: dynam_func

  public :: &
       custom_dynam_set, custom_dynam_unset, &
       custom_dynam_is_set, &
       custom_dynam_call

contains

  subroutine custom_dynam_set(new_func) bind(c)
    implicit none
    procedure(callback) :: new_func
    dynam_func => new_func
  end subroutine custom_dynam_set

  subroutine custom_dynam_unset() bind(c)
    implicit none
    dynam_func => null()
  end subroutine custom_dynam_unset

  function custom_dynam_call(current_step, natoms, &
            vx, vy, vz, &
            xnew, ynew, znew, &
            xold, yold, zold) result(ret_val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none
    integer(c_int), value :: current_step, natoms
    real(c_double) :: ret_val
    real(c_double), dimension(*) :: &
         vx, vy, vz, &
         xnew, ynew, znew, &
         xold, yold, zold

    ret_val = dynam_func(current_step, natoms, &
         vx, vy, vz, &
         xnew, ynew, znew, &
         xold, yold, zold)
  end function custom_dynam_call

  logical function custom_dynam_is_set()
    implicit none
    custom_dynam_is_set = associated(dynam_func)
  end function custom_dynam_is_set

end module api_custom
