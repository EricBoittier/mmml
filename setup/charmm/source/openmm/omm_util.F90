module omm_util
#if KEY_OPENMM==1
   implicit none

    ! See ommPlus.cpp for the implementation of this helper interface
    ! functionality not provided by OpenMM default fortran wrapper
    interface
        function omm_group_potential(context, pbox, groups)
          use OpenMM

          implicit none

          type (OpenMM_Context), intent(in) :: context
          integer*4, intent(in) :: pbox
          integer*4, intent(in) :: groups

          real*8 :: omm_group_potential
        end function omm_group_potential

        function omm_platform_getPlatformByName(name, result)
          use OpenMM

          implicit none

          character(*) name
          type (OpenMM_Platform) result

          integer*4 :: omm_platform_getplatformbyname
        end function
    end interface

contains

   !> Initializes an OpenMM_DoubleArray, working around an OpenMM 4.0 bug.
   ! retained to take array of new elements
   subroutine omm_param_set(params, mdata)
     use chm_kinds, only: chm_real
     use OpenMM, only: OpenMM_DoubleArray, OpenMM_DoubleArray_set

     implicit none

     type (OpenMM_DoubleArray), intent(in) :: params
     real(chm_real), intent(in) :: mdata(:)
     integer i

     do i = 1, size(mdata)
       call OpenMM_DoubleArray_set(params, i, mdata(i))
     end do
   end subroutine omm_param_set
#endif
end module omm_util
