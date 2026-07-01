module colfft_types

  !
  ! Type definitions for colfft
  !

  use chm_kinds
  implicit none
  public

  type q_grid_t
     real(chm_real), allocatable, dimension(:) :: array_dp
     real(chm_real4), allocatable, dimension(:) :: array_sp
     integer x0, x1, y0, y1, z0, z1         ! Region boundary
     integer xlo, xhi, ylo, yhi, zlo, zhi   ! Writing region
     integer xsize, ysize, zsize            ! Writing region sizes
     integer tot_size                       ! Total size
     integer xgridlo, xgridhi, ygridlo, ygridhi, zgridlo, zgridhi ! Grid point bound., for testing
     integer tx, ty, tz
  end type q_grid_t

end module colfft_types
